import cv2
import numpy as np
import argparse
import imgaug as ia
import torch
import pandas as pd
import os
import glob
from tqdm import tqdm

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels
from PIL import Image

import functools
from multiprocessing import Pool

import traceback
import sys
import warnings

#from tqdm.contrib.concurrent import process_map
#from p_tqdm import p_map
#import time

def bb_midpoint_to_corner(bb):
    label = bb[0]
    x1 = bb[1] - bb[3]/2
    x2 = bb[1] + bb[3]/2
    y1 = bb[2] - bb[4]/2
    y2 = bb[2] + bb[4]/2
    # A: area will only be used for sorting
    area = bb[3]*bb[4]
    corner_list = [label, x1, x2, y1, y2, area]
    return np.array(corner_list)

def open_yolo_sort(path, image_name):
    try:
        image = cv2.imread(path + image_name)
        shape = image.shape
        width = shape[1]
        height = shape[0]
        #print(width, height)
        label = path + os.path.splitext(image_name)[0] + ".txt"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            boxes = np.genfromtxt(label, delimiter=' ')
        bb = boxes
        # reshaping the np array is necessary in case a file with a single box is read
        boxes = boxes.reshape(boxes.size//5, 5)
        #print(boxes.shape)
        boxes = np.apply_along_axis(bb_midpoint_to_corner, axis=1, arr=boxes)
        # A: sorting by area
        boxes = boxes[boxes[:, 5].argsort()]
        # A: reversing the sorted list so bigger areas come first
        boxes = boxes[::-1]
        return image, boxes, width, height
    except Exception as e:
        #print(e)
        #print(image_name)
        return image, None, None, None

def create_segclass(image_path, save_path, image_name):
    image, bb, w, h = open_yolo_sort(image_path, image_name)
    image_copy = image.copy()*0 
    if bb is not None:       
        for label, x1, x2, y1, y2, area in bb:
            # A: the white outline with four pixels of thickness
            #cv2.rectangle(image_copy, (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), (255, 255, 255), 4)
            # A: the class coded filing, specified by -1
            cv2.rectangle(image_copy, (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), colors[int(label)], -1)
        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
        cv2.imwrite(save_path + os.path.splitext(image_name)[0] + ".png", image_copy)
    else:
        cv2.imwrite(save_path + os.path.splitext(image_name)[0] +".png", image_copy)
        
colors = [(0, 0, 0),      # background
         (162, 0, 255),   # Chave seccionadora lamina (Aberta)
         (97, 16, 162),   # Chave seccionadora lamina (Fechada)
         (81, 162, 0),    # Chave seccionadora tandem (Aberta)
         (48, 97, 165),   # Chave seccionadora tandem (Fechada)
         (121, 121, 121), # Disjuntor
         (255, 97, 178),  # Fusivel
         (154, 32, 121),  # Isolador disco de vidro
         (255, 255, 125), # Isolador pino de porcelana
         (162, 243, 162), # Mufla
         (143, 211, 255), # Para-raio
         (40, 0, 186),    # Religador
         (255, 182, 0),   # Transformador
         (138, 138, 0),   # Transformador de Corrente (TC)
         (162, 48, 0)]    # Transformador de Potencial (TP)

colors_rgb = []
for c in colors:
    colors_rgb.append((c[2], c[1], c[0]))
    
# A: Needed to encode the colors as (M, N) labels for the loss.
# Since the loss can't be calculated with a tensor of shape 
# [batch_size, height, width, channels] if the three channels are
# present, we need this to be able to encode the rgb class colors
# into a single value to represent them.
def get_labels():
    """Load the mapping that associates classes with label colors.
       Our electrical substation dataset has 14 objects + background.
    Returns:
        np.ndarray with dimensions (15, 3)
    """
    return np.asarray([(0, 0, 0),       # Background
                       (162, 0, 255),   # Chave seccionadora lamina (Aberta)
                       (97, 16, 162),   # Chave seccionadora lamina (Fechada)
                       (81, 162, 0),    # Chave seccionadora tandem (Aberta)
                       (48, 97, 165),   # Chave seccionadora tandem (Fechada)
                       (121, 121, 121), # Disjuntor
                       (255, 97, 178),  # Fusivel
                       (154, 32, 121),  # Isolador disco de vidro
                       (255, 255, 125), # Isolador pino de porcelana
                       (162, 243, 162), # Mufla
                       (143, 211, 255), # Para-raio
                       (40, 0, 186),    # Religador
                       (255, 182, 0),   # Transformador
                       (138, 138, 0),   # Transformador de Corrente (TC)
                       (162, 48, 0)]    # Transformador de Potencial (TP)
                      )

def decode_target(label_mask):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
    Returns:
        (np.ndarray): the resulting decoded color image.
    """
    label_colors = get_labels()
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for l in range(0, len(label_colors)):
        r[label_mask == l] = label_colors[l, 0]
        g[label_mask == l] = label_colors[l, 1]
        b[label_mask == l] = label_colors[l, 2]
    image = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    image[:, :, 0] = r
    image[:, :, 1] = g
    image[:, :, 2] = b
    return image.astype('uint8')

# A: in order for the mask to go through the loss function, the classes need to be
# represented as a single value, opposed to three channels.
def encode_target(mask):
    """Encode segmentation label images as classes
    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the classes are encoded as colors.
    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int8)
    for i, label in enumerate(get_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = i
    label_mask = label_mask.astype(int)
    return label_mask

# https://www.programcreek.com/python/?code=1044197988%2FSemantic-segmentation-of-remote-sensing-images%2FSemantic-segmentation-of-remote-sensing-images-master%2FCRF.py
def crf(original_image, annotated_image, inference):
    """Applies the denseCRF (Conditional Random Fields) algorithm.
    Args:
        original_image (np.ndarray) : the original image of dimension (M, N, 3).
        annotated_image (np.ndarray): associated segmentation mask 
        image of dimension (M, N, 3), in which the classes are encoded as colors.
    Returns:
        (np.ndarray): segmentation mask of dimension (M, N, 3) after the denseCRF
        filtered the fully connected components.
    """
    # Convert the annotation RGB color to a single integer
    annotated_label = encode_target(annotated_image)
     # Convert 32-bit integer colors to 0, 1, 2, ... labels.
    og_colors, labels = np.unique(annotated_label, return_inverse=True)
#     print(og_colors)
    n_labels = len(set(labels.flat))
    if (n_labels > 1):
        # Setting up the CRF model
        #if use_2d :
        d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)

        # get unary potentials (neg log probability)
        U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
        d.setUnaryEnergy(U)

        # This adds the color-independent term, features are the locations only.
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)

        # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
        d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=original_image,
                           compat=10,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)
        # end of if  
        #Run Inference for 5 steps 
        Q = d.inference(inference)

        # Find out the most probable class for each pixel.
        MAP = np.argmax(Q, axis=0)
        colors, labels = np.unique(MAP, return_inverse=True)
        #print(len(colors))
    #     print(colors)
        # C Convert the map (label) back to the corresponding color and save the image.
        # Note that there is no more "unknown" here, no matter what we had in the first place.
        # A: reshaping the MAP into the dimensions of the image
        MAP = MAP.reshape(original_image.shape[0], original_image.shape[1])
        MAP_copy = MAP.copy()
        # A: this, essentially, is an encoder for the encoder. og_colors has the indexes that go
        # with the decoder. However, the argmax doesn't keep that value, instead replacing it with
        # 0... number of unique colors. But the order is retained, so by mapping the index of each
        # unique color with the og_colors one, it's possible to go back to the correct indexes that
        # will then decode into the associated BGR mask color.
        for l in range(0, len(og_colors)):
            MAP_copy[MAP == l] = og_colors[l]
        colors, labels = np.unique(MAP_copy, return_inverse=True)
    #     print(colors)
        MAP_copy = decode_target(MAP_copy)

        return MAP_copy
    else:
        return annotated_image
    
def box_method(image_path, seg_path, save_path, image_name):
    #print(image_name)
    image, bb, w, h = open_yolo_sort(image_path, image_name)
    seg_image = cv2.imread(seg_path + os.path.splitext(image_name)[0] + ".png")
    #print(seg_path + os.path.splitext(image_name)[0] + ".png")
    seg_image = cv2.cvtColor(seg_image, cv2.COLOR_BGR2RGB)
    image_copy = seg_image.copy()*0 
    presentation = 0
    if bb is not None:       
        for label, x1, x2, y1, y2, area in bb:
        # 1. Any pixel outside the box annotations is reset to background label.
        # A: getting the scaled version of the normalized coordinates.
        # A: this method will also get rid of any false positives from DeepLab.
            sx1 = int(x1*w)
            sx2 = int(x2*w)
            sy1 = int(y1*h)
            sy2 = int(y2*h)
            # A: this +1 is because the background color is the 0th index in the color
            # list, but this label did not exist in YOLO. Thus we are shifting every
            # label by 1.
            label = int(label)+1
            
            # 2. If the area of a segment is too small compared to its corresponding
            # bounding box (e.g. IoU< 50%), the box area is reset to its initial label
            # (fed in the first round). This enforces a minimal area.
            # A: We'll check for 2. before applying 1., since if the condition is true,
            # the whole box area will needs to be set with the label color. This will be
            # done by counting the total number of pixels of that color in the seg_mask.
            # If it's less than half the total area of that bounding box, the whole area
            # will be set to that label color.
            slice_area = seg_image[sy1:sy2, sx1:sx2].copy()
            # the area is all pixels of the bounding box
            area = int((sx2 - sx1)*(sy2 - sy1))
            color_mask = np.all(slice_area == colors[label], axis=-1)

            count = int(np.sum((slice_area == colors[label]))/3) 

            # A: This is where the second condition is effectly applied. The entire bounding box
            # area is set to that label color, if the pixel count of that label is smaller
            # than 50% of the area.
            # Most classes will use 50% of the IoU.
            # problem classes: [2, 4, 6] (needs to be smaller) (chaves fechadas, fusivel)
            #                  [12]  (needs to not shrink too much) (transformador)
            # A: for classes that it's known that the object will be much smaller than the original
            # bounding box, 1/2 of the area is too much. 1/3 will be used to allow the model to
            # shrink those objects towards something.
            if label in [2, 4, 6]:
                if(count < int(area * 0.25)):
                    image_copy[sy1:sy2, sx1:sx2] = colors[label]
                else:
                    image_copy[sy1:sy2, sx1:sx2][np.where(np.all(seg_image[sy1:sy2, sx1:sx2] == colors[label], axis=-1))[:2]] = colors[label]
            # A: for classes that it's known the bounding box actually was pretty close to the object itself
            elif label == 12:
                if(count < int(area * 0.8)):
                    image_copy[sy1:sy2, sx1:sx2] = colors[label]
                else:
                    image_copy[sy1:sy2, sx1:sx2][np.where(np.all(seg_image[sy1:sy2, sx1:sx2] == colors[label], axis=-1))[:2]] = colors[label]
            # Most classes will use 50% of the IoU.
            else:
                if(count < int(area * 0.5)):
                    image_copy[sy1:sy2, sx1:sx2] = colors[label]
                # A: By default, any pixel not within a bounding box is black.
                # To make 1. happen, we'll work within the constraints of the original bounding
                # boxes, so that false positives are removed and the prediction blobs can't be
                # bigger than the bounding boxes. Then, within those constraints, the pixels
                # coordinates that share the correct label color will be set to that color.
                # Do not forget that images are y, x.
                else:
                    image_copy[sy1:sy2, sx1:sx2][np.where(np.all(seg_image[sy1:sy2, sx1:sx2] == colors[label], axis=-1))[:2]] = colors[label] 
          
        return image, seg_image, image_copy#, dense
    else:
        #print(f"Empty image: {image_name}")
        return image, image*0, image*0#, image*0
            
def crf_method(image_path, seg_image, image_name):
    image, bb, w, h = open_yolo_sort(image_path, image_name)
    seg_copy = seg_image.copy()*0 
    dense_copy = seg_image.copy()*0 
    presentation = 0
    if bb is not None:  
        dense = crf(image, seg_image, 5)        
        for label, x1, x2, y1, y2, area in bb:
            sx1 = int(x1*w)
            sx2 = int(x2*w)
            sy1 = int(y1*h)
            sy2 = int(y2*h)
            label = int(label)+1
            area = int((sx2 - sx1)*(sy2 - sy1))
            
            # A: Since the CRF can guess the wrong color, we need to correct that if it happens.
            # We'll assume that, within a bounding box, the majority of the non-background pixels
            # belong to that object class, in the dense image.
            crf_slice_area = dense[sy1:sy2, sx1:sx2].copy()
            
            count = int(np.sum((crf_slice_area == colors[label]))/3)
            #print(f"Area: {area}, Class pixels: {count}")

            # A: if the list is empty, it's because the background was the only color
            # present.
            if count > 0:
                # A: reapplying the second condition - the IoU - to the image. This is already excluding the
                # known problematic classes - the fuses and switches will always be much, much smaller than their
                # depicted boxes, and as such they need to be compared with less than half the area.
                # Meanwhile, transformers are usually closer to their bounding boxes.
                
                # for classes that it's known they're pretty small
                # chaves fechadas, fusivel
                if label in [2, 4, 6]:
                    if(count < int(area * 0.25)):
                        #print("IoU < 0.25 for problematic class. Replacing.")
                        dense_copy[sy1:sy2, sx1:sx2][np.where(np.all(seg_image[sy1:sy2, sx1:sx2] == colors[label], axis=-1))[:2]] = colors[label]
                    else:
                        dense_copy[sy1:sy2, sx1:sx2][np.where(np.all(crf_slice_area == colors[label], axis=-1))[:2]] = colors[label]
                # A: for classes that it's known the bounding box actually was pretty close to the object itself
                # transformador
                elif label == 12:
                    if(count < int(area * 0.8)):
                        #print("IoU < 0.80 problematic class. Replacing.")
                        dense_copy[sy1:sy2, sx1:sx2][np.where(np.all(seg_image[sy1:sy2, sx1:sx2] == colors[label], axis=-1))[:2]] = colors[label]
                    else:
                        dense_copy[sy1:sy2, sx1:sx2][np.where(np.all(crf_slice_area == colors[label], axis=-1))[:2]] = colors[label]
                
                else:
                    # the remaining classes will keep at the 50% threshold for IoU
                    if(count < int(area * 0.5)):
                        #print("IoU < 0.50. Replacing.")
                        dense_copy[sy1:sy2, sx1:sx2][np.where(np.all(seg_image[sy1:sy2, sx1:sx2] == colors[label], axis=-1))[:2]] = colors[label]
                    else:
                        dense_copy[sy1:sy2, sx1:sx2][np.where(np.all(crf_slice_area == colors[label], axis=-1))[:2]] = colors[label]
                    
            # A: assuming background was the only color, the CRF removed it, so we'll just copy whatever is
            # in the corrected mask, instead of the crf.
            else:
                #print("Object not detected by denseCRF. Adding original from segmentation mask.")
                dense_copy[sy1:sy2, sx1:sx2][np.where(np.all(seg_image[sy1:sy2, sx1:sx2] == colors[label], axis=-1))[:2]] = colors[label]
                

        return dense_copy
    else:
        #print(f"Empty image: {image_name}")
        return image*0

def post_process(image_path, seg_path, save_path, image_name):
    try:
        image, seg_image, cut_image = box_method(image_path, seg_path, save_path, image_name)
        colors, labels = np.unique(cut_image, return_inverse=True)
        n_labels = len(set(labels.flat))
        print(image_name)
        if (n_labels > 1):
            crf_image = crf_method(image_path, cut_image, image_name)
            #kernel = np.ones((10,10),np.uint8)
            #crf_image = cv2.morphologyEx(crf_image, cv2.MORPH_CLOSE, kernel, iterations=1)
            #crf_image = cv2.morphologyEx(crf_image, cv2.MORPH_OPEN, kernel, iterations=1)
            #crf_image = cv2.dilate(crf_image, kernel, iterations=1)
            cv2.imwrite(save_path + os.path.splitext(image_name)[0] +".png", cv2.cvtColor(crf_image, cv2.COLOR_BGR2RGB))
            #return True
        else:
            cv2.imwrite(save_path + os.path.splitext(image_name)[0] +".png", cv2.cvtColor(cut_image, cv2.COLOR_BGR2RGB))
            #return False
    except Exception as e:
        cv2.imwrite(save_path + os.path.splitext(image_name)[0] +".png", cv2.cvtColor(cut_image, cv2.COLOR_BGR2RGB))
        print(image_name)
        print(e)
        #return False
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='original images path')
    parser.add_argument('--seg_path', type=str, help='segmantation mask path for the equivalent images from image path')
    parser.add_argument('--save_path', type=str, help='path where the postprocessed masks will be saved')
    opt = parser.parse_args()
    print(f"Image: {opt.image_path}\nSeg: {opt.seg_path}\nSave: {opt.save_path}")

    if not os.path.isdir(opt.save_path):
        os.mkdir(opt.save_path)
        print("Created dir " + opt.save_path)

    file_list = os.path.join(os.path.join(opt.image_path, "*.*"))
    # using glob to get the image names, since they can have more than one extension
    file_list = glob.glob(file_list)
    image_list = []
    for file in file_list:
        if ".txt" not in file:
            file = file.split("/")[-1]
            #file = os.path.splitext(file)[0]
            image_list.append(file)
    # making a partial function, since the only variable argument will be the image_name
    part_post_process = functools.partial(post_process, opt.image_path, opt.seg_path, opt.save_path)
    
    pool = Pool(5)

    #for r in tqdm(pool.imap_unordered(part_post_process, image_list), total=len(image_list)):
    #    pass
    pool.map(part_post_process, image_list)
    pool.close()
    
    #r = p_map(part_post_process, image_list)
    # the image_list is made into an iterator
    #r = process_map(part_post_process, iter(image_list))
    #for image_name in tqdm(image_list[0:10], total=len(image_list[0:10])):
    #    part_post_process(image_name)
    print(f"Processed {len(image_list)} images. Saved images to {opt.save_path}")
    
if __name__ == '__main__':
    main()