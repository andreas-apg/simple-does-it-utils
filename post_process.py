import cv2
import numpy as np
import argparse
import imgaug as ia
import torch
import pandas as pd
import os
import glob
from tqdm import tqdm

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
        
colors = [(162, 0, 255), # Chave seccionadora lamina (Aberta)
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
    
def box_method(image_path, seg_path, save_path, image_name):
    #print(image_name)
    image, bb, w, h = open_yolo_sort(image_path, image_name)
    seg_image = cv2.imread(seg_path + os.path.splitext(image_name)[0] + ".png")
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
            label = int(label)
            
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
            if(count < int(area/2)):
                image_copy[sy1:sy2, sx1:sx2] = colors[label]    
            # A: By default, any pixel not within a bounding box is black.
            # To make 1. happen, we'll work within the constraints of the original bounding
            # boxes, so that false positives are removed and the prediction blobs can't be
            # bigger than the bounding boxes. Then, within those constraints, the pixels
            # coordinates that share the correct label color will be set to that color.
            # Do not forget that images are y, x.
            else:
                image_copy[sy1:sy2, sx1:sx2][np.where(np.all(seg_image[sy1:sy2, sx1:sx2] == colors[label], axis=-1))[:2]] = colors[label]            
            #x_test = mean_substraction(copy.deepcopy(image))
            #dense_CRF_ = dense_CRF(x_test.astype(np.uint8), image_copy, 15)
            #crf_mask = dense_CRF_.run_dense_CRF()
        return image, seg_image, image_copy#, crf_mask
    else:
        #print(f"Empty image: {image_name}")
        return image, image*0, image*0
    
    # AT THE MOMENT, STEP 3 IS NOT WORKING
    # 3. As it is common practice among semantic labelling methods, we filter
    # the output of the network to better respect the object boundaries.
    #(We use DenseCRFwith the DeepLabv1 parameters ). 
    # In our weakly supervised scenario, boundary-aware filtering is particu
    # larly useful to improve objects delineation.
    
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
    count = 0
    for image_name in tqdm(image_list):
        count += 1
        image, seg_image, cut_image = box_method(opt.image_path, opt.seg_path, opt.save_path, image_name)
        cv2.imwrite(opt.save_path + os.path.splitext(image_name)[0] +".png", cv2.cvtColor(cut_image, cv2.COLOR_BGR2RGB))
    print(f"Processed {count} images. Saved images to {opt.save_path}")
    
if __name__ == '__main__':
    main()