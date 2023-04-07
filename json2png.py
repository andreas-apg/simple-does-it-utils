import json
import numpy as np
import argparse
import cv2
import os

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str)
    return parser

def get_class_item(input):
    """The 15 class names (+ background).
    Returns:
        array of str
    """
    names = ["Background",
            "Chave seccionadora lamina (aberta)",
            "Chave seccionadora lamina (fechada)",
            "Chave seccionadora tandem (aberta)",
            "Chave seccionadora tandem (fechada)",
            "Disjuntor",
            "Fusivel",
            "Isolador disco de vidro",
            "Isolador pino de porcelana",
            "Mufla",
            "Para-raio",
            "Religador",
            "Transformador",
            "Transformador de Corrente (TC)",
            "Transformador de Potencial (TP)",
            "Chave tripolar"]

    indexes = { "Background": 0,
                "Chave seccionadora lamina (aberta)": 1,
                "Chave seccionadora lamina (fechada)": 2,
                "Chave seccionadora tandem (aberta)": 3,
                "Chave seccionadora tandem (fechada)": 4,
                "Disjuntor": 5,
                "Fusivel": 6,
                "Isolador disco de vidro": 7,
                "Isolador pino de porcelana": 8,
                "Mufla": 9,
                "Para-raio": 10,
                "Religador": 11,
                "Transformador": 12,
                "Transformador de Corrente (TC)": 13,
                "Transformador de Potencial (TP)": 14,
                "Chave tripolar": 15}
    try:
        if type(input) == str:
            return indexes[input]
        elif type(input) == int:
            return names[input]
    except Exception as e:
        print(e)

def get_labels():
    """Load the mapping that associates classes with label colors.
       Our electrical substation dataset has 15 objects + background.
    Returns:
        np.ndarray with dimensions (16, 3)
    """
    return [(0, 0, 0),       # Background
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
            (162, 48, 0),   # Transformador de Potencial (TP)
            (162, 0, 96)]     # Chave tripolar
  
def main():
    # loading the argparser
    opts = get_argparser().parse_args()
    name = os.path.basename(os.path.splitext(opts.file_name)[0])
    if ".json" not in opts.file_name:
        return
    # Opening JSON file
    f = open(opts.file_name)
    # returns JSON object as 
    # a dictionary
    img = np.zeros((100, 100, 3), np.uint8)
    img_insulator = np.zeros((100, 100, 3), np.uint8)
    try:  
        data = json.load(f)  
        img = np.zeros((data["imageHeight"], data["imageWidth"], 3), np.uint8)
        img_insulator = np.zeros((data["imageHeight"], data["imageWidth"], 3), np.uint8)
        labels = get_labels()

        # Iterating through the json
        # list

        for shape in data['shapes']:
            class_index = get_class_item(shape['label'])
            #print(shape['label'])
            #print(class_index)
            #print(labels[class_index])

            # converting the points into a np array to use in fillPoly
            points = np.array(shape['points']).astype(int)
            
            # https://www.geeksforgeeks.org/draw-a-filled-polygon-using-the-opencv-function-fillpoly/
            if class_index == 8: # porcelain pin insulator
                cv2.fillPoly(img_insulator, pts=[points], color=labels[class_index])
            else: # the other 14 classes
                cv2.fillPoly(img, pts=[points], color=labels[class_index])
            # for point in shape['points']:
            #     print(point)
            #

        if not os.path.exists('14_masks'):
            os.mkdir('14_masks')
        if not os.path.exists('porcelain_masks'):
            os.mkdir('porcelain_masks')
        
        cv2.imwrite(f"./14_masks/{name}.png", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        cv2.imwrite(f"./porcelain_masks/{name}.png", cv2.cvtColor(img_insulator, cv2.COLOR_BGR2RGB))
        # Closing file
        f.close()

    except Exception as e:
        print(e)
        print(opts.file_name)
        cv2.imwrite(f"./14_masks/{name}.png", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        cv2.imwrite(f"./porcelain_masks/{name}.png", cv2.cvtColor(img_insulator, cv2.COLOR_BGR2RGB))
        # Closing file
        f.close()



if __name__ == '__main__':
    main()
