import cv2
import numpy as np
from utils.util import sub_folders
from glob import glob
from tqdm import tqdm
import sys
import argparse
from demographic_face_analyze import without_beard_region
sys.path.insert(0, "../face-parsing.PyTorch-master/")
from test import evaluate


def weights_calc(data):
    weights = []
    num_list = list(data.values())
    total = sum(num_list)
    for num in num_list:
        weights.append(num / total)
    return np.array(weights)
        
    
def information_analysis(image):
    '''grayscale image'''
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness_data = {}
    for pixel in image:
        if pixel not in brightness_data:
           brightness_data[pixel] = brightness_data.get(pixel, 0) + 1
        else:
           brightness_data[pixel] += 1
    weight_list = weights_calc(brightness_data)
    level_list = np.array(list(brightness_data.keys()))
    avg_level = sum(level_list * weight_list)
    
    information = 0
    for index, cur_level in enumerate(level_list, 0):
        information += abs(cur_level - avg_level) * weight_list[index]
    return information


def start(*root_dirs):
    for root_dir in root_dirs:
        for sub_folder in sub_folders(root_dir):
            cate = f"{sub_folder.split('/')[-3]}_{sub_folder.split('/')[-2]}"
            im_list = glob(sub_folder + "/*")
            ims = []
            for im in tqdm(im_list):
                ims.append(cv2.imread(im))
            mask = evaluate(ims)
            upper_face = []
            for i in range(len(mask)):
                upper_face.append(without_beard_region(ims[i], mask[i]))
            information_list = []
            for image in tqdm(upper_face):
                information_list.append(information_analysis(image)) 
            average_information = np.mean(information_list)
            print(f"Information of {cate}: {average_information}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "This is for calculating the information of a demographic group")
    parser.add_argument("--source", "-s", help="Folder of the first category.")
    args = parser.parse_args()

    start(args.source)
    