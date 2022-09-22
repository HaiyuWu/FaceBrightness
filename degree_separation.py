from demographic_face_analyze import without_beard_region
import numpy as np
from glob import glob
import shutil
import cv2
import matplotlib.pyplot as plt
import os
import seaborn as sns
from tqdm import tqdm
from utils.util import check_folder
from config import strongly_over_exposed_dir, strongly_under_exposed_dir, under_exposed_dir, over_exposed_dir, \
    appropriately_exposed_dir, AA_F_dir, AA_M_dir, C_F_dir, C_M_dir
import sys

sys.path.insert(0, "../face-parsing.PyTorch-master/")
from test import evaluate


# get the all the sub-folders under one folder
# Need to make sure that no other type of files/zips at the same level of the sub-folder
def get_the_end(folder_list, folder):
    child_paths = os.listdir(folder)
    for child_path in child_paths:
        path = os.path.join(folder, child_path)
        if os.path.isdir(path):
            get_the_end(folder_list, path)
        elif folder not in folder_list:
            folder_list.append(folder)
    return


def get_distribution(dataset_folder="./data/", percentile=(5, 15, 85, 95, 100)):
    """
    This function is used to get the distribution of the face lightness degree in the whole dataset.
    :param dataset_folder:
    :param percentile:
    :return:
    """
    image_folders = []
    degree = []
    get_the_end(image_folders, dataset_folder)
    with open("degree_results.txt", "a+") as file:
        for im_dir in image_folders:
            paths = []
            images_path = glob(im_dir + "/*")
            ims = []
            for image in tqdm(images_path):
                ims.append(cv2.imread(image))
                paths.append(image)
            mask = evaluate(ims)
            for i in range(len(ims)):
                # calculate the brightness degree by using the upper face region.
                this_degree = np.mean(without_beard_region(ims[i], mask[i]))
                degree.append(this_degree)
                file.write(f"{paths[i]} {this_degree}\n")
    # calculate the brightness boundaries, which is used to separate the images into five brightness groups.
    percentile_matched_value = np.percentile(np.array(degree), percentile)
    print(percentile_matched_value)


def get_proper_degree_values(percentile=(5, 15, 75, 95, 100)):
    assert os.path.exists(
        "degree_results.txt"), print(
        "Run get_distribution function first to get the lightness degree of each image in the database!")
    degrees = []
    with open("degree_results.txt", "r") as file:
        contents = file.readlines()
        for content in contents:
            _, degree = content.replace("\n", "").split()
            degrees.append(float(degree))

    percentile_matched_value = np.percentile(degrees, percentile)
    return percentile_matched_value


def initialization(*argv):
    for brightness_group in argv:
        check_folder(AA_F_dir + brightness_group, AA_M_dir + brightness_group, C_F_dir + brightness_group, C_M_dir + brightness_group)


if __name__ == '__main__':
    # get the txt of the brightness values
    get_distribution(dataset_folder="./images/", percentile=(5, 15, 75, 95, 100))
    exit()
    # Separate images to folder
    degrees = get_proper_degree_values((15, 85))
    print(degrees)
    # max number of images in one folder
    chunk = 30000
    # name of sub-folder of SU,U,M,O,SO for four demographic groups
    dic = {"AA_F": [1, 1, 1, 1, 1], "AA_M": [1, 1, 1, 1, 1], "C_F": [1, 1, 1, 1, 1], "C_M": [1, 1, 1, 1, 1], }
    # initialization
    so, su, o, u, a = dic["AA_F"]
    SO = strongly_over_exposed_dir + f"{so}/"
    SU = strongly_under_exposed_dir + f"{su}/"
    O = over_exposed_dir + f"{o}/"
    U = under_exposed_dir + f"{u}/"
    A = appropriately_exposed_dir + f"{a}/"

    initialization(SO, SU, O, U, A)

    with open("./degree_results.txt", "r") as file:
        contents = file.readlines()
        for content in tqdm(contents):
            image, degree = content.replace("\n", "").split()[0], float(content.replace("\n", "").split()[1])
            name = image.split("/")[2]
            main_folder = AA_F_dir.replace("AA_F", f"{name}")

            SO = strongly_over_exposed_dir + f"{dic[name][0]}/"
            SU = strongly_under_exposed_dir + f"{dic[name][1]}/"
            O = over_exposed_dir + f"{dic[name][2]}/"
            U = under_exposed_dir + f"{dic[name][3]}/"
            A = appropriately_exposed_dir + f"{dic[name][4]}/"

            if degree < degrees[0]:
                if len(os.listdir(main_folder + SU)) == chunk:
                    dic[name][1] += 1
                    SU = strongly_under_exposed_dir + f"{dic[name][1]}/"
                    check_folder(main_folder + SU)
                shutil.copy(image, main_folder + SU)
            elif degree < degrees[1]:
                if len(os.listdir(main_folder + U)) == chunk:
                    dic[name][3] += 1
                    U = under_exposed_dir + f"{dic[name][3]}/"
                    check_folder(main_folder + U)
                shutil.copy(image, main_folder + U)
            elif degree < degrees[2]:
                if len(os.listdir(main_folder + A)) == chunk:
                    dic[name][4] += 1
                    A = appropriately_exposed_dir + f"{dic[name][4]}/"
                    check_folder(main_folder + A)
                shutil.copy(image, main_folder + A)
            elif degree < degrees[3]:
                if len(os.listdir(main_folder + O)) == chunk:
                    dic[name][2] += 1
                    O = over_exposed_dir + f"{dic[name][2]}/"
                    check_folder(main_folder + O)
                shutil.copy(image, main_folder + O)
            elif degree <= degrees[4]:
                if len(os.listdir(main_folder + SO)) == chunk:
                    dic[name][0] += 1
                    SO = strongly_over_exposed_dir + f"{dic[name][0]}/"
                    check_folder(main_folder + SO)
                shutil.copy(image, main_folder + SO)