import os
import shutil
import cv2
import numpy as np
from config import threshold_txt
from glob import glob


def brightness_analyses(image, mask):
    # convert colored image to gray, then analyzing the mean value of their faces.
    assert ((len(image.shape) == 3) or (
            len(image.shape) == 2)), f"Expect a gray or colored image, but get {len(image.shape)} channel image"
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = mask.astype(np.uint8)
    mask = cv2.resize(mask, (224, 224), fx=1, fy=1, interpolation=cv2.INTER_NEAREST)
    pos = np.where(mask == 1)
    return gray[pos[0], pos[1]]


def check_folder(*argv):
    for path in argv:
        if not os.path.exists(path):
            os.makedirs(path)
            print("% has been created" % path)
        else:
            shutil.rmtree(path)
            os.makedirs(path)
            print("% has been created" % path)


def FMR_calculation(path="./results/same_gamma.txt"):
    assert (os.path.exists(
        threshold_txt)), "Please run pick_up_threshold_value function first to generate a threshold value!"
    assert (os.path.exists(path)), f"{path} doesn't exist!"

    with open(threshold_txt, "r") as file:
        threshold = float(file.read())

    npy_files = glob(path + "/*.npy")

    for npy in npy_files:
        group = npy.split("/")[-1].split(".")[0]
        data = np.load(npy)
        FP = np.count_nonzero(data >= threshold)
        FMR = FP / len(data) * 100
        print(f"{group}: FMR is {FMR}")


def get_the_end(folder_list, folder):
    child_paths = os.listdir(folder)
    for child_path in child_paths:
        path = os.path.join(folder, child_path)
        if os.path.isdir(path):
            get_the_end(folder_list, path)
        elif folder not in folder_list:
            folder_list.append(folder)
    return


def sub_folders(path):
    folder = []
    get_the_end(folder, path)
    return folder
