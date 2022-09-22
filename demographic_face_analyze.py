from utils.util import sub_folders
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import sys

sys.path.insert(0, "../face-parsing.PyTorch-master/")
from test import evaluate


def without_beard_degree_level_analysis(*argv):
    for im_dir in argv:
        cate = im_dir.split("/")[-2]
        data = []
        for path in sub_folders(im_dir):
            images_path = glob(path + "/*")
            for image in tqdm(images_path):
                im = cv2.imread(image)
                mask = evaluate([im, ])
                data.append(round(np.mean(without_beard_region(im, mask[0])), 2))
        np.save(f"./without_beard_data/{cate}_without_beard.npy", data)


def without_beard_region(image, mask, position=False):
    assert ((len(image.shape) == 3) or (
            len(image.shape) == 2)), f"Expect a gray or colored image, but get {len(image.shape)} channel image"
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = mask.astype(np.uint8)
    mask = cv2.resize(mask, (224, 224), fx=1, fy=1, interpolation=cv2.INTER_NEAREST)
    face_pos = np.where(mask == 1)

    nose = np.where(mask == 10)
    b_lim = nose[0][-1]
    without_bread = [[], []]
    i = 0
    while face_pos[0][i] < b_lim:
        without_bread[0].append(face_pos[0][i])
        without_bread[1].append(face_pos[1][i])
        i += 1
    # gray[without_bread[0], without_bread[1]] = 255
    # cv2.imwrite(f"{gray[0, 0]}.png", gray)
    if position:
       return gray[without_bread[0], without_bread[1]], without_bread
    return gray[without_bread[0], without_bread[1]]


if __name__ == '__main__':
    without_beard_degree_level_analysis("./res/")


