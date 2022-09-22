from glob import glob
import argparse
import numpy as np
from torch.nn.functional import cosine_similarity
import torch
from tqdm import tqdm
import os
from utils.util import sub_folders


def get_the_end(folder_list, folder):
    child_paths = os.listdir(folder)
    for child_path in child_paths:
        path = os.path.join(folder, child_path)
        if os.path.isdir(path):
            get_the_end(folder_list, path)
        elif folder not in folder_list:
            folder_list.append(folder)
    return


def cosine_similarity_calc(feature1, feature2):
    return cosine_similarity(torch.tensor(feature1), torch.tensor(feature2), dim=0)


def get_names(folder):
    people = {}
    for sub_folder in sub_folders(folder):
        for npy in glob(sub_folder + "/*.npy"):
            current_people = npy.split("/")[-1].split("_")[0]
            if current_people not in people:
               people[current_people] = [npy,]
            else:
               people[current_people].append(npy)
    return people


def start_comparison(folder1, folder2, order_matters=True):
    demographic = folder1.split("/")[-3]
    cate1 = folder1.split("/")[-2]
    cate2 = folder2.split("/")[-2]
    if folder1 == folder2:
        order_matters = False
    nm_lst1 = get_names(folder1)
    nm_lst2 = get_names(folder2)
    data = []

    people_list = list(nm_lst1.keys())

    for people in tqdm(people_list):
        try:
           length1 = len(nm_lst1[people])
           length2 = len(nm_lst2[people])
           first_end = length1
           second_start = 0

           if not order_matters:
               first_end = length1 - 1

           for i in range(first_end):
               feature1 = np.load(nm_lst1[people][i], allow_pickle=True)
               person1 = nm_lst1[people][i].split("/")[-1]
               if not order_matters:
                   second_start = i + 1
               for j in range(second_start, length2):
                   person2 = nm_lst2[people][j].split("/")[-1]
                   if person1 != person2:
                       feature2 = np.load(nm_lst2[people][j], allow_pickle=True)
                       res = cosine_similarity_calc(feature1, feature2)
                       data.append(float(res))
        except Exception:
               continue
    np.save(f"{demographic}_{cate1}_{cate2}_genuine.npy", np.array(data))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "This is for calculating the cosine similarity between categories with features extracted")
    parser.add_argument("--source1", "-s1", help="Folder of the first category.")
    parser.add_argument("--source2", "-s2", help="Folder of the first category.")
    args = parser.parse_args()

    start_comparison(args.source1, args.source2)
