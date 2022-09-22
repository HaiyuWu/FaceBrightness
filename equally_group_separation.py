from utils.util import sub_folders, check_folder
import os
import shutil
from tqdm import tqdm


start = 115
end = 200
interval = 20
step = 5
# boundaries, for # of sub-groups
boundaries = [[x, x + interval] for x in range(start, end - interval + step, step)]
# save data in a dictionary{im_name:(group, degree)}
data = {}
with open("./degree_results.txt", "r") as f:
     for content in f.readlines():
         im_name, group, degree = content.split(" ")[0].split("/")[-1],\
                                  content.split(" ")[0].split("/")[2], float(content.split(" ")[1])
         data[im_name] = (group, degree)
# create a base path
base_path = "./label_based_images/d_group/sub_group/"
# read the images in M-ex
im_dir = "./degree_analyzing_images/"
print("Start moving...")
for sub_folder in tqdm(sub_folders(im_dir)):
    for im in tqdm(os.listdir(sub_folder)):
        flag = False
        i = 0
        # get the degree and group
        group, degree = data[im]
        while i < len(boundaries):
            # compare with the boundaries, and put the image into corresponding group /demographic_group/sub_group/
            if boundaries[i][0] <= degree < boundaries[i][1]:
               save_path = base_path.replace("d_group", group).replace("sub_group", str(i))
               if not os.path.exists(save_path):
                  check_folder(save_path)
               shutil.copy(f"{sub_folder}/{im}", save_path)
               flag = True
            elif flag:
               i = len(boundaries)
            i += 1
