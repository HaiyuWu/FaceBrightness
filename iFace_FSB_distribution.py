import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import numpy as np
import seaborn as sns
import shutil

fsb_data = {}
x_axis = []
y_axis = []
colors = sns.color_palette("husl")

names = {}

with open("./degree_results.txt", "r") as f:
    contents = f.readlines()
    for content in contents:
        fsb_data[content.split(" ")[0].split("/")[-1]] = float(content.split(" ")[1])

        names[content.split(" ")[0].split("/")[-1]] = content.split(" ")[0]

name = []
brightness_files = glob("./brightness/*")
for file in brightness_files:
    with open(file, "r") as f:
        contents = f.readlines()
    for content in tqdm(contents):
        try:
            x_axis.append(fsb_data[content.split(",")[0]])
            y_axis.append(float(content.split(",")[1]))
            name.append(content.split(",")[0])
        except Exception:
            continue

x_axis = np.array(x_axis)
y_axis = np.array(y_axis)
y_b_axis = x_axis * 100 - 15000
difference = abs(y_axis - y_b_axis)
with open("./max_ten_im.txt", "a+") as f:
    for i in range(20):
        print(np.max(difference), max(difference))
        max_pos = np.where(difference == np.max(difference))
        print(x_axis[max_pos], y_axis[max_pos], y_b_axis[max_pos], name[int(max_pos[0][0])])
        shutil.copy(names[name[int(max_pos[0][0])]], f"{i}.png")

        f.write(f"{i}.png, {x_axis[max_pos]}, {y_axis[max_pos]}, {names[name[int(max_pos[0][0])]]}\n")
        difference = np.delete(difference, int(max_pos[0][0]))
        x_axis = np.delete(x_axis, int(max_pos[0][0]))
        y_axis = np.delete(y_axis, int(max_pos[0][0]))
        y_b_axis = np.delete(y_b_axis, int(max_pos[0][0]))
        name.pop(int(max_pos[0][0]))

degrees_x = np.percentile(x_axis, [5, 15, 85, 95])
degrees_y = np.percentile(y_axis, [5, 15, 85, 95])
print(degrees_x, degrees_y)
exit()
labels = ["SU|U", "U|M", "M|O", "O|SO"]
for i, (degree_x, degree_y) in enumerate(zip(degrees_x, degrees_y), 0):
    plt.axvline(degree_x, color=colors[i], linestyle=":", label=labels[i])
    plt.axhline(degree_y, color=colors[i], linestyle=":")

m, b = np.polyfit(x_axis, y_axis, 1)
plt.scatter(x_axis, y_axis, s=1, c="orange")

plt.text(182.20158298, -4148, s=chr(0x245F + 1))
plt.text(211.76864621, -476, s=chr(0x245F + 2))
plt.text(220.85059962, 937, s=chr(0x245F + 3))
plt.text(207.66581395, -98, s=chr(0x245F + 4))

plt.scatter(180.20158298, -4148, s=30, c="red")
plt.scatter(209.76864621, -476, s=30, c="red")
plt.scatter(218.85059962, 937, s=30, c="red")
plt.scatter(205.66581395, -98, s=30, c="red")

plt.title("Relationship between iFace approach and Face-skin metric")
plt.plot(np.array(x_axis), m * x_axis + b, 'b--', label="estimated line")
plt.plot(np.array(x_axis), 100 * x_axis - 15000, 'g', label="base line")
plt.legend(title="line types")
plt.xlabel("Face-brightness metric brightness value")
plt.ylabel("iFace brightness value")
plt.savefig("FSB_iFace.png")
plt.show()