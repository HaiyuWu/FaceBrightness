import numpy as np
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors


def similarity_distribution(group, gender, plot_type="all"):
    colors = list(mcolors.CSS4_COLORS.keys())

    # genuine
    npy_dir_root_gen = "./genuine_npys/"
    aim_npys_gen = glob(f"{npy_dir_root_gen}/{group}_{gender}*")
    aim_npys_gen.sort()
    # imposter
    npy_dir_root_imp = "./npys_main/"
    aim_npys_imp = glob(f"{npy_dir_root_imp}/{group}_{gender}*")
    aim_npys_imp.sort()

    labels = []

    if plot_type == "all":
        data_gen = []
        data_imp = []
        for aim_npy_gen in tqdm(aim_npys_gen):
            data_gen = np.concatenate((data_gen, np.load(aim_npy_gen)))

        for aim_npy_imp in tqdm(aim_npys_imp):
            data_imp = np.concatenate((np.load(aim_npy_imp)))
        d_prime = abs(np.mean(data_gen) - np.mean(data_imp)) / np.sqrt(
            0.5 * (np.var(data_gen) + np.var(data_imp)))
        cate = f"{group}_{gender}_"
        labels.append(f"d-prime: {round(d_prime, 3)}")
        plt.hist(data_gen, bins='auto', histtype='step', density=True, color='b',
                 label=f"{cate} genuine", linewidth=1.5)
        plt.hist(data_imp, bins='auto', histtype='step', density=True, color='b',
                 label=f"{cate} imposter", linestyle='dashed',
                 linewidth=1.5)
        ncol = 2
    else:
        cate_labels = ["M_M", "M_O", "O_O", "SU_O", "U_O", "M_SO", "O_SO", "SO_SO", "SU_SO", "U_SO", "SU_M", "SU_SU",
                       "U_M", "SU_U", "U_U"]
        for i, aim_npys_gen in enumerate(aim_npys_gen, 0):
            data_gen = np.load(aim_npys_gen)
            data_imp = np.load(aim_npys_imp[i])
            d_prime = abs(np.mean(data_gen) - np.mean(data_imp)) / np.sqrt(
                0.5 * (np.var(data_gen) + np.var(data_imp)))
            cate = f"{group}_{gender}_{cate_labels[i]}_"
            labels.append(f"{cate}d-prime: {round(d_prime, 3)}")
            plt.hist(data_gen, bins='auto', histtype='step', density=True, color=colors[i],
                     label=f"{cate} genuine", linewidth=1.5)
            plt.hist(data_imp, bins='auto', histtype='step', density=True, color=colors[i],
                     label=f"{cate} imposter", linestyle='dashed',
                     linewidth=1.5)
        ncol = len(cate_labels) // 3

    legend1 = plt.legend(
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        loc="lower left",
        mode="expand",
        borderaxespad=0,
        ncol=ncol,
        fontsize=10,
        edgecolor="black",
        handletextpad=0.3,
    )

    handles = []
    for c in colors[:len(labels)]:
        handles.append(Rectangle((0, 0), 1, 1, color=c, fill=True))

    handles = np.asarray(handles)

    plt.legend(handles, labels, loc="upper left", fontsize=10)
    plt.gca().add_artist(legend1)

    plt.xlabel("Match Scores")
    plt.ylabel("Relative Frequency")
    plt.savefig(f"{group}_{gender}_{plot_type}.png")
    plt.show()


def draw_similarities(*argv):
    cates = {"M_M": "appropriately_exposed_appropriately_exposed", "M_O": "over_exposed_appropriately_exposed",
             "O_O": "over_exposed_over_exposed", "SU_O": "over_exposed_strongly_under_exposed",
             "U_O": "over_exposed_under_exposed", "M_SO": "strongly_over_exposed_appropriately_exposed",
             "O_SO": "strongly_over_exposed_over_exposed", "SO_SO": "strongly_over_exposed_strongly_over_exposed",
             "SU_SO": "strongly_over_exposed_strongly_under_exposed", "U_SO": "strongly_over_exposed_under_exposed",
             "SU_M": "strongly_under_exposed_appropriately_exposed",
             "SU_SU": "strongly_under_exposed_strongly_under_exposed", "U_M": "under_exposed_appropriately_exposed",
             "SU_U": "under_exposed_strongly_under_exposed", "U_U": "under_exposed_under_exposed"}
    name = argv[0]
    categories = argv[1]
    colors = sns.color_palette("colorblind")
    labels = []
    for i, group in enumerate(categories, 0):
        npy_dir_root_gen = "./genuine_npys/"
        npy_dir_root_imp = "./npys_main/"
        if len(argv) == 3:
            npy_dir_root_gen = f"./{argv[2]}_genuine_npys/"
            npy_dir_root_imp = f"./{argv[2]}_npys_main/"
        # genuine
        aim_npys_gen = glob(f"{npy_dir_root_gen}/{name}_{cates[group]}*")
        data_gen = np.load(aim_npys_gen[0])

        # imposter
        aim_npys_imp = glob(f"{npy_dir_root_imp}/{name}_{cates[group]}*")
        data_imp = np.load(aim_npys_imp[0])
        # print(len(np.where(data_imp == 0)[0])/len(data_imp))
        # exit()
        d_prime = abs(np.mean(data_gen) - np.mean(data_imp)) / np.sqrt(
            0.5 * (np.var(data_gen) + np.var(data_imp)))

        labels.append(f"{name}_{group}_d-prime: {round(d_prime, 3)}")

        plt.hist(data_gen, bins='auto', histtype='step', density=True, label=f"{name}_{group} genuine",
                     color=colors[i], linewidth=1.5)
        plt.hist(data_imp, bins='auto', histtype='step', density=True, label=f"{name}_{group} imposter",
                     color=colors[i], linestyle='dashed', linewidth=1.5)

    legend1 = plt.legend(
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        loc="lower left",
        mode="expand",
        borderaxespad=0,
        ncol=len(categories),
        fontsize=10,
        edgecolor="black",
        handletextpad=0.3,
    )

    handles = []
    for c in colors[:len(labels)]:
        handles.append(Rectangle((0, 0), 1, 1, color=c, fill=True))

    handles = np.asarray(handles)

    plt.legend(handles, labels, loc="upper left", fontsize=10)
    plt.gca().add_artist(legend1)

    plt.xlabel("Match Scores")
    plt.ylabel("Relative Frequency")

    if len(argv) == 3:
        plt.savefig(f"{argv[2]}_{name}_mix.png")
    else:
        plt.savefig(f"{name}_mix.png")
    plt.show()


draw_similarities("AA_M", ["M_M"], "COTS")
# draw_similarities("AA_M", ["M_M", "SU_M", "M_SO", "SU_O"], "COTS")
# draw_similarities("AA_F", ["M_M", "SU_SU", "SU_M"], "COTS")
# draw_similarities("C_F", ["M_M", "SO_SO", "M_SO"], "COTS")
# draw_similarities("C_M", ["M_M", "SO_SO", "M_SO"], "COTS")

# similarity_distribution("AA", "F", plot_type="individual")
# similarity_distribution("AA", "M", plot_type="individual")
# similarity_distribution("C", "F", plot_type="individual")
# similarity_distribution("C", "M", plot_type="individual")
