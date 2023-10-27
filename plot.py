import matplotlib.pyplot as plt
import numpy as np
import csv
import scipy.stats as stats
from utils import read_BO_results, read_ea_results


def all_in_one():

    y_baseline = {"adaptec1":6.38, "adaptec2":73, "adaptec3":84, "adaptec4":79, "bigblue1":2.39, "bigblue3":91} # as reported in their paper

    ylim = {"adaptec1":[5.7,7], "adaptec2":[45,90], "adaptec3":[55,90], "adaptec4":[56,84], "bigblue1":[2.1,2.45], "bigblue3":[55,105]}
    seed_ls = [2023, 2024, 2025, 2026, 2027]
    time_budget = 1000
    color = [(0.9412, 0.2392, 0.2549),(0.9137, 0.7608, 0.1216),(0.1804, 0.6196, 0.2824),(0.3216,0.3725,0.6784)]
    flag = 0
    alpha = 0.15
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    plt.subplots_adjust(hspace=0.4)

    for dataset in list(y_baseline.keys()):

        print("{}:".format(dataset))

        first_coor = flag // 3
        second_coor = flag - first_coor * 3
        ax = axes[first_coor][second_coor]
        flag += 1

        # Draw maskplace baseline
        ax.axhline(y=y_baseline[dataset], ls=":",c="black",label="MaskPlace",linewidth =2.5)
        print("mask: ", y_baseline[dataset])

        # Draw Random curve
        xs = []
        ys = []
        for seed in seed_ls:
            dir = "result/Random/curve/{}_seed_{}.csv".format(dataset, seed)
            time_ls, hpwl_ls, hpwl_ls_min = read_ea_results(dir, time_budget)
            xs.append(time_ls)
            ys.append(hpwl_ls_min)
        len_max = 0
        for i in range(len(xs)):
            if len(xs[i]) > len_max:
                len_max_id = i
                len_max = len(xs[i])
        mean_x_axis = xs[len_max_id].copy()
        ys_interp = [np.interp(mean_x_axis, xs[i], ys[i]) for i in range(len(xs))]
        mean_y_axis = np.mean(ys_interp, axis=0)
        std_y_axis = np.std(ys_interp, axis=0)

        ax.plot(mean_x_axis,mean_y_axis,label="WireMask-RS",linewidth =1.5,color = color[0])
        ax.fill_between(mean_x_axis, mean_y_axis-std_y_axis, mean_y_axis+std_y_axis, facecolor=color[0], alpha=alpha)
        print("Random: ", round(mean_y_axis[-1],2), "std: ", round(std_y_axis[-1],2))

        # Draw BO curve
        xs = []
        ys = []
        for seed in seed_ls:
            dir = "result/BO/curve/{}_seed_{}.csv".format(dataset, seed)
            time_ls, hpwl_ls, hpwl_ls_min = read_BO_results(dir, time_budget)
            xs.append(time_ls)
            ys.append(hpwl_ls_min)
        len_max = 0
        for i in range(len(xs)):
            if len(xs[i]) > len_max:
                len_max_id = i
                len_max = len(xs[i])
        mean_x_axis = xs[len_max_id].copy()
        ys_interp = [np.interp(mean_x_axis, xs[i], ys[i]) for i in range(len(xs))]
        mean_y_axis = np.mean(ys_interp, axis=0)
        std_y_axis = np.std(ys_interp, axis=0)

        ax.plot(mean_x_axis,mean_y_axis,label="WireMask-BO",linewidth =1.5,color = color[1])
        ax.fill_between(mean_x_axis, mean_y_axis-std_y_axis, mean_y_axis+std_y_axis, facecolor=color[1], alpha=alpha)
        print("BO: ", round(mean_y_axis[-1],2), "std: ", round(std_y_axis[-1],2))

        # Draw EA_swap_only curve
        xs = []
        ys = []
        for seed in seed_ls:
            dir = "result/EA_swap_only/curve/{}_seed_{}.csv".format(dataset, seed)
            time_ls, hpwl_ls, hpwl_ls_min = read_ea_results(dir, time_budget)
            xs.append(time_ls)
            ys.append(hpwl_ls_min)
        len_max = 0
        for i in range(len(xs)):
            if len(xs[i]) > len_max:
                len_max_id = i
                len_max = len(xs[i])
        mean_x_axis = xs[len_max_id].copy()
        ys_interp = [np.interp(mean_x_axis, xs[i], ys[i]) for i in range(len(xs))]
        mean_y_axis = np.mean(ys_interp, axis=0)
        std_y_axis = np.std(ys_interp, axis=0)
        for id in range(len(mean_y_axis)):
            if mean_y_axis[id] <= y_baseline[dataset]:
                print(mean_x_axis[id])
                break

        ax.plot(mean_x_axis,mean_y_axis,label="WireMask-EA",linewidth =1.5,color = color[2])
        ax.fill_between(mean_x_axis, mean_y_axis-std_y_axis, mean_y_axis+std_y_axis, facecolor=color[2], alpha=alpha)
        print("EA: ", round(mean_y_axis[-1],2), "std: ", round(std_y_axis[-1],2))

        ax.set_ylim(ylim[dataset][0],ylim[dataset][1])
        ax.set_xlabel("Wall Clock Time (min)",fontsize=15)
        ax.set_ylabel("HPWL",fontsize=15)
        ax.set_title(dataset, fontsize=17)

    lines, labels = fig.axes[-1].get_legend_handles_labels()
    plt.subplots_adjust(bottom=0.12)
    fig.legend( lines, labels, loc='lower center',borderaxespad=0.1, ncol=4, fontsize = 15)
    
    plt.savefig("all.pdf",dpi=1000,bbox_inches="tight")
    plt.close()



if __name__ == "__main__":
    all_in_one()

