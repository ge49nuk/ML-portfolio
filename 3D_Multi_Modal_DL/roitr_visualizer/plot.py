import matplotlib.pyplot as plt
import numpy as np
import os


def draw_distance_geo_feat(geo_distance, feat_distance):
    geo_distance = geo_distance.flatten()
    feat_distance = feat_distance.flatten()
    plt.scatter(geo_distance, feat_distance)
    plt.show()

def update_acc_info_plots(acc_logger, dataset_name):
    os.makedirs("output/plots_"+dataset_name, exist_ok=True)
    plot_samples_and_acc(acc_logger, dataset_name)
    print("Updated plots")
    
def plot_samples_and_acc(acc_logger, dataset_name):
    for metric in acc_logger.keys():
        if metric == "acc_ms":
            xlabel = "#mask pixels"
            step_size = 4096
        elif metric == "acc_rel_tgt_corrs":
            xlabel = "relative object #correspondences"
            step_size = 0.005
        elif metric == "acc_dif_src_corrs":
            xlabel = "distance of relative depth #correspondences"
            step_size = 0.002
        elif metric == "acc_conf":
            xlabel = "mean confidence"
            step_size = 0.01
        elif metric == "acc_conf_dif":
            xlabel = "confidence distance"
            step_size = 0.004
        else:
            xlabel = "semantic score distance"
            step_size = 0.01
        

        samples_per_bin = np.sum(acc_logger[metric], axis = 1)
        accuracies = acc_logger[metric][:,0].astype(np.float32) / np.where(samples_per_bin < 10, 1e10, samples_per_bin)
        x_values = np.arange(0, step_size*10, step_size)

        if metric in ["acc_dif_src_corrs", "acc_conf_dif", "acc_dif_sem_score"]:
            with open(f"output/plots_{dataset_name}/{metric}.txt", "w") as f:
                for i,x in enumerate(x_values):
                    f.write(f"({x},{accuracies[i]}) ")
            with open(f"output/plots_{dataset_name}/{metric}_samples.txt", "w") as f:
                for i,x in enumerate(x_values):
                    f.write(f"({x},{samples_per_bin[i]/np.sum(samples_per_bin)}) ")

        plt.plot(x_values, accuracies, marker='o', linestyle='-', color='b')
        plt.xlabel(xlabel)
        plt.ylabel('Accuracy')
        plt.title('Accuracy per ' + xlabel)
        plt.savefig(f'output/plots_{dataset_name}/{"_".join(metric.split("_")[1:])}_acc.png')
        plt.clf()
        
        plt.plot(x_values, samples_per_bin, marker='o', linestyle='-', color='b')
        plt.xlabel(xlabel)
        plt.ylabel('#Samples')
        plt.title('#Samples per ' + xlabel)
        plt.savefig(f'output/plots_{dataset_name}/{"_".join(metric.split("_")[1:])}_samples.png')
        plt.clf()