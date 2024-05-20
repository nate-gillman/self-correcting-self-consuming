import os, json

from collections import defaultdict

import matplotlib as mpl
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Matplot lib formatting
mpl.rcParams['font.family'] = 'serif'
cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
mpl.rcParams['font.serif'] = cmfont.get_name()
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['axes.unicode_minus'] = False

from scipy.ndimage import gaussian_filter1d
import numpy as np
import sys

MAX_NUMBER_OF_GENERATIONS = 50
LATEST_MODELS_CONSIDERED = set()


def read_results(exp_dir, results_type=""):
    gen_dirs = sorted(os.listdir(exp_dir))[:MAX_NUMBER_OF_GENERATIONS + 1]

    results = {}

    for i, gen_dir in enumerate(
            gen_dirs):  # SHOULD INCLIDE GENERATION_0!! ITll make graphs more intuitive... cuz B and C will agree!!
        
        gen_dir_path = os.path.join(exp_dir, gen_dir)

        eval_dict_path = os.path.join(gen_dir_path, "eval_dict.json")
        if not os.path.isfile(eval_dict_path):
            break
        with open(eval_dict_path, "r") as f:
            eval_dict = json.load(f)

        if results_type == "average":

            avg_eval_data = {
                "FID_test": 0.0,
                "Diversity_test": 0.0,
                "Matching Score_test": 0.0,
            }
            num_evals = len(eval_dict.keys())
            for model_name in eval_dict.keys():
                for score_name in avg_eval_data.keys():
                    avg_eval_data[score_name] += eval_dict[model_name][score_name] / num_evals

            # dont report average for first generation; just report the last one, replacing pervious computation
            if gen_dir == "generation_0":
                last_model = sorted(list(eval_dict.keys()))[-1]
                for score_name in avg_eval_data.keys():
                    avg_eval_data[score_name] = eval_dict[last_model][score_name]

            results[gen_dir] = avg_eval_data

        elif results_type == "latest":

            latest_model = max(eval_dict, key=lambda k: int(k))
            LATEST_MODELS_CONSIDERED.add(latest_model)
            # print(LATEST_MODELS_CONSIDERED)
            results[gen_dir] = eval_dict[latest_model]

        elif results_type == "list_of_all":
            list_eval_data = {
                "FID_test": [],
                "Diversity_test": [],
                "Matching Score_test": [],
            }
            num_evals = len(eval_dict.keys())
            for model_name in eval_dict.keys():
                for score_name in list_eval_data.keys():
                    list_eval_data[score_name].append(eval_dict[model_name][score_name])

            for score_name in list_eval_data.keys():
                list_eval_data[score_name] = np.asarray(list_eval_data[score_name])

            results[gen_dir] = list_eval_data

    return results


def build_graphs(
        exp_A_results,
        exp_B_results,
        exp_C_results,
        output_fname,
        graph_keys,
        aug_percent,
        smoothing=False,
        add_smoothing_to_graph=False,
        title=""
):
    fig, (ax1) = plt.subplots(1, 1, figsize=(6, 4))

    fig.suptitle(title, fontsize=17, y=0.95)

    for i, ax_i in enumerate([ax1]):

        # get the values for the ith graph
        
        graph_i_values = defaultdict(list)
        for generation in sorted(exp_A_results.keys()):
            try:
                result_value = exp_A_results[generation][graph_keys[i]]
            except:
                result_value = exp_A_results[generation]['0'][graph_keys[i]]
            graph_i_values[f"exp_A_{graph_keys[i]}"].append(result_value)
        for generation in sorted(exp_B_results.keys()):
            try:
                result_value = exp_B_results[generation][graph_keys[i]]
            except:
                result_value = exp_B_results[generation]['0'][graph_keys[i]]
            graph_i_values[f"exp_B_{graph_keys[i]}"].append(result_value)
        for generation in sorted(exp_C_results.keys()):
            try:
                result_value = exp_C_results[generation][graph_keys[i]]
            except:
                result_value = exp_C_results[generation]['0'][graph_keys[i]]
            graph_i_values[f"exp_C_{graph_keys[i]}"].append(result_value)
        exp_A_vals = graph_i_values[f"exp_A_{graph_keys[i]}"]
        exp_B_vals = graph_i_values[f"exp_B_{graph_keys[i]}"]
        exp_C_vals = graph_i_values[f"exp_C_{graph_keys[i]}"]

        SIGMA = 4
        if smoothing:
            exp_A_vals = list(gaussian_filter1d(exp_A_vals, SIGMA, mode="nearest"))
            exp_B_vals = list(
                gaussian_filter1d(graph_i_values[f"exp_B_{graph_keys[i]}"], SIGMA, mode="nearest"))
            exp_C_vals = list(
                gaussian_filter1d(graph_i_values[f"exp_C_{graph_keys[i]}"], SIGMA, mode="nearest"))

        if add_smoothing_to_graph:
            exp_A_vals_smoothed = list(gaussian_filter1d(exp_A_vals, SIGMA, mode="nearest"))
            exp_B_vals_smoothed = list(
                gaussian_filter1d(graph_i_values[f"exp_B_{graph_keys[i]}"], SIGMA, mode="nearest"))
            exp_C_vals_smoothed = list(
                gaussian_filter1d(graph_i_values[f"exp_C_{graph_keys[i]}"], SIGMA, mode="nearest"))

        # alpha = 0.4 if add_smoothing_to_graph else 1.0
        alpha = 0.2
        # generation 0
        ax_i.plot(
            [i for i in list(range(max(len(exp_B_vals), len(exp_C_vals)) + 1))],
            [exp_A_vals[0]] * (max(len(exp_B_vals), len(exp_C_vals)) + 1),
            c="tab:red",
            label=("Generation-0" if i == 0 else None),
            linestyle="--"
        )

        # baseline
        ax_i.plot(
            [i for i in list(range(len(exp_A_vals)))],
            exp_A_vals_smoothed,
            c="peru",
            label=(f"Baseline (continue training on gt data), 0%" if i == 0 else None),
        )
        ax_i.plot(
            [i for i in list(range(len(exp_A_vals)))],
            exp_A_vals,
            c="peru",
            alpha=alpha
        )

        # iterative fine-tuning
        ax_i.plot(
            [i for i in list(range(len(exp_B_vals_smoothed)))],
            exp_B_vals_smoothed,
            c="forestgreen",
            label=(f"Iterative fine-tuning, {aug_percent}%" if i == 0 else None),
        )
        ax_i.plot(
            [i for i in list(range(len(exp_B_vals)))],
            exp_B_vals,
            c="forestgreen",
            alpha=alpha
        )

        # terative fine-tuning with self-correction
        ax_i.plot(
            [i for i in list(range(len(exp_C_vals)))],
            exp_C_vals_smoothed,
            c="tab:blue",
            label=(f"Iterative fine-tuning with self-correction, {aug_percent}%" if i == 0 else None),
        )
        ax_i.plot(
            [i for i in list(range(len(exp_C_vals)))],
            exp_C_vals,
            c="tab:blue",
            alpha=alpha
        )

        # Set x-axis major ticks to multiples of 5
        ax_i.xaxis.set_major_locator(MultipleLocator(5))
        ax_i.set_xlabel("Generation", fontsize=16)
        ax_i.set_ylabel(f"{graph_keys[i].split('_')[0]}", fontsize=16)
        ax_i.grid(True, linewidth=0.3)
        ax_i.set_xlim(0.0, max(len(exp_B_vals), len(exp_C_vals)) - 1)

        if add_smoothing_to_graph:
            const = 0.5
            # min_y_axis = (min(exp_A_vals_smoothed + exp_B_vals_smoothed + exp_C_vals_smoothed) + min(
            #     exp_A_vals + exp_B_vals + exp_C_vals)) / 2
            # max_y_axis = (max(exp_A_vals_smoothed + exp_B_vals_smoothed + exp_C_vals_smoothed) + max(
            #     exp_A_vals + exp_B_vals + exp_C_vals)) / 2
            min_y_axis = const*min(exp_A_vals_smoothed + exp_B_vals_smoothed + exp_C_vals_smoothed) \
                + (1-const)*min(exp_A_vals + exp_B_vals + exp_C_vals)
            max_y_axis = const*max(exp_A_vals_smoothed + exp_B_vals_smoothed + exp_C_vals_smoothed) \
                + (1-const)*max(exp_A_vals + exp_B_vals + exp_C_vals)
        else:
            min_y_axis = min(exp_B_vals + exp_C_vals)
            max_y_axis = max(exp_B_vals + exp_C_vals)

        if graph_keys[i] == 'FID':
            ax_i.set_ylim(min_y_axis, max_y_axis)
        

    handles, labels = [], []
    for ax in [ax1]:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.append(handle)
            labels.append(label)

    # Create the legend outside the plot area
    plt.figlegend(handles, labels, loc='lower center', ncol=2, fontsize=9)

    plt.tight_layout()
    plt.subplots_adjust(top=0.80, bottom=0.27, wspace=0.2)
    plt.savefig(output_fname, dpi=300)
    plt.clf()

    return None


def make_graphs_iterative_finetuning(smoothing=False, results_type="", add_smoothing_to_graph=False):

    base_dir = sys.argv[1]
    aug_percents = ["0.2", "0.5", "1.0", "1.5"]
    baseline_dir = os.path.join(base_dir, "baseline_0.2")

    output_dir = os.path.join(base_dir, "graphs")
    os.makedirs(output_dir, exist_ok=True)

    baseline_results = read_results(baseline_dir, results_type=results_type)
    
    for aug_percent in aug_percents:
        
        iterative_finetuning_dir = os.path.join(base_dir, f"iterative_finetuning_{aug_percent}")
        iterative_finetuning_with_correction_dir = os.path.join(base_dir, f"iterative_finetuning_with_correction_{aug_percent}")
        output_fname = f"aug_percent_{aug_percent}_graphs.png"
        iterative_finetuning_results = read_results(iterative_finetuning_dir, results_type=results_type)
        iterative_finetuning_with_correction_results = read_results(iterative_finetuning_with_correction_dir, results_type=results_type)

        # graph FID and Diversity
        build_graphs(
            baseline_results, iterative_finetuning_results, iterative_finetuning_with_correction_results,
            os.path.join(output_dir, output_fname),
            graph_keys=["FID"],
            aug_percent=int(100*float(aug_percent)),
            smoothing=smoothing,
            add_smoothing_to_graph=add_smoothing_to_graph,
            title=f"MNIST Conditional Generation Results\n{int(100*float(aug_percent))}% Synthetic Augmentation"
        )

    return None


def main():
    make_graphs_iterative_finetuning(results_type="latest", add_smoothing_to_graph=True)


if __name__ == "__main__":
    main()
