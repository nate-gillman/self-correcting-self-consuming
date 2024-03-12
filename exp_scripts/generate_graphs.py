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
    gen_dirs = os.listdir(exp_dir)
    gen_dirs = sort_keys(gen_dirs)[:MAX_NUMBER_OF_GENERATIONS + 1]

    results = {}
    for i, gen_dir in enumerate(
            gen_dirs):  # SHOULD INCLIDE GENERATION_0!! ITll make graphs more intuitive... cuz B and C will agree!!
        gen_dir_path = os.path.join(exp_dir, gen_dir)

        eval_dict_path = os.path.join(gen_dir_path, "eval_dict.json")
        if not os.path.isfile(eval_dict_path):
            break
        with open(eval_dict_path, "r") as f:
            eval_dict = json.load(f)

        # if i > 0 and checkpoint_to_check_for not in eval_dict.keys():
        #     break # in this case, this is not a complete experiment run

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
            latest_model = max(eval_dict, key=lambda k: k)
            LATEST_MODELS_CONSIDERED.add(latest_model);
            print(LATEST_MODELS_CONSIDERED)
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


def sort_keys(keys):
    keys_single_digit = sorted([key for key in keys if key.__contains__("generation_") and len(key) == 12])
    keys_double_digit = sorted([key for key in keys if key.__contains__("generation_") and len(key) == 13])

    sorted_keys = keys_single_digit + keys_double_digit

    return sorted_keys


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
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))

    fig.suptitle(title, fontsize=24, y=0.95)

    for i, ax_i in enumerate((ax1, ax2, ax3)):

        # get the values for the ith graph
        graph_i_values = defaultdict(list)
        for generation in sort_keys(exp_A_results.keys()):
            result_value = exp_A_results[generation][graph_keys[i]]
            graph_i_values[f"exp_A_{graph_keys[i]}"].append(result_value)
        for generation in sort_keys(exp_B_results.keys()):
            result_value = exp_B_results[generation][graph_keys[i]]
            graph_i_values[f"exp_B_{graph_keys[i]}"].append(result_value)
        for generation in sort_keys(exp_C_results.keys()):
            result_value = exp_C_results[generation][graph_keys[i]]
            graph_i_values[f"exp_C_{graph_keys[i]}"].append(result_value)
        exp_A_vals = graph_i_values[f"exp_A_{graph_keys[i]}"]
        exp_B_vals = graph_i_values[f"exp_B_{graph_keys[i]}"]
        exp_C_vals = graph_i_values[f"exp_C_{graph_keys[i]}"]

        SIGMA = 2
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
        alpha = 0.4

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
            min_y_axis = (min(exp_A_vals_smoothed + exp_B_vals_smoothed + exp_C_vals_smoothed) + min(
                exp_A_vals + exp_B_vals + exp_C_vals)) / 2
            max_y_axis = (max(exp_A_vals_smoothed + exp_B_vals_smoothed + exp_C_vals_smoothed) + max(
                exp_A_vals + exp_B_vals + exp_C_vals)) / 2
        else:
            min_y_axis = min(exp_B_vals + exp_C_vals)
            max_y_axis = max(exp_B_vals + exp_C_vals)

        if graph_keys[i] == 'FID_test':
            ax_i.set_ylim(0.0, max_y_axis + 0.15)
        elif graph_keys[i] == 'Diversity_test':
            ax_i.set_ylim(min_y_axis - 0.25, max_y_axis + 0.25)
        elif graph_keys[i] == 'Matching Score_test':
            ax_i.set_ylim(min_y_axis - 0.25, max_y_axis + 0.25)

    handles, labels = [], []
    for ax in [ax1, ax2, ax3]:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.append(handle)
            labels.append(label)

    # Create the legend outside the plot area
    plt.figlegend(handles, labels, loc='lower center', ncol=4, fontsize=16)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.27, wspace=0.2)

    plt.savefig(output_fname)
    plt.clf()

    return None


def make_graphs_iterative_finetuning(smoothing=False, results_type="", add_smoothing_to_graph=False):

    subset_size = str(sys.argv[1])
    if subset_size in ["0064", "0128", "0256"]:
        aug_percents = ["025", "050", "075", "100"]
    elif subset_size in ["2974"]:
        aug_percents = ["05", "10", "15", "20", "25"]
    else:
        raise NotImplementedError

    for seed_no in [""]: # for optional suffix to the directory, e.g. "_seed_10" --> "exp_outputs/dataset_0064_seed_10"

        output_dir = f"exp_outputs/dataset_{subset_size}{seed_no}/graphs"
        os.makedirs(output_dir, exist_ok=True)

        baseline_dir = f"exp_outputs/dataset_{subset_size}{seed_no}/baseline"
        baseline_results = read_results(baseline_dir, results_type=results_type)

        for aug_percent in aug_percents:
            iterative_finetuning_dir = f"exp_outputs/dataset_{subset_size}{seed_no}/synthetic_percent_{aug_percent}_iterative_finetuning"
            iterative_finetuning_with_correction_dir = f"exp_outputs/dataset_{subset_size}{seed_no}/synthetic_percent_{aug_percent}_iterative_finetuning_with_correction"
            output_fname = f"{subset_size}_{aug_percent}_iterative_finetuning_graphs.png"

            iterative_finetuning_results = read_results(iterative_finetuning_dir, results_type=results_type)
            iterative_finetuning_with_correction_results = read_results(
                iterative_finetuning_with_correction_dir, results_type=results_type)

            # graph FID and Diversity
            build_graphs(
                baseline_results, iterative_finetuning_results, iterative_finetuning_with_correction_results,
                os.path.join(output_dir, output_fname),
                graph_keys=["FID_test", "Diversity_test", "Matching Score_test"],
                aug_percent=int(aug_percent),
                smoothing=smoothing,
                add_smoothing_to_graph=add_smoothing_to_graph,
                title=f"Human Motion Generation Results: Dataset Size {int(subset_size)}, with {int(aug_percent)}% Synthetic Augmentation"
            )

    return None


def main():
    make_graphs_iterative_finetuning(results_type="latest", add_smoothing_to_graph=True)


if __name__ == "__main__":
    main()
