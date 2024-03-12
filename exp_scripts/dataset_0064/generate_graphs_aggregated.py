import os, json
import matplotlib as mpl
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.ndimage import gaussian_filter1d
import numpy as np

# Matplotlib formatting
mpl.rcParams['font.family'] = 'serif'
cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
mpl.rcParams['font.serif'] = cmfont.get_name()
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['axes.unicode_minus'] = False

MAX_NUMBER_OF_GENERATIONS = 50
LATEST_MODELS_CONSIDERED = set()


def read_results(exp_dir, results_type=""):
    def sort_keys(keys):
        keys_single_digit = sorted(
            [key for key in keys if key.__contains__("generation_") and len(key) == 12])
        keys_double_digit = sorted(
            [key for key in keys if key.__contains__("generation_") and len(key) == 13])
        return keys_single_digit + keys_double_digit

    gen_dirs = os.listdir(exp_dir)
    gen_dirs = sort_keys(gen_dirs)[:MAX_NUMBER_OF_GENERATIONS + 1]

    results = {}
    for gen_dir in gen_dirs:
        gen_dir_path = os.path.join(exp_dir, gen_dir)
        eval_dict_path = os.path.join(gen_dir_path, "eval_dict.json")
        if not os.path.isfile(eval_dict_path):
            break
        with open(eval_dict_path, "r") as f:
            eval_dict = json.load(f)

        if results_type == "average":
            # Add logic for "average" if needed
            pass
        elif results_type == "latest":
            latest_model = max(eval_dict, key=lambda k: k)
            LATEST_MODELS_CONSIDERED.add(latest_model)
            results[gen_dir] = eval_dict[latest_model]
        elif results_type == "list_of_all":
            # Add logic for "list_of_all" if needed
            pass

    return results


def sort_keys(keys):
    keys_single_digit = sorted([key for key in keys if key.__contains__("generation_") and len(key) == 12])
    keys_double_digit = sorted([key for key in keys if key.__contains__("generation_") and len(key) == 13])
    return keys_single_digit + keys_double_digit


def build_graphs_for_all_seeds(seeds_results, output_fname, aug_percent, title="title", smoothing=False,
                               add_smoothing_to_graph=False):
    fig, axs = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle(title, fontsize=24)
    graph_keys = ["FID_test", "Diversity_test", "Matching Score_test"]
    y_labels = ["FID", "Diversity", "Matching"]
    SIGMA = 2

    # Colors as per your original specification
    colors = {"Baseline": "peru", "Iterative": "forestgreen", "Iterative with Correction": "tab:blue",
              "generation-0": "tab:red"}

    for row, (seed, results) in enumerate(seeds_results.items()):
        exp_A_results, exp_B_results, exp_C_results = results
        for col, (graph_key, y_label) in enumerate(zip(graph_keys, y_labels)):
            ax = axs[row, col]
            gen0_value = exp_A_results['generation_0'][graph_key] if 'generation_0' in exp_A_results else None
            if gen0_value is not None:
                ax.axhline(y=gen0_value, color=colors["generation-0"], linestyle='--',
                           label='Generation-0' if row == 0 and col == 0 else "")
            for exp_label, exp_results, color in [
                ("Baseline (continue training on gt data), 0%", exp_A_results, colors["Baseline"]),
                ("Iterative fine-tuning, {}%".format(int(aug_percent)), exp_B_results, colors["Iterative"]),
                ("Iterative fine-tuning with self-correction, {}%".format(int(aug_percent)), exp_C_results,
                 colors["Iterative with Correction"])
            ]:
                generations = sort_keys(exp_results.keys())
                values = np.array([exp_results[gen][graph_key] for gen in generations if gen in exp_results])
                if smoothing:
                    values_smoothed = gaussian_filter1d(values, SIGMA, mode='nearest')
                    ax.plot(values_smoothed, label=exp_label if row == 0 and col == 0 else "", color=color)
                ax.plot(values, color=color, alpha=0.4)

                # if row == 0:
                #     ax.set_title("Human Motion Results" if col == 0 else "")
                ax.set_xlabel("Generation" if row == 2 else "")

                # Set x-axis major ticks to multiples of 5
                ax.xaxis.set_major_locator(MultipleLocator(5))
                ax.set_ylabel(f"{graph_key}", fontsize=16)
                ax.grid(True, linewidth=0.3)
                ax.set_xlim(0.0, MAX_NUMBER_OF_GENERATIONS)

                ax.set_ylabel(y_label)
                ax.grid(True)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize='large')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(bottom=0.1)
    plt.savefig(output_fname)
    plt.close()


def make_graphs_iterative_finetuning(smoothing=False, results_type="latest", add_smoothing_to_graph=False):
    subset_size = "0064"
    seeds = ["_seed_10", "_seed_42", "_seed_47"]
    aug_percents = ["025", "050", "075", "100"]

    for aug_percent in aug_percents:
        seeds_results = {}
        for seed_no in seeds:
            # output_dir = f"exp_outputs/dataset_{subset_size}{seed_no}/graphs"
            # os.makedirs(output_dir, exist_ok=True)

            baseline_dir = f"exp_outputs/dataset_{subset_size}{seed_no}/baseline"
            baseline_results = read_results(baseline_dir, results_type=results_type)

            iterative_finetuning_dir = f"exp_outputs/dataset_{subset_size}{seed_no}/synthetic_percent_{aug_percent}_iterative_finetuning"
            iterative_finetuning_with_correction_dir = f"exp_outputs/dataset_{subset_size}{seed_no}/synthetic_percent_{aug_percent}_iterative_finetuning_with_correction"

            iterative_finetuning_results = read_results(iterative_finetuning_dir, results_type=results_type)
            iterative_finetuning_with_correction_results = read_results(
                iterative_finetuning_with_correction_dir, results_type=results_type)

            seeds_results[seed_no] = (
            baseline_results, iterative_finetuning_results, iterative_finetuning_with_correction_results)

        output_dir = f"exp_outputs/dataset_{subset_size}_aggregated/graphs"
        os.makedirs(output_dir, exist_ok=True)
        output_fname = f"{output_dir}/{subset_size}_{aug_percent}_combined_graphs.png"
        build_graphs_for_all_seeds(
            seeds_results, output_fname, aug_percent, smoothing=smoothing,
            add_smoothing_to_graph=add_smoothing_to_graph,
            title=f"Human Motion Generation Results: Dataset Size {int(subset_size)}, with {int(aug_percent)}% Synthetic Augmentation, Across Three Seeds"
        )


def main():
    make_graphs_iterative_finetuning(smoothing=True, results_type="latest", add_smoothing_to_graph=True)


if __name__ == "__main__":
    main()
