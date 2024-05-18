from abc import ABC, abstractmethod
from pathlib import Path
from tqdm import tqdm

import scipy
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


class GaussianSelfConsumingLoop(ABC):
    """
    This is a base class for running the self consuming loop experiments.
    Need to implement custom_projection() function in any class that inherits from this
    """

    def __init__(self, n_samples: int) -> None:
        self.n_samples = n_samples
        self.dim = 2
        self.components = 1
        self.centers = [[0, 0]]
        self.cluster_std = 1.0

        self.initial_mean = np.zeros(2)
        self.initial_cov = np.eye(2)
        self.target_mean = np.zeros(2)
        self.target_cov = np.eye(2)

        self.rng = np.random.default_rng()

        # defaults
        self.n_steps = 0
        self.projection_strength = 1.0

    @abstractmethod
    def self_correction(self, data, **kwargs):
        """
        To be implemented by all child classes
        """
        pass

    def run_self_correction_augmentation_loop(self, n_steps, synth_aug_percent):
        """
        Augmentation loop where we sample and apply self-correction, and augment the gt data
        """
        self.n_steps = n_steps

        gt_data = self.rng.multivariate_normal(self.initial_mean, self.initial_cov, self.n_samples)

        # initialize model by fitting it on A
        gmm: GaussianMixture = (
            GaussianMixture(n_components=self.components, tol=1e-9, random_state=1).fit(gt_data)
        )

        means, covs = [gmm.means_[0]], [gmm.covariances_[0]]  # initial means and covariances
        w2s = [self.w2_metric_gaussian(self.initial_mean, self.initial_cov, means[0], covs[0])]

        n_samples_synthetic = int(synth_aug_percent * self.n_samples)

        print(
            f"Running self_correction_augmentation_loop simulation for "
            f"gamma = {(self.projection_strength / (1 - self.projection_strength)):.3f}"
        )

        for i in tqdm(range(n_steps)):
            # get the new parameters computed from the latest iteration of the gaussian mixture model
            mean = gmm.means_[0]
            cov = gmm.covariances_[0]

            # sample synthetic data from Gaussian with new parameters
            synth_data = self.rng.multivariate_normal(mean, cov, n_samples_synthetic)

            # project this synthetic data into a more physically plausible space
            synth_data_projection = self.self_correction(synth_data, step=i)

            # define the augmented dataset
            gt_data_augmented = np.concatenate((gt_data, synth_data_projection), axis=0)

            # re-fit the ground model to the new augmented data
            np.random.shuffle(gt_data_augmented)
            gmm = GaussianMixture(
                n_components=self.components, warm_start=True, random_state=1
            ).fit(gt_data_augmented)

            means.append(gmm.means_[0])
            covs.append(gmm.covariances_[0])
            w2s.append(self.w2_metric_gaussian(self.target_mean, self.target_cov, means[-1], covs[-1]))

        return w2s

    def w2_metric_gaussian(self, mu1, cov1, mu2, cov2):
        """
        Wasserstein-2 metric for two Gaussians
        See e.g. https://djalil.chafai.net/blog/2010/04/30/wasserstein-distance-between-two-gaussians/
        """
        sqrt_cov2 = scipy.linalg.sqrtm(cov2)
        M = scipy.linalg.sqrtm(sqrt_cov2 @ cov1 @ sqrt_cov2)

        mean_component = la.norm(mu1 - mu2) ** 2
        var_component = np.trace(cov1 + cov2 - 2 * M)

        return np.sqrt(mean_component + var_component), mean_component, var_component

class PointwiseCorrection(GaussianSelfConsumingLoop):
    """
    The task here is to start with a Gaussian, and over time, via self-augmentation
    with self-correction move towards the actual distribution.

    The self-correction operation here is based on rejection sampling, according to projection_strength,
    as follows: given a set of n examples to which we want to apply self_correction, we draw a sample of size n 
    using rejection sampling from the pdf that is the weighted average of the empirical pdf of the set and the target distribution,
    where the weight corresponds to the projection strength.
    The pointwise correction on the original data is then the bijection from the data to the sampled data that satisfies
    optimal transport.
    """

    def __init__(
            self,
            n_samples_gt=1000,
            accumulate_synth_examples=False,
            projection_strength=1.0,
            initial_cov=np.eye(2),
            initial_mean=np.zeros(2),
            target_cov=np.eye(2),
            target_mean=np.zeros(2)
    ):
        super().__init__(n_samples_gt)
        self.initial_cov = initial_cov
        self.initial_mean = initial_mean
        self.target_cov = target_cov
        self.target_mean = target_mean
        self.projection_strength = projection_strength

        self.accumulate_synth_examples = accumulate_synth_examples

        if self.accumulate_synth_examples:
            self.accumulated_synth_examples = {}  # initialize empty dict

    def self_correction(self, data, **kwargs):

        empirical_pdf, xedges, yedges = np.histogram2d(data[:, 0], data[:, 1], bins=10, density=True)

        # if n = data.shape[0], then this function call samples n data points from the averaged PDF 
        #     (target_pdf + gamma*empirical_pdf(data))/(1+gamma),
        # and names the data points replacement_data. this effectively maps each point in data to a 
        # point in replacement_data in a way that minimizes the total distance traveled.
        replacement_data = self.rejection_sampling(empirical_pdf, xedges, yedges, data.shape[0])

        if self.accumulate_synth_examples:

            # we accumulate augmentation examples between generations to simulate fine-tuning
            self.accumulated_synth_examples[kwargs["step"]] = replacement_data

            accumulated_synth_examples = np.zeros((0, 2))
            num_generations = len(self.accumulated_synth_examples)

            for i in range(num_generations):
                ith_synth_examples = self.accumulated_synth_examples[i]
                num_include = int(((i + 1) / num_generations) * data.shape[0])
                ith_synth_examples = ith_synth_examples[:num_include]

                accumulated_synth_examples = np.concatenate((accumulated_synth_examples, ith_synth_examples))

            replacement_data = accumulated_synth_examples

        return replacement_data


    def rejection_sampling(self, empirical_pdf, xedges, yedges, n_samples):
        """
        Implements the rejection sampling algorithm spelled out in https://en.wikipedia.org/wiki/Rejection_sampling#Algorithm
        """

        # Use ideal Gaussian for the proposal distribution since it eventually dominates the distribution we want to sample from
        proposal_cov = self.target_cov
        proposal_mean = self.target_mean

        xdelta = xedges[1] - xedges[0]
        ydelta = yedges[1] - yedges[0]
        num_bins = empirical_pdf.shape[0]

        # Compute a bound M on the ratio (pdf of distribution we want to sample from / pdf of proposal distribution)
        max_of_pdf_we_want_to_sample_from = (1-self.projection_strength) * np.max(empirical_pdf) + self.projection_strength * scipy.stats.multivariate_normal.pdf([0,0], self.target_mean, self.target_cov)
        max_of_pdf_of_proposal_distribution = scipy.stats.multivariate_normal.pdf([0,0], proposal_mean, proposal_cov)
        M = max_of_pdf_we_want_to_sample_from / max_of_pdf_of_proposal_distribution

        count = 0
        synth_data = np.zeros((n_samples, 2))
        while count < n_samples:
            candidate = self.rng.multivariate_normal(proposal_mean, proposal_cov)
            u = self.rng.uniform()
            # find x and y index for generated candidate value
            i_x = int((candidate[0] - xedges[0])/ xdelta) 
            i_y = int((candidate[1] - yedges[0])/ ydelta)

            emp = 0.0
            # if indices are not out of bounds assign empirical pdf value else 0
            if not(i_x >= num_bins or i_y >= num_bins or i_x < 0 or i_y < 0):
                emp = empirical_pdf[i_x][i_y] 

            pdf_eval = (1-self.projection_strength) * emp + self.projection_strength * scipy.stats.multivariate_normal.pdf(candidate, self.target_mean, self.target_cov)
            
            if pdf_eval/(M*scipy.stats.multivariate_normal.pdf(candidate, proposal_mean, proposal_cov)) > u:
                synth_data[count] = candidate
                count += 1

        return synth_data


def get_w2(lst_with_tuples):
    lst = [elt[0] for elt in lst_with_tuples]
    return lst


def build_graph(w2s, output_fname: str) -> None:
    max_value = -100.0

    w2s_to_keep = w2s

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    # https://matplotlib.org/stable/gallery/color/named_colors.html
    colors = ["red", "darkorange", "gold", "green", "lightseagreen", "deepskyblue", "mediumpurple", "magenta"]
    if len(w2s_to_keep) > 8:
        colors = ["red", "red", "darkorange", "darkorange", "gold", "gold", "green", "green", "lightseagreen",
                  "lightseagreen", "deepskyblue", "deepskyblue", "mediumpurple", "mediumpurple", "magenta",
                  "magenta", "magenta"] * len(w2s_to_keep)

    w2_vals = []
    for i, key in enumerate(sorted(w2s_to_keep.keys())):
        color = colors[i]

        # get the values to graph for this projection strength
        w2_vals = get_w2(w2s_to_keep[key])

        # update value for graph bounds
        max_value = max(w2_vals + [max_value])

        strength_string = '{:.2f}'.format(key)
        ax.plot(
            [i for i in list(range(len(w2_vals)))],
            w2_vals, c=color, label=f"Correction strength {strength_string}",
        )

    # Set x-axis major ticks to multiples of 100
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.set_xlabel("Generation", fontsize=16)
    ax.set_ylabel(f"Wasserstein distance", fontsize=16)
    ax.grid(True, linewidth=0.3)
    ax.set_xlim(0.0, len(w2_vals))
    ax.set_ylim(0.0, max_value)

    handles, labels = [], []
    for handle, label in zip(*ax.get_legend_handles_labels()):
        handles.append(handle)
        labels.append(label)

    # Create the legend outside the plot area
    plt.figlegend(handles, labels, loc='upper right', ncol=1, fontsize=6)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.16, wspace=0.2)

    # create the enclosing folder on disk if it doesn't exist
    p = Path(output_fname)
    p.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_fname)
    plt.clf()

    temp_str = "-" * 85
    print(f"\n{temp_str}\nWrote results graph with Wasserstein metrics to: {output_fname}\n{temp_str}\n")


def run_exps() -> None:
    n_steps = 500

    initial_cov = np.eye(2)
    initial_mean = np.zeros(2)

    target_cov = np.eye(2)
    target_mean = np.zeros(2)

    accumulate_synth_examples = True

    gamma_values = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 2.0, 5.0]  # values used in the paper
    # NOTE: if you want to see a trend across a finer grid, replace the above with these values instead...
    # gamma_values = [
    #   0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0
    # ]

    n_samples_gt = 50
    synth_aug_percent = 0.5

    w2s = {}
    for projection_strength in gamma_values:
        t_value = projection_strength / (1 + projection_strength)

        scsc_loop = PointwiseCorrection(
            n_samples_gt=n_samples_gt,
            accumulate_synth_examples=accumulate_synth_examples,
            projection_strength=t_value,
            initial_cov=initial_cov,
            initial_mean=initial_mean,
            target_cov=target_cov,
            target_mean=target_mean
        )

        w2_values = scsc_loop.run_self_correction_augmentation_loop(
            n_steps=n_steps,
            synth_aug_percent=synth_aug_percent
        )

        w2s[projection_strength] = w2_values

    # given the w2s, produce a graph and save to disk at this location
    build_graph(w2s, "exp_outputs/gaussian_toy_example.png")


if __name__ == "__main__":
    run_exps()
