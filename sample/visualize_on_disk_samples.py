"""
Counterpart to generate.py.

If we generate samples during:
  training_loop.py -> SelfConsumingTrainLoop

Then the samples are saved to disk by the function:
  data_loaders/humanml/data/dataset.py -> Text2MotionDatasetV2.save_generated_samples_to_disk

The input type to the above function for the samples is T_DATA_DICT. They are saved
directly to disk via the np.save function. To generate visualizations for them, we must
load them from disk and ensure we know all the same information as we have in generate.py
"""
from torch.utils.data import DataLoader

from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.data.dataset import load_pickled_np_dict
from sample.types import GeneratedSampleBatchDict
from train.train_platforms import NoPlatform
from train.training_loop import SelfConsumingTrainLoop
from utils import dist_util as distributed_training_util
from utils.fixseed import fixseed
from utils.model_util import create_model_and_diffusion
from utils.parser_util import autophagous_train_args
from visualize.generated_sample_to_mp4 import visualize_samples_and_save_to_disk


def main() -> None:
    args = autophagous_train_args()
    # --- CONFIG ---
    fixseed(args.seed)
    my_npy_file_path = (
        "dataset/Generated/epoch_7_gen_samps_df2eb7998a10d99ee239a1c1f5b45292.npy"
    )

    data: DataLoader = get_dataset_loader(
        name=args.dataset,
        batch_size=args.batch_size,
        # num_frames defaults to 60 and is the maximum number of frames to use in
        # training. If training with HumanML3D, this field is ignored
        num_frames=args.num_frames,
    )

    # --- initialize models ---
    model, diffusion = create_model_and_diffusion(args, data)
    model.to(distributed_training_util.dev())
    model.rot2xyz.smpl_model.eval()
    print(
        "Diffusion Model Total params: %.2fM"
        % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0)
    )

    training_util = SelfConsumingTrainLoop(
        args, NoPlatform(), model, diffusion, data, util_mode=True
    )

    raw_samples = load_pickled_np_dict(my_npy_file_path)["raw_samples"]
    _, idxs, num_reps, _ = raw_samples
    num_samples = len(idxs)
    all_motions, all_lengths, all_caption, _ = training_util.process_raw_samples(
        raw_samples,
        return_format="to_visualize",
    )

    samples: GeneratedSampleBatchDict = {
        "motion": all_motions,
        "text": all_caption,
        "lengths": all_lengths,
        "num_samples": num_samples,
        "num_repetitions": num_reps,
        "conditioning_idxs": idxs,
    }

    # placeholder
    # name = "g_test"
    # niter = "0"
    # out_path = os.path.join(
    #     os.path.dirname("save/gen_test"),
    #     "samples_{}_{}_seed{}".format(name, niter, args.seed),
    # )
    out_path = "/oscar/data/csun45/txt2vid/mdm/motion-diffusion-model/save/gen_test/samples_g_test_0_seed10"

    visualize_samples_and_save_to_disk(
        args.dataset,
        args.unconstrained,
        samples,
        out_path,
        fps=20,
        num_samples=num_samples,
        num_reps=1,
        batch_size=64,
    )


if __name__ == "__main__":
    main()
