from typing import List, Any
import numpy as np
from pathlib import Path
from sample.types import Prompts
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
from translation.mdm_to_amass import mdm_to_skin_result, VposerSkinResult
from visualize.pro_render import skin_result_to_video
from sample.generate_scsc import (
    load_model_from_checkpoint_and_sample
)


def _check_folder(save_to_folder: Any) -> None:
    if not isinstance(save_to_folder, Path):
        raise TypeError(
            "Please provide the folder to save videos as a Path object. "
            "Do: from pathlib import Path")

    if not save_to_folder.is_dir():
        raise RuntimeError(
            f"The folder where we wish to save videos is not a directory. "
            f"Please check the path: {save_to_folder}"
        )


def save_videos_as_wireframes(prompts: Prompts, motions: np.ndarray, save_to_folder: Path) -> List[Path]:
    _check_folder(save_to_folder)

    # using humanml3d
    skeleton = paramUtil.t2m_kinematic_chain

    if len(prompts) != motions.shape[0]:
        raise ValueError("Prompts and motions must be the same length.")

    output_videos_paths = []
    for i in range(len(prompts)):
        prompt = prompts[i]
        motion = motions[i].transpose(2, 0, 1)

        caption, length = prompt

        animation_save_path = save_to_folder / f"wireframe_video_prompt_{i}.mp4"

        try:
            # produce a video of this motion at the specified path
            # Credit for visualization: https://github.com/EricGuo5513/text-to-motion
            plot_3d_motion(
                str(animation_save_path),
                skeleton,
                motion,
                dataset="humanml",
                title=caption,
                fps=20,
            )

            # keep track of the paths for each video
            output_videos_paths.append(animation_save_path)
        except ValueError as e:
            if str(e).startswith("unknown file extension: "):
                raise RuntimeError(
                    "Visualization does not have a routine for file extension. "
                    "Is ffmpeg installed?"
                )
            else:
                raise e

    return output_videos_paths


def save_videos_as_skinned_person(prompts: Prompts, motions: np.ndarray, save_to_folder: Path,
                                  with_joint_and_snapshots: bool = True,
                                  vposer_batch_size: int = 32) -> List[Path]:
    _check_folder(save_to_folder)
    num_motions = motions.shape[0]
    b = vposer_batch_size

    saved_video_paths = []

    for i in range(0, num_motions, b):
        # get captions corresponding to these samples
        batched_prompts = prompts[i: i + b]
        batched_lengths = [x[1] for x in batched_prompts]

        # get the motions for these samples
        batched_motions = motions[i: i + b]

        # skin them with vposer
        skin_results: List[VposerSkinResult] = mdm_to_skin_result(
            batched_motions, np.array(batched_lengths)
        )

        for j, skin_result in enumerate(skin_results):
            p = save_to_folder / f"skinned_video_{j}.gif"
            skin_result_to_video(
                skin_result,
                str(p),
                800,
                1200,
                camera_translation=[0.0, 1.0, 5.0],
                with_joint_and_snapshots=with_joint_and_snapshots
            )
            saved_video_paths.append(p)

    return saved_video_paths


if __name__ == "__main__":
    """This is a simple test to see if model sampling and visualizing works correctly.
    """
    # the best model from the MDM repo, for example
    path_name = "./save/humanml_trans_enc_512"
    my_model_folder = Path(path_name)

    if not my_model_folder.exists():
        raise RuntimeError(
            f"Model folder: {my_model_folder} does not exist. Cannot visualizer test.")

    # using the model, make some prompts and then generate samples from them
    my_prompts = [("the person is jumping up and down", 196),
                  ("the person sits down and does not move", 100)]
    motion_result, _, my_model_util = load_model_from_checkpoint_and_sample(
        my_prompts,
        model_folder=my_model_folder
    )

    # save videos to here
    save_videos_dir = Path("./test_videos")

    # --- generate videos in a wireframe style ---
    wireframe_paths = save_videos_as_wireframes(my_prompts, motion_result, save_videos_dir)
    print("Successfully saved wireframe videos to:")
    print(wireframe_paths)

    # --- generate videos using a fully skinned, humanoid model ---
    skinned_paths = save_videos_as_skinned_person(my_prompts, motion_result, save_videos_dir)
    print("Successfully saved skinned videos to:")
    print(skinned_paths)
