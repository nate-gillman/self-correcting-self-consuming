"""
Given a collection of model checkpoints of MDM, for each, sample k propts, with n samples each.
With those samples, produce a 3D rendered video of the sequence.
"""

from pathlib import Path
import os
import numpy as np
import torch
from translation.amass_to_mdm import recover_from_ric
from translation.mdm_to_amass import mdm_to_skin_result, VposerSkinResult
from visualize.pro_render import skin_result_to_video
import sys

# sRGBA color
ORANGE = [1.0, 0.35, 0.0, 1.0]  # A
BLUE = [0.0, 0.7, 1.0, 1.0]  # E
GREEN = [0.5, 1.0, 0.0, 1.0]  # D
GRAY = [0.55, 0.55, 0.55, 1.0]  # GROUND TRUTH

EXP_COLOR = {
    "ground_truth" : GRAY,
    "baseline": ORANGE,
    "iterative_finetuning": GREEN,
    "iterative_finetuning_with_correction": BLUE,
}

def single_motion_to_skin_result(
        single_motion_sequence: np.ndarray, num_joints=22
) -> VposerSkinResult:
    # turn (seq_len, 263) into --> (1, seq_len, 263), and make it a tensor
    motion_formatted = torch.from_numpy(single_motion_sequence).unsqueeze(0).float()

    # recover dimension: (1, seq_len, num_joints, 3)
    # recover_from_ric expects data shape: (1, seq_len, 263)
    pos_data = recover_from_ric(motion_formatted, num_joints).cpu().numpy()

    # pass to skin function as: (1, num_joints, 3, seq_len)
    pos_data = pos_data.transpose(0, 2, 3, 1)

    # skin the result, motions lengths is just [seq_len]
    skin_results = mdm_to_skin_result(pos_data, np.array([pos_data.shape[-1]]))

    # return the only skin result
    return skin_results[0]


def skin_result_to_multiple_videos(
        outputs_dir: Path, base_name: str, skin_result: VposerSkinResult, person_color=None
) -> None:
    # tuples of (camera translation, output filename)
    video_configs = [
        # video filenames are '/009871_${base_name}_view_${a|b}.gif'
        ([0.0, 1.0, 5.0], outputs_dir / (base_name + "_view_a")),
        ([0.0, 0.0, 5.0], outputs_dir / (base_name + "_view_b")),
    ]

    # render n different views of the same motion
    super_suffix = "_super.gif"
    normal_suffix = ".gif"
    for video_config in video_configs:
        _cam, _fname = video_config
        _fname: Path

        print("Saving a video to location: {0}".format(str(_fname)))

        p = Path(str(_fname) + super_suffix)
        if p.is_file():
            # prioritze the super render, overwrite the non-super one
            print(f"A video at {str(p)} already exists! Skipping.")
            continue

        # super
        skin_result_to_video(
            skin_result,
            output_path=str(_fname) + super_suffix,
            # dimensions of the video, (height, width)
            h=1600,
            w=2400,
            # --- the below args are optional, if left as `None`, function uses defaults ---
            #
            # camera_translation is the x, y, z position of the camera. The default value is: (0.0, 2.0, 10.0)
            #
            # here are two presets values for the camera position that seem to work well:
            #   completely level: [0.0, 0.0, 5.0]
            #       the floor plane is a line on the screen, and we can see more clearly float, penetrate, skate physics metrics
            #   visually appealing: [0.0, 1.0, 5.0]
            #       reasonably decent camera angle that showcases an overhead view of someone walking around the floor plane
            camera_translation=_cam,
            # the default color is gray
            person_color_rgb_perc=person_color or GRAY,
            # the default floor color is indigo
            floor_color_rgb_perc=None,
            with_joint_and_snapshots=True,
            snapshot_interval=40,
        )

        # regular
        skin_result_to_video(
            skin_result,
            output_path=str(_fname) + normal_suffix,
            h=800,
            w=1200,
            camera_translation=_cam,
            person_color_rgb_perc=person_color or GRAY,
            floor_color_rgb_perc=None,
            with_joint_and_snapshots=False,
        )

        # warning: if an exception is thrown somewhere in this block, the output video file may get corrupted


# from checkpoint_visual_sampler import single_motion_to_skin_result
# from checkpoint_visual_sampler import skin_result_to_multiple_videos


if __name__ == "__main__":
    """
    If you see this:

        AttributeError: 'NoneType' object has no attribute 'glGetError'

    You may need to do the following in the shell:

        module load mesa patchelf glew
        export PYOPENGL_PLATFORM=osmesa
        export PYTHONPATH=$pwd

    'osmesa' doesn't seem to anti-alias the render. Try removing the `PYOPENGL_PLATFORM` variable
    from the environment to default to pyglet. Pyglet appears to anti-alias the render by default.
    """
    # run this from root mdm dir
    motions_path = Path(str(sys.argv[1])) / "synthetic_motions"
    motions_visualized_path = Path(str(sys.argv[1])) / "synthetic_motions_viz"

    # motions and texts have the same names, sort them for stability (hopefully the filenames are unique)
    motion_paths = sorted(os.listdir(motions_path))

    # --- LOAD GROUND TRUTHS AND RENDER THEM ---
    motions_visualized_path.mkdir(exist_ok=True, parents=True)
    for motion_filename in motion_paths:

        print(f"Processing ground truth motion sequence saved at: {motion_filename}")
        motion_path = os.path.join(motions_path, motion_filename)
        single_motion_sequence = np.load(str(motion_path)) # (71, 263)
        skin_result = single_motion_to_skin_result(single_motion_sequence)

        # e.g. 009871
        base_name = motion_filename.split(".")[0]

        # produce n videos per skin result
        skin_result_to_multiple_videos(motions_visualized_path, base_name, skin_result)