"""
This code is based on https://github.com/openai/guided-diffusion

Originally appeared in sample/generate.py

"""
import os
from typing import List, Tuple

import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
from sample.types import T_SUPPORTED_DATASET_NAMES, GeneratedSampleBatchDict


def visualize_samples_and_save_to_disk(
    dataset: T_SUPPORTED_DATASET_NAMES,
    is_unconstrained: bool,
    samples: GeneratedSampleBatchDict,
    out_path: str,
    fps: float,
    num_samples,
    num_reps,
    batch_size,
) -> str:
    all_text = samples["text"]
    all_lengths = samples["lengths"]
    all_motions = samples["motion"]

    # --- visualizations ---
    print(f"saving visualizations to [{out_path}]...")
    skeleton = (
        paramUtil.kit_kinematic_chain
        if dataset == "kit"
        else paramUtil.t2m_kinematic_chain
    )

    sample_files = []

    # TODO: why is this hard-coded?
    num_samples_in_out_file = 7

    (
        sample_print_template,
        row_print_template,
        all_print_template,
        sample_file_template,
        row_file_template,
        all_file_template,
    ) = construct_template_variables(is_unconstrained)

    # this never reads the samples from disk again, kept in memory
    for sample_i in range(num_samples):
        rep_files = []

        # gather the n reps of each sample
        for rep_i in range(num_reps):
            caption = all_text[rep_i * batch_size + sample_i]
            length = all_lengths[rep_i * batch_size + sample_i]
            motion = all_motions[rep_i * batch_size + sample_i].transpose(2, 0, 1)[
                :length
            ]
            save_file = sample_file_template.format(sample_i, rep_i)
            print(sample_print_template.format(caption, sample_i, rep_i, save_file))

            animation_save_path = os.path.join(out_path, save_file)
            # the below actually saves to disk
            try:
                plot_3d_motion(
                    animation_save_path,
                    skeleton,
                    motion,
                    dataset=dataset,
                    title=caption,
                    fps=fps,
                )
                # Credit for visualization: https://github.com/EricGuo5513/text-to-motion
                rep_files.append(animation_save_path)
            except ValueError as e:
                if str(e).startswith("unknown file extension: "):
                    raise RuntimeError(
                        "Visualization does not have a routine for file extension. Did you load the ffmpeg module? Try running: `module load ffmpeg`."
                    )
                else:
                    raise e

        sample_files = save_multiple_samples(
            out_path,
            row_print_template,
            all_print_template,
            row_file_template,
            all_file_template,
            caption,
            num_samples_in_out_file,
            rep_files,
            sample_files,
            sample_i,
            num_reps,
            num_samples,
        )

    abs_path = os.path.abspath(out_path)
    print(f"[Done] Results are at [{abs_path}]")

    # return sample enclosing folder
    return abs_path


def construct_template_variables(unconstrained: bool) -> Tuple[str]:
    row_file_template = "sample{:02d}.mp4"
    all_file_template = "samples_{:02d}_to_{:02d}.mp4"
    if unconstrained:
        sample_file_template = "row{:02d}_col{:02d}.mp4"
        sample_print_template = "[{} row #{:02d} column #{:02d} | -> {}]"
        row_file_template = row_file_template.replace("sample", "row")
        row_print_template = "[{} row #{:02d} | all columns | -> {}]"
        all_file_template = all_file_template.replace("samples", "rows")
        all_print_template = "[rows {:02d} to {:02d} | -> {}]"
    else:
        sample_file_template = "sample{:02d}_rep{:02d}.mp4"
        sample_print_template = '["{}" ({:02d}) | Rep #{:02d} | -> {}]'
        row_print_template = '[ "{}" ({:02d}) | all repetitions | -> {}]'
        all_print_template = "[samples {:02d} to {:02d} | all repetitions | -> {}]"

    return (
        sample_print_template,
        row_print_template,
        all_print_template,
        sample_file_template,
        row_file_template,
        all_file_template,
    )


def save_multiple_samples(
    out_path: str,
    row_print_template: str,
    all_print_template: str,
    row_file_template: str,
    all_file_template: str,
    caption: str,
    num_samples_in_out_file: int,
    rep_files: List[str],
    sample_files: List[str],
    sample_i: int,
    num_reps: int,
    num_samples,
) -> List[str]:
    all_rep_save_file: str = row_file_template.format(sample_i)
    all_rep_save_path: str = os.path.join(out_path, all_rep_save_file)

    # --- ffmpeg stuff for each n rep of k samples ---
    ffmpeg_rep_files = [f" -i {f} " for f in rep_files]
    hstack_args = f" -filter_complex hstack=inputs={num_reps}" if num_reps > 1 else ""
    ffmpeg_rep_cmd = (
        f"ffmpeg -y -loglevel warning "
        + "".join(ffmpeg_rep_files)
        + f"{hstack_args} {all_rep_save_path}"
    )

    # run ffmpeg to get videos of each repo
    os.system(ffmpeg_rep_cmd)

    # this consolidates all individual reps into a single video
    print(row_print_template.format(caption, sample_i, all_rep_save_file))
    sample_files.append(all_rep_save_path)

    if (sample_i + 1) % num_samples_in_out_file == 0 or (sample_i + 1 == num_samples):
        all_sample_save_file = all_file_template.format(
            sample_i - len(sample_files) + 1, sample_i
        )
        all_sample_save_path = os.path.join(out_path, all_sample_save_file)

        print(
            all_print_template.format(
                sample_i - len(sample_files) + 1, sample_i, all_sample_save_file
            )
        )

        # -- ffmpeg stuff ---
        ffmpeg_rep_files = [f" -i {f} " for f in sample_files]

        # unfortunately when the -filter_complex argument is given, VSCode
        # can't preview the video in the editor, but it does indeed work.
        vstack_args = (
            f" -filter_complex vstack=inputs={len(sample_files)}"
            if len(sample_files) > 1
            else ""
        )
        ffmpeg_rep_cmd = (
            f"ffmpeg -y -loglevel warning "
            + "".join(ffmpeg_rep_files)
            + f"{vstack_args} {all_sample_save_path}"
        )
        os.system(ffmpeg_rep_cmd)

        # ???
        sample_files = []

    return sample_files
