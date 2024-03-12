# system path is '/oscar/data/superlab/users/nates_stuff/motion-diffusion-model/scripts', want to go down one
import sys
for pth in sys.path:
    if "motion-diffusion-model/scripts" in pth:
        sys.path.append(pth[:-8]) 
        break

import argparse
import os
from os.path import join as pjoin
import numpy as np
from mdm_to_uhc_to_mdm import mdm_to_uhc_to_mdm

import time
from tqdm import tqdm

sep_str = "------------------------------------------------------------------------------------"


def get_prompts_from(idx_to_caption):

    prompts = []
    for idx, caption in idx_to_caption.items():
        motion_dict = {}
        motion_dict["text"] = caption
        motion_dict["file"] = idx + "-sampled.npy"
        prompts.append(motion_dict)

    return prompts


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--motions_dir",
        required=True,
        type=str,
    )

    # for when we want to restrict to a subset
    parser.add_argument(
        "--remainder",
        required=False,
        type=int,
        default=0
    )
    parser.add_argument(
        "--modulus",
        required=False,
        type=int,
        default=1
    )
    
    args = parser.parse_args()

    return args


def uhc_correction_batch(
        motions_dir, 
        RECOMPUTE_AND_OVERWRITE=True, 
        VIS=False, 
        MAX_BATCH_DURATION=6400, 
        remainder=0, 
        modulus=1
    ):

    # STEP 1: read all files inside motions_dir
    files_in_motion_dir = os.listdir(motions_dir)
    if RECOMPUTE_AND_OVERWRITE:
        sampled_motion_files = [
            file for file in files_in_motion_dir 
            if file.endswith("-sampled.npy")
        ]
    else:
        sampled_and_imitated_motion_files = [
            file for file in  files_in_motion_dir
            if file.endswith("-sampled_and_imitated.npy")]
        sampled_motion_files = [ # we don't want to recompute ones that have already been computed...
            file for file in files_in_motion_dir 
            if file.endswith("-sampled.npy") and file[:-4]+"_and_imitated.npy" not in sampled_and_imitated_motion_files
        ]
    sampled_motion_files = sorted(sampled_motion_files)

    # compute motion durations
    sampled_motion_files_to_durations = {}
    MIN_LEN, MAX_LEN = 40, 196
    print("Computing lengths of motion files...")
    for sampled_motion_file in tqdm(sampled_motion_files):
        motion_duration = np.load(pjoin(motions_dir, sampled_motion_file)).shape[0]
        if motion_duration in range(MIN_LEN, MAX_LEN+1):
            sampled_motion_files_to_durations[sampled_motion_file] = motion_duration

    # batch em up
    batches = []
    batch = {}
    batch_duration = 0
    print("Batching motion files...")
    for motion_file, motion_duration in sampled_motion_files_to_durations.items():


        if "M" in motion_file:
            idx = motion_file.split("-")[0][1:]     # 'M000001-sampled.npy' --> '000001'
            idx_name = motion_file.split("-")[0]    # 'M000001-sampled.npy' --> 'M000001'
        else:
            idx = motion_file.split("-")[0] # '000001-sampled.npy' --> '000001'
            idx_name = idx
        # for restricting to a subset; useful for batch processing
        if int(idx) % modulus != remainder:
            continue

        if batch_duration + motion_duration < MAX_BATCH_DURATION:
            # add to batch dict, increase counter
            batch[motion_file] = motion_duration
            batch_duration += motion_duration
        else:
            batches.append(batch)
            # reset counter and batch dict
            batch = {motion_file : motion_duration}
            batch_duration = motion_duration
    batches.append(batch)

    # pass a batch at a time into the mdm_to_uhc_to_mdm pipeline
    for batch in batches:

        start = time.time()

        motions = []
        save_file_prefixes = []
        durations = []
        for fname, duration in batch.items():

            if "M" in fname:
                idx = fname.split("-")[0][1:]     # 'M000001-sampled.npy' --> '000001'
                idx_name = fname.split("-")[0]    # 'M000001-sampled.npy' --> 'M000001'
            else:
                idx = fname.split("-")[0] # '000001-sampled.npy' --> '000001'
                idx_name = idx
            # for restricting to a subset; useful for batch processing
            if int(idx) % modulus != remainder:
                continue

            save_file_prefixes.append(idx_name+"-sampled")
            durations.append(duration)

            src_path = pjoin(motions_dir, fname)
            motion = np.load(src_path) # (196, 263)
            motions.append(motion)
             

        # run translation + imitation pipeline
        if VIS: # time with visualization: 144.58s
            motions_imitated_batch = mdm_to_uhc_to_mdm(
                motions, 
                save_dir=motions_dir, 
                save_file_prefixes=save_file_prefixes
            )
        else:   # time without visualization: 53.55s
            motions_imitated_batch = mdm_to_uhc_to_mdm(motions)

        for i, (fname, _) in enumerate(batch.items()):

            if "M" in fname:
                idx = fname.split("-")[0][1:]     # 'M000001-sampled.npy' --> '000001'
                idx_name = fname.split("-")[0]    # 'M000001-sampled.npy' --> 'M000001'
            else:
                idx = fname.split("-")[0] # '000001-sampled.npy' --> '000001'
                idx_name = idx
            # for restricting to a subset; useful for batch processing
            if int(idx) % modulus != remainder:
                continue

            dst_path = pjoin(motions_dir, idx_name+"-sampled_and_imitated.npy")
            motion_mdm_sampled_imitation = motions_imitated_batch[i]
            np.save(dst_path, motion_mdm_sampled_imitation)
            print(f"... just wrote UHC-corrected motion to {dst_path}")

        print(f"{sep_str}\nBATCH TIME = {time.time() - start}, TIME PER MOTION = {(time.time() - start) / len(batch)}\n{sep_str}")


def main():

    # STEP 0: parse args
    args = parse_args()
    RECOMPUTE_AND_OVERWRITE = True
    VIS = False
    MAX_BATCH_DURATION = 6400 # 6400

    uhc_correction_batch(
        args.motions_dir, 
        RECOMPUTE_AND_OVERWRITE=RECOMPUTE_AND_OVERWRITE, 
        VIS=VIS, 
        MAX_BATCH_DURATION=MAX_BATCH_DURATION, 
        remainder=args.remainder, 
        modulus=args.modulus
    )


    return None


if __name__ == "__main__":

    main()