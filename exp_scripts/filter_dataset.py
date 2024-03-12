import sys
import os
import shutil
from tqdm import tqdm

import torch
import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.append('.')

from translation.amass_to_mdm import recover_from_ric
from visualize.viz_mdm import visualize_samples_and_save_to_disk

mdm_kinematic_chain = [
    [0, 2, 5, 8, 11],  # pelvis, right hip, right knee, right ankle, right foot
    [0, 1, 4, 7, 10],  # pelvis, left hip, left knee, left ankle, left foot
    [0, 3, 6, 9, 12, 15],  # pelvis, spine1, spine2, spine3, neck, head
    [9, 14, 17, 19, 21],  # spine3, right collar, right shoulder, right elbow, right wrist
    [9, 13, 16, 18, 20]  # spine3, left collar, left shoulder, left elbow, left wrist
]


def mean_variance(data_dir, joints_num):
    file_list = os.listdir(data_dir)
    data_list = []

    for file in file_list:
        data = np.load(os.path.join(data_dir, file))
        if np.isnan(data).any():
            continue
        data_list.append(data)

    data = np.concatenate(data_list, axis=0)
    Mean = data.mean(axis=0)
    Std = data.std(axis=0)
    Std[0:1] = Std[0:1].mean() / 1.0
    Std[1:3] = Std[1:3].mean() / 1.0
    Std[3:4] = Std[3:4].mean() / 1.0
    Std[4: 4 + (joints_num - 1) * 3] = (
            Std[4: 4 + (joints_num - 1) * 3].mean() / 1.0
    )
    Std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] = (
            Std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9].mean() / 1.0
    )
    Std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] = (
            Std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3].mean() / 1.0
    )
    Std[4 + (joints_num - 1) * 9 + joints_num * 3:] = (
            Std[4 + (joints_num - 1) * 9 + joints_num * 3:].mean() / 1.0
    )

    assert 8 + (joints_num - 1) * 9 + joints_num * 3 == Std.shape[-1]

    return Mean, Std


# Converting MoVi style mat file to nested dictionary; copied from BMLmovi:
# source: https://github.com/saeed1262/MoVi-Toolbox/blob/master/MoCap/utils.py
def mat2dict(filename):
    """Converts MoVi mat files to a python nested dictionary.
    This makes a cleaner representation compared to sio.loadmat

    Arguments:
        filename {str} -- The path pointing to the .mat file which contains
        MoVi style mat structs

    Returns:
        dict -- A nested dictionary similar to the MoVi style MATLAB struct
    """
    # Reading MATLAB file
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)

    # Converting mat-objects to a dictionary
    for key in data:
        if key != "__header__" and key != "__global__" and key != "__version__":
            if isinstance(data[key], sio.matlab.mio5_params.mat_struct):
                data_out = matobj2dict(data[key])
    return data_out


# copied from BMLmovi
# source: https://github.com/saeed1262/MoVi-Toolbox/blob/master/MoCap/utils.py
def matobj2dict(matobj):
    """A recursive function which converts nested mat object
    to a nested python dictionaries

    Arguments:
        matobj {sio.matlab.mio5_params.mat_struct} -- nested mat object

    Returns:
        dict -- a nested dictionary
    """
    ndict = {}
    for fieldname in matobj._fieldnames:
        attr = matobj.__dict__[fieldname]
        if isinstance(attr, sio.matlab.mio5_params.mat_struct):
            ndict[fieldname] = matobj2dict(attr)
        elif isinstance(attr, np.ndarray) and fieldname == "move":
            for ind, val in np.ndenumerate(attr):
                ndict[
                    fieldname
                    + str(ind).replace(",", "").replace(")", "").replace("(", "_")
                    ] = matobj2dict(val)
        elif fieldname == "skel":
            tree = []
            for ind in range(len(attr)):
                tree.append(matobj2dict(attr[ind]))
            ndict[fieldname] = tree
        else:
            ndict[fieldname] = attr
    return ndict


def filter_HumanML3D_2974():
    # from HumanML3D repo; associates AMASS name to code name
    file_names = pd.read_csv("dataset/HumanML3D/index.csv")
    v3d_files_dir = "dataset/F_Subjects_1_90"

    file_names_filtered = file_names[file_names["source_path"].str.contains("BMLmovi")]

    for v3d_fname in tqdm(os.listdir(v3d_files_dir)):
        v3d_filename = os.path.join(v3d_files_dir, v3d_fname)
        v3d_file = mat2dict(v3d_filename)
        person_id = v3d_file["id"]
        motions_list = v3d_file["move"]["motions_list"]
        sitting_down_idx = np.where(motions_list == "sitting_down")[0].item()

        # this is the number that we need to remove, keep track of it
        sitting_down_number = sitting_down_idx + 1

        file_names_filtered_temp = file_names_filtered[
            file_names_filtered["source_path"].str.contains(f"{person_id}_")
        ]
        file_names_filtered_temp = file_names_filtered_temp[
            file_names_filtered_temp["source_path"].str.contains(f"F_{sitting_down_number}_poses")
        ]

        if len(file_names_filtered_temp) != 1:
            print(
                f"len(file_names_filtered_temp) = {len(file_names_filtered_temp)} for "
                f"person_id {person_id}, skipping"
            )
            continue
        new_name = file_names_filtered_temp["new_name"].item()

        # remove sample where the person is sitting
        file_names_filtered = file_names_filtered[~file_names_filtered["new_name"].str.contains(new_name)]

    # copy the relevant file names
    file_names_for_subset = []
    for index, row in file_names_filtered.iterrows():
        file_name = row["new_name"]
        file_name_mirrored = "M" + file_name
        file_names_for_subset += [file_name, file_name_mirrored]

    # for every new_name in HumanML3D that is also in our filtered set, 
    # copy it over to the new dataset in new_joints dir; also make sure to get the mirrored versions...
    humanml3d_dir = "dataset/HumanML3D"
    humanml3d_dir_motions = os.path.join(humanml3d_dir, "new_joint_vecs")
    humanml3d_subset_dir = "dataset/HumanML3D_subset_2794"
    humanml3d_subset_dir_motions = os.path.join(humanml3d_subset_dir, "new_joint_vecs")
    os.makedirs(humanml3d_subset_dir_motions, exist_ok=True)
    for fname in tqdm(file_names_for_subset):

        fname_path_src = os.path.join(humanml3d_dir_motions, fname)
        fname_path_dst = os.path.join(humanml3d_subset_dir_motions, fname)

        assert os.path.isfile(fname_path_src)
        if os.path.isfile(fname_path_dst):
            continue

        shutil.copy2(fname_path_src, fname_path_dst)

    # create a new texts dir with just this subset
    humanml3d_dir_texts = os.path.join(humanml3d_dir, "texts")
    humanml3d_subset_dir_texts = os.path.join(humanml3d_subset_dir, "texts")
    os.makedirs(humanml3d_subset_dir_texts, exist_ok=True)

    for fname in tqdm(file_names_for_subset):

        fname_path = os.path.join(humanml3d_dir_texts, fname[:-4] + ".txt")
        assert os.path.isfile(fname_path)
        fname_new_path = os.path.join(humanml3d_subset_dir_texts, fname[:-4] + ".txt")
        if os.path.isfile(fname_new_path):
            continue

        shutil.copy2(fname_path, fname_new_path)

    # create new train.txt, train_val.txt, val.txt, test.txt with just this subset
    # i.e., for each of these files in humanml3d_dir... copy them over directly, 
    # after filtering out the ones that don't appear in file_names_dict (or their mirrors)
    for split in ["train", "train_val", "val", "test"]:
        txt_src = os.path.join(humanml3d_dir, f"{split}.txt")
        assert os.path.isfile(txt_src)
        txt_dst = os.path.join(humanml3d_subset_dir, f"{split}.txt")
        if os.path.isfile(txt_dst):
            continue

        with open(txt_src, 'r') as source_file:
            with open(txt_dst, 'a') as destination_file:
                for line in tqdm(source_file):
                    if f"{line.strip()}.npy" in file_names_for_subset:
                        destination_file.write(line)

    # compute new Mean.npy, Std.npy (copy over their code from HumanML3D...)
    joints_num = 22
    Mean, Std = mean_variance(humanml3d_subset_dir_motions, joints_num)
    np.save(os.path.join(humanml3d_subset_dir, 'Mean.npy'), Mean)
    np.save(os.path.join(humanml3d_subset_dir, 'Std.npy'), Std)

    # create a new folder with visuals of every motion in that folder... so we can verify
    # visually that they only interact with the ground plane!!
    humanml3d_subset_dir_animations = os.path.join(humanml3d_subset_dir, "animations")
    os.makedirs(humanml3d_subset_dir_animations, exist_ok=True)

    for fname in tqdm(os.listdir(humanml3d_subset_dir_motions)[:10]):
        fname_path = os.path.join(humanml3d_subset_dir_motions, fname)
        assert os.path.isfile(fname_path)
        animation_path = os.path.join(humanml3d_subset_dir_animations, fname[:-4] + ".mp4")
        if os.path.isfile(animation_path):
            continue

        motion_seq_hml = np.load(fname_path)

        mdm_data_3d = recover_from_ric(
            torch.from_numpy(motion_seq_hml).unsqueeze(0).float(), 22
        )[0]  # (123, 22, 3)

        # visualize the MDM motion sequence...
        visualize_samples_and_save_to_disk(
            mdm_data_3d.numpy(),
            humanml3d_subset_dir_animations,
            fname[:-4] + ".mp4",
            20,
            humanml3d_subset_dir_animations,
            mdm_kinematic_chain
        )

    return None


def filter_HumanML3D_subset(size):
    """
    Uses the same randomly generated subsets in the paper, for reproducibility.
    """
    # copy over all the files
    src_dir = "dataset/HumanML3D_subset_2794"
    dst_dir = f"dataset/HumanML3D_subset_{size}"
    shutil.copytree(src_dir, dst_dir)

    # then just replace the train split
    src_file = f"assets/train_subsplits/{size}_train.txt"
    dst_file = f"dataset/HumanML3D_subset_{size}/train.txt"
    shutil.copy2(src_file, dst_file)

    return None


def main():
    filter_HumanML3D_2974()

    for dataset_size in ["0064", "0128", "0256"]:
        filter_HumanML3D_subset(dataset_size)


if __name__ == "__main__":
    main()
