import time
import numpy as np
import torch
from mujoco_py.builder import cymj

from translation.amass_to_mdm import recover_from_ric, amass_to_mdm
from translation.mdm_to_amass import (mdm_to_amass, unconcatenate_amass_rec_concat,
                                      get_poses_and_trans_batches)
from visualize.viz_mdm import visualize_samples_and_save_to_disk
from UniversalHumanoidControl.amass_to_imitation import amass_to_imitation
from UniversalHumanoidControl.uhc.smpllib.smpl_mujoco import qpos_to_smpl
from UniversalHumanoidControl.uhc.smpllib.smpl_robot import Robot

load_model_from_xml = cymj.load_model_from_xml
robot_cfg = {
    'mesh': True, 'model': 'smpl', 'body_params': {},
    'joint_params': {}, 'geom_params': {}, 'actuator_params': {}
}

# load UHC smpl robot.
smpl_robot_orig = Robot(robot_cfg, data_dir="UniversalHumanoidControl/data/smpl")
smpl_model = load_model_from_xml(smpl_robot_orig.export_xml_string().decode("utf-8"))

sep_str = "------------------------------------------------------------------------------------"

mdm_kinematic_dict = {
    0: "pelvis",
    2: "right hip",
    5: "right knee",
    8: "right ankle",
    11: "right foot",
    1: "left hip",
    4: "left knee",
    7: "left ankle",
    10: "left foot",
    3: "spine1",
    6: "spine2",
    9: "spine3",
    12: "neck",
    15: "head",
    14: "right collar",
    17: "right shoulder",
    19: "right elbow",
    21: "right wrist",
    13: "left collar",
    16: "left shoulder",
    18: "left elbow",
    20: "left wrist"
}

mdm_kinematic_chain = [
    [0, 2, 5, 8, 11],  # pelvis, right hip, right knee, right ankle, right foot
    [0, 1, 4, 7, 10],  # pelvis, left hip, left knee, left ankle, left foot
    [0, 3, 6, 9, 12, 15],  # pelvis, spine1, spine2, spine3, neck, head
    [9, 14, 17, 19, 21],  # spine3, right collar, right shoulder, right elbow, right wrist
    [9, 13, 16, 18, 20]  # spine3, left collar, left shoulder, left elbow, left wrist
]

uhc_kinematic_dict_inv = {
    "pelvis": 0,
    "left hip": 1,
    "left knee": 2,
    "left ankle": 3,
    "left foot": 4,
    "right hip": 5,
    "right knee": 6,
    "right ankle": 7,
    "right foot": 8,
    "spine1": 9,
    "spine2": 10,
    "spine3": 11,
    "neck": 12,
    "head": 13,
    "left collar": 14,
    "left shoulder": 15,
    "left elbow": 16,
    "left wrist": 17,
    "right collar": 19,
    "right shoulder": 20,
    "right elbow": 21,
    "right wrist": 22
}

uhc_kinematic_chain = [  # NOTE: the hands are 18 and 23, hence they're ignored
    [0, 5, 6, 7, 8],  # pelvis, right hip, right knee, right ankle, right foot
    [0, 1, 2, 3, 4],  # pelvis, left hip, left knee, left ankle, left foot
    [0, 9, 10, 11, 12, 13],  # pelvis, spine1, spine2, spine3, neck, head
    [11, 19, 20, 21, 22],  # spine3, right collar, right shoulder, right elbow, right wrist
    [11, 14, 15, 16, 17]  # spine3, left collar, left shoulder, left elbow, left wrist
]


def mdm_to_uhc(mdm_data, save_dir=None, save_file_prefixes=None):
    mdm_data_3d_batch = []
    for i in range(len(mdm_data)):
        # forward kinematics, to obtain 3D positional joint representation
        mdm_data_3d = recover_from_ric(
            torch.from_numpy(mdm_data[i]).unsqueeze(0).float(), 22
        )[0]  # (123, 22, 3)

        mdm_data_3d_batch.append(mdm_data_3d)

        # visualize the MDM motion sequence...
        if save_dir:
            visualize_samples_and_save_to_disk(
                mdm_data_3d.numpy(),
                save_dir,
                f"{save_file_prefixes[i]}_01_mdm_original.mp4",
                20,
                f"{save_file_prefixes[i]}_01_MDM_original",
                mdm_kinematic_chain
            )

    motion_lengths = [mdm_data_3d_batch[i].shape[0] for i in range(len(mdm_data_3d_batch))]
    mdm_data_3d_batch_concat = np.concatenate(mdm_data_3d_batch)  # (933, 22, 3)

    # inverse kinematics, to obtain original amass format (axis-angle representation)
    start = time.time()
    amass_rec_concat = mdm_to_amass(  # poses.shape = (372, 66), trans.shape = (372, 3)
        mdm_data_3d_batch_concat,
        model_type="smplh",  # ["smplh", "smplx"]; former is used
        gender="neutral"  # ["male", "female", "neutral"]
    )
    print(
        f"{sep_str}\nvposer total time for this batch: {time.time() - start}, time per motion: {(time.time() - start) / len(motion_lengths)}\n{sep_str}")

    # un-batching them in preparation for individually upscaling...
    amass_rec_batch = unconcatenate_amass_rec_concat(amass_rec_concat, motion_lengths)
    poses_batch, trans_batch = get_poses_and_trans_batches(amass_rec_batch)

    amass_data_batch = []
    for poses, trans in zip(poses_batch, trans_batch):
        # UHC data preprocessing, and imitate that motion sequence using UHC pipeline
        poses_buffered = np.concatenate([poses, np.zeros((poses.shape[0], 6))], axis=1)
        amass_data = {"poses": poses_buffered, "trans": trans}
        amass_data_batch.append(amass_data)

    return amass_data_batch


def uhc_imitation(amass_data_batch, save_dir=None, save_file_prefixes=None):
    start = time.time()
    imitations_batch = amass_to_imitation(amass_data_batch)
    print(
        f"{sep_str}\nUHC total time for this batch: {time.time() - start}, time per motion: {(time.time() - start) / len(amass_data_batch)}\n{sep_str}")

    gt_batch, pred_batch = [], []

    for i, temp_motion_name in enumerate(sorted(imitations_batch.keys())):
        # sorting alphabetically ensures that the ordering of the imitations corresponds
        # to the ordering in amass_data_batch
        gt_motion = imitations_batch[temp_motion_name]["gt"]
        pred_motion = imitations_batch[temp_motion_name]["pred"]
        gt_batch.append(gt_motion)
        pred_batch.append(pred_motion)

        if save_dir:
            gt_jpos_motion = imitations_batch[temp_motion_name]["gt_jpos"]
            pred_jpos_motion = imitations_batch[temp_motion_name]["pred_jpos"]

            visualize_samples_and_save_to_disk(
                gt_jpos_motion, save_dir, f"{save_file_prefixes[i]}_02_uhc-gt_jpos.mp4", 20,
                f"{save_file_prefixes[i]}_02_uhc-gt_jpos", uhc_kinematic_chain)
            visualize_samples_and_save_to_disk(
                pred_jpos_motion, save_dir, f"{save_file_prefixes[i]}_02_uhc-pred_jpos.mp4", 20,
                f"{save_file_prefixes[i]}_02_uhc-pred_jpos", uhc_kinematic_chain)

    return gt_batch, pred_batch


def uhc_to_mdm(gt_batch, pred_batch, save_dir=None, save_file_prefixes=None):
    if save_dir:
        splits_to_compute = {"gt": gt_batch, "pred": pred_batch}
    else:
        splits_to_compute = {"pred": pred_batch}

    def qpos_to_mdm(split_name, qpos, i):

        poses_amass, trans = qpos_to_smpl(qpos, smpl_model, "smpl")
        poses_amass = poses_amass[:, :22, :]
        poses_amass = poses_amass.reshape(poses_amass.shape[0], 22 * 3)
        poses_buffered = np.concatenate([poses_amass, np.zeros((poses_amass.shape[0], 156 - 66))],
                                        axis=1)  # (372, 156)

        amass_reconstructed = {
            "poses": poses_buffered,
            "trans": trans,
            "mocap_framerate": 60.0,
            "gender": "neutral",
            "betas": np.zeros(10)
        }

        (
            pose_seq,  # (124, 263)
            pose_seq_3D  # (124, 22, 3)
        ) = amass_to_mdm(amass_reconstructed)

        if save_dir:
            visualize_samples_and_save_to_disk(pose_seq_3D.numpy(), save_dir,
                                               f"{save_file_prefixes[i]}_03_uhc-{split_name}_qpos_to_amass.mp4",
                                               20,
                                               f"{save_file_prefixes[i]}_03_uhc-{split_name}_qpos_to_amass",
                                               mdm_kinematic_chain)

        return pose_seq

    mdm_format_outputs = {}
    for split_name, qpos_batch in splits_to_compute.items():

        mdm_format_motions = []
        for i, qpos in enumerate(qpos_batch):
            mdm_format = qpos_to_mdm(split_name, qpos, i)
            mdm_format_motions.append(mdm_format)

        mdm_format_outputs[split_name] = mdm_format_motions

    return mdm_format_outputs


def mdm_to_uhc_to_mdm(mdm_data, save_dir=None, save_file_prefixes=None):
    amass_data = mdm_to_uhc(mdm_data, save_dir=save_dir, save_file_prefixes=save_file_prefixes)

    gt_batch, pred_batch = uhc_imitation(amass_data, save_dir=save_dir, save_file_prefixes=save_file_prefixes)

    mdm_formatted_outputs = uhc_to_mdm(gt_batch, pred_batch, save_dir=save_dir,
                                       save_file_prefixes=save_file_prefixes)

    return mdm_formatted_outputs["pred"]


if __name__ == "__main__":
    # load the mdm format data
    mdm_data = np.load("translation/data/mdm/CMU_SMPL_HG/75_09_poses-mdm.npy")  # (123, 263)

    # translate to UHC format, imitate it inside UHC, then translate back to MDM format
    # time(VIS=True) = 1.75 mins; time(VIS=False) = 1.0 mins 
    imitation_mdm = mdm_to_uhc_to_mdm(mdm_data, save_dir="visualize/", save_file_prefixes="")  # (124, 263)
