import os
from typing import Dict, NewType, Any, List
import argparse
import torch
import numpy as np
from functools import lru_cache
from pathlib import Path
from scipy.spatial.transform import Rotation, RotationSpline
from scipy.interpolate import CubicSpline

from translation.amass_to_mdm import amass_to_mdm
from translation.utils_diffmimic.conversions import axis_angle_to_quaternion, quaternion_to_axis_angle
from translation.utils_mdm_to_amass.mdm_motion2smpl import pose_seq_3D_to_axis_angle
from translation.utils_mdm_to_amass.human_body_prior.src.human_body_prior.body_model.body_model import \
    BodyModel

# just to make a more specific type for the output of Vposer
VposerSkinResult = NewType("VposerSkinResult", Dict[Any, Any])

# os.environ['PYOPENGL_PLATFORM'] = 'egl'

# Choose the device to run the body model on.
comp_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_rot(rot, dt, target_dt):
    rot = rot[..., [1, 2, 3, 0]]
    nframe = rot.shape[0]
    x = np.arange(nframe) * dt
    x_target = np.arange(int(nframe * dt / target_dt)) * target_dt

    rotations = Rotation.from_quat(rot)
    spline = RotationSpline(x, rotations)

    return spline(x_target, 0).as_quat()[:, [3, 0, 1, 2]]


def interpolate(y, dt, target_dt):
    x = np.arange(y.shape[0]) * dt
    x_target = np.arange(int(y.shape[0] * dt / target_dt)) * target_dt
    cs = CubicSpline(x, y)
    return cs(x_target)


def get_pose_seq_dict(pose_seq):
    joints_num = 22
    num_samples = 0
    reps = 0
    n_frames = pose_seq.shape[0]
    # print('n_frames ', n_frames)

    pose_seq = np.stack([pose_seq[:, j, :].T for j in range(joints_num)])

    get_pose_seq_dict = {
        # similar to torch.unsqueeze(0)
        'motion': np.array([pose_seq]),
        'text': ['dummy'],
        'lengths': np.array([n_frames]),
        'num_samples': num_samples,
        'num_repetitions': reps
    }

    return get_pose_seq_dict


def mdm_to_amass(pose_seq, model_type='smplh', gender='neutral'):
    print('POSE_SEQ shape', pose_seq.shape)
    pose_seq_dict = get_pose_seq_dict(pose_seq)

    # inverse kinematics step
    amass_rec = pose_seq_3D_to_axis_angle(pose_seq_dict, surface_model_type=model_type, gender=gender)
    return amass_rec


@lru_cache()
def load_body_model(num_betas: int, surface_model_type: str, gender: str,
                    support_dir: str = './body_models') -> BodyModel:
    bm_fname = Path(support_dir) / f'{surface_model_type}/{gender}/model.npz'
    return BodyModel(bm_fname=str(bm_fname), num_betas=num_betas)


def vposer_to_skin_result(vposer_seq: Dict, surface_model_type: str = "smplh",
                          gender: str = "neutral") -> VposerSkinResult:
    """Given the output of a VPoser sequence, skin it using Linear Blend Skinning.

    Loads a (potentially cached) instance of a `BodyModel`. Takes the parameters from a VPoser
    optimization and uses the lbs function from Human Body Prior to produce SMPL mesh vertices,
    faces (normals [I think?]), and global Joint positions. 

    Returns a dictionary, in which there are the following keys:
    - 'v': the XYZ positions of the vertices of the SMPL mesh over time
    - 'f': the faces of the vertices of the SMPL mesh over time, required for video export
    - 'Jtr': the XYZ positions of the joints of the SMPL skeleton over time

    There are other keys in addition to those listed above, but the above are the ones we 
    need to output video and compute physics metrics.
    """
    bm = load_body_model(vposer_seq['num_betas'], surface_model_type, gender)

    # use SMPL to apply linear blend skinning
    res = bm.forward(
        root_orient=torch.from_numpy(vposer_seq['root_orient']),
        pose_body=torch.from_numpy(vposer_seq['pose_body']),
        trans=torch.from_numpy(vposer_seq['trans']),
        pose_hand=None,
        pose_jaw=None,
        pose_eye=None,
        betas=None,
        dmpls=None,
        expression=None,
        return_dict=True
    )

    return res


def mdm_to_skin_result(batched_motions, motion_lengths, surface_model_type: str = "smplh",
                       gender: str = "neutral") -> \
        List[VposerSkinResult]:
    """Given an MDM output as a numpy array, compute the skinned model parmaters of a specific SMPL model.

    Uses VPoser and the `mdm_to_amass` function to produce SMPL parameters. Then does a forward
    pass of the Human Body Prior `BodyModel` with no shape (beta) parameters. 

    batched_motions is an array of (bs, n_joints, 3, seq_length)

    returns a list of size bs, where each element is a skin result from vposer
    """
    # expect concatenation of several MDM motions of the form: (bs, n_joints, 3, seq_length)
    # transpose individual motions to be of the form: (seq_len, n_joints, 3)
    t = [batched_motions[i].transpose(2, 0, 1)[:motion_lengths[i]] for i in range(len(batched_motions))]

    # get the motion lengths
    motion_lengths = [x.shape[0] for x in t]

    # (bs * seq_len, 22, 3)
    mdm_data_3d_batch_concat = np.concatenate(t)

    # given raw MDM sequence(s), use Vposer to convert to AMASS. 
    amass_result = mdm_to_amass(mdm_data_3d_batch_concat, surface_model_type, gender)

    # unconcatenate the batches of converted mdm -> amass
    amass_rec_batch = unconcatenate_amass_rec_concat(amass_result, motion_lengths)

    results = []
    for i, vposer_seq in enumerate(amass_rec_batch):
        # check that the lengths match up with the pose sequence time / frames dimension
        assert motion_lengths[i] == vposer_seq["poses"].shape[0]

        # Ensure that the Betas (shape params) are set to 0 for all
        res = vposer_to_skin_result(vposer_seq, surface_model_type, gender)
        results.append(res)

    # return all the results
    return results


def unconcatenate_amass_rec_concat(amass_rec_concat, motion_lengths, num_betas: int = 16):
    assert amass_rec_concat["trans"].shape[0] == amass_rec_concat["poses"].shape[0] == sum(motion_lengths)
    assert len(motion_lengths) > 0

    motion_lengths_cumulative = [sum(motion_lengths[:i]) for i in range(len(motion_lengths) + 1)]
    amass_rec_batch = []
    for i in range(len(motion_lengths)):
        start_idx, end_idx = motion_lengths_cumulative[i], motion_lengths_cumulative[i + 1]

        # print(start_idx, end_idx, end_idx-start_idx)
        poses = amass_rec_concat["poses"][start_idx:end_idx, :]
        trans = amass_rec_concat["trans"][start_idx:end_idx, :]
        root_orient = amass_rec_concat["root_orient"][start_idx:end_idx, :]
        pose_body = amass_rec_concat["pose_body"][start_idx:end_idx, :]

        amass_rec = {
            "poses": poses,
            "trans": trans,
            # these are required for vertex skinning
            "root_orient": root_orient,
            "pose_body": pose_body,
            "motion_length": motion_lengths[i],
            "num_betas": num_betas
        }
        amass_rec_batch.append(amass_rec)

    return amass_rec_batch


def upscale_framerate(amass_recs_batch, frate=20, target_frate=60):
    poses_batch, trans_batch = [], []
    for amass_rec in amass_recs_batch:
        # amass_rec = np.load(path+'sample00_rep00.npz') # (123, 165)
        poses = amass_rec['poses'][:, :66]
        poses = poses.reshape((poses.shape[0], -1, 3))
        trans = amass_rec['trans']

        dt = 1 / frate
        target_dt = 1 / target_frate

        poses = np.vstack((poses, poses[-1].reshape(-1, 22, 3)))
        trans = np.vstack((trans, trans[-1].reshape(-1, 3)))

        poses = axis_angle_to_quaternion(torch.from_numpy(poses))
        poses = np.stack(
            [get_rot(poses[:, i, :], dt, target_dt) for i in range(poses.shape[1])], axis=1
        )
        poses = quaternion_to_axis_angle(torch.from_numpy(poses))
        poses = poses.numpy()
        poses = poses.reshape(-1, 22 * 3)

        trans = interpolate(trans, dt, target_dt)

        poses_batch.append(poses)
        trans_batch.append(trans)

    return poses_batch, trans_batch


def get_poses_and_trans_batches(amass_recs_batch):
    poses_batch, trans_batch = [], []
    for amass_rec in amass_recs_batch:
        # amass_rec = np.load(path+'sample00_rep00.npz') # (123, 165)
        poses = amass_rec['poses'][:, :66]
        poses = poses.reshape((poses.shape[0], -1, 3))
        trans = amass_rec['trans']

        frate = target_frate = 20.0
        dt = 1 / frate
        target_dt = 1 / target_frate

        poses = np.vstack((poses, poses[-1].reshape(-1, 22, 3)))
        trans = np.vstack((trans, trans[-1].reshape(-1, 3)))

        poses = axis_angle_to_quaternion(torch.from_numpy(poses))
        poses = np.stack(
            [get_rot(poses[:, i, :], dt, target_dt) for i in range(poses.shape[1])], axis=1
        )
        poses = quaternion_to_axis_angle(torch.from_numpy(poses))
        poses = poses.numpy()
        poses = poses.reshape(-1, 22 * 3)

        trans = interpolate(trans, dt, target_dt)

        poses_batch.append(poses)
        trans_batch.append(trans)

    return poses_batch, trans_batch


# ----


def parse_args():
    parser = argparse.ArgumentParser(prog='MDM --> AMASS')
    parser.add_argument(
        "--model-type",
        default='smplh',
        choices=['smplh', 'smplx'],
        type=str,
        help="The desired output model type"
    )
    parser.add_argument(
        "--gender",
        default='neutral',
        type=str,
        choices=['neutral', 'female', 'male'],
        help="Gender"
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    amass_raw_path = 'translation/data/amass/CMU_SMPL_HG/75_09_poses.npz'

    amass_raw = np.load(amass_raw_path)
    (
        pose_seq,  # (123, 263); 263-dim is input and output from the MDM model
        pose_seq_3d  # (123, 22, 3); (22, 3) represents 3D cartesian coordinates
    ) = amass_to_mdm(amass_raw)
    # np.save("translation/data/mdm/CMU_SMPL_HG/75_09_poses-mdm.npy", pose_seq)

    dirs = os.path.normpath(amass_raw_path).split(os.sep)
    subdir = dirs[4].split('.')[0]
    write_path = os.path.join("translation/data/mdm/", dirs[3], subdir) + "/"

    print("write_path: ", write_path)
    if not os.path.exists(write_path):
        os.makedirs(write_path)

    print('args.model_type" ', args.model_type)
    print('args.gender: ', args.gender)
    (
        amass_rec,
        # amass_rec.files = ['trans', 'betas', 'root_orient', 'poZ_body', 'pose_body', 'poses', 'surface_model_type', 'gender', 'mocap_framerate', 'num_betas']
        poses,  # (372, 22, 3)
        trans  # (372, 3)
    ) = mdm_to_amass(
        pose_seq_3d,
        write_path,
        frate=20,
        target_frate=amass_raw['mocap_framerate'],
        model_type=args.model_type,
        gender=args.gender
    )
