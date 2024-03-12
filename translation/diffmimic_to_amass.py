import numpy as np
import torch
import os
import copy

from utils_diffmimic.joint_utils import (
    JOINT_NAMES_SMPL,
    JOINT_NAMES_HUMANOID,
    SMPL2IDX,
    SMPL2HUMANOID,
)
from utils_diffmimic.conversions import *

from utils_diffmimic.system_configs.SMPL import _SYSTEM_CONFIG_SMPL
from google.protobuf import text_format

from brax.physics.base import QP, vec_to_arr
from brax.physics.config_pb2 import Config
from brax.physics import bodies
from brax import jumpy as jp
from brax import math

from scipy.spatial.transform import Rotation, RotationSpline
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline

from amass_to_diffmimic import amass_to_diffmimic, get_rot, interpolate

import io

TOL = 1e-10


def get_system_cfg(system_type):
    return {
        "smpl": _SYSTEM_CONFIG_SMPL,
    }[system_type]


def process_system_cfg(cfg):
    return text_format.Parse(cfg, Config())


CFG_SMPL = process_system_cfg(get_system_cfg("smpl"))

HUMANOID2SMPL = [0] * (len(SMPL2HUMANOID) + 4)
for i in range(len(SMPL2HUMANOID)):
    HUMANOID2SMPL[SMPL2HUMANOID[i]] = i


def state_to_qp(state):
    N = len(CFG_SMPL.bodies)
    qp = QP.zero(shape=(N,))
    pos = np.reshape(state[: 3 * N], (N, 3))
    rot = np.reshape(state[3 * N : 7 * N], (N, 4))
    vel = np.reshape(state[7 * N : 10 * N], (N, 3))
    ang = np.reshape(state[10 * N : 13 * N], (N, 3))
    qp = qp.replace(pos=pos, rot=rot, vel=vel, ang=ang)
    return qp


def recover_data(qp, joint_body, joint_off):
    N = len(joint_body)
    amp_rot = [
        math.quat_mul(math.quat_inv(qp.rot[p]), qp.rot[c]) for (p, c) in joint_body
    ]
    local_pos = [
        joint_off[i][0] - math.rotate(joint_off[i][1], amp_rot[i]) for i in range(N)
    ]

    world_pos_trans = np.zeros((N + 1, 3))
    for i in range(N):
        world_pos_trans[joint_body[i][1]] = world_pos_trans[
            joint_body[i][0]] + math.rotate(local_pos[i], qp.rot[joint_body[i][0]])

    world_ang = [qp.ang[c] for (_, c) in joint_body]
    amp_rot = np.array(amp_rot)
    world_ang = np.array(world_ang)
    amp_rot[np.abs(amp_rot) < TOL] = 0
    world_ang[np.abs(world_ang) < TOL] = 0
    return amp_rot, world_ang, world_pos_trans


def diffmimic_to_amass(demo_traj, frate=30, target_frate=60):
    body = bodies.Body(CFG_SMPL)
    # set any default qps from the config
    joint_idxs = []
    j_idx = 0
    for j in CFG_SMPL.joints:
        beg = joint_idxs[-1][1][1] if joint_idxs else 0
        dof = len(j.angle_limit)
        joint_idxs.append((j, (beg, beg + dof), j_idx))
        j_idx += 1
    lineage = {j.child: j.parent for j in CFG_SMPL.joints}
    depth = {}
    for child, parent in lineage.items():
        depth[child] = 1
        while parent in lineage:
            parent = lineage[parent]
            depth[child] += 1
    joint_idxs = sorted(joint_idxs, key=lambda x: depth.get(x[0].parent, 0))
    joint = [j for j, _, _ in joint_idxs]
    joint_order = [i for _, _, i in joint_idxs]
    # update qp in depth order
    joint_body = jp.array([(body.index[j.parent], body.index[j.child]) for j in joint])
    joint_off = jp.array(
        [(vec_to_arr(j.parent_offset), vec_to_arr(j.child_offset)) for j in joint]
    )

    joint_rot = jp.array([math.euler_to_quat(vec_to_arr(j.rotation)) for j in joint])
    joint_ref = jp.array(
        [math.euler_to_quat(vec_to_arr(j.reference_rotation)) for j in joint]
    )

    fixed = {j.child for j in joint}
    root_idx = {
        b.name: [i] for i, b in enumerate(CFG_SMPL.bodies) if b.name not in fixed
    }
    for j in joint:
        parent = j.parent
        while parent in lineage:
            parent = lineage[parent]
        if parent in root_idx:
            root_idx[parent].append(body.index[j.child])

        def local_rot_ang_inv(_, x):
            angles, vels, rot, ref = x
            axes = jp.vmap(math.rotate, [True, False])(jp.eye(3), math.quat_inv(rot))
            ang = jp.dot(axes.T, vels).T
            rot = ref
            for axis, angle in zip(axes, angles):
                # these are euler intrinsic rotations, so the axes are rotated too:
                axis = math.rotate(axis, rot)
                next_rot = math.quat_rot_axis(axis, angle)
                rot = math.quat_mul(next_rot, rot)
            return (), (rot, ang)

    orig_order = [0] * len(joint)
    for j in range(len(joint)):
        orig_order[joint_order[j]] = j

    trans = []
    poses = []
    print("frames ", len(demo_traj))
    for i in range(len(demo_traj)):
        qp = state_to_qp(demo_traj[i])
        amp_rot, world_ang, world_pos_trans = recover_data(qp, joint_body, joint_off)
        # xs = (amp_rot, world_ang, joint_rot, joint_ref)
        # _, (amp_rot, _) = jp.scan(local_rot_ang_inv, (), xs, len(joint))
        # print("amp_rot ", amp_rot.shape)

        amp_rot = amp_rot[orig_order]
        # print(amp_rot)
        pose0 = math.quat_mul(
            math.quat_inv(math.euler_to_quat(np.array([0.0, -90, 0.0]))), qp.rot[0]
        )
        amp_rot = np.vstack((pose0, amp_rot))
        poses.append(amp_rot)
        world_pos_trans = np.vstack((world_pos_trans, np.zeros(3)))
        trans.append(qp.pos[0])
        

    trans = np.array(trans)
    poses = np.array(poses)

    dt = 1 / frate
    target_dt = 1 / target_frate

    poses = np.stack(
        [get_rot(poses[:, i, :], dt, target_dt) for i in range(poses.shape[1])], axis=1
    )

    poses = quaternion_to_matrix(torch.from_numpy(poses))
    poses = matrix_to_euler_angles(poses, "XYZ")
    poses = euler_angles_to_matrix(poses, "ZXY")
    poses = matrix_to_axis_angle(poses).numpy()
    # poses = poses[:, HUMANOID2SMPL]

    trans, _ = interpolate(trans, dt, target_dt)
    rec_offset_trans = copy.deepcopy(trans[:20])
    trans += 0.05

    return trans, poses, rec_offset_trans


if __name__ == "__main__":
    amass_raw_path = "data/amass/CMU/75/75_09_poses.npz"
    amass_raw = np.load(amass_raw_path)

    print("\n\nEXAMINING INPUT:")
    for k in amass_raw.files:
        print("  ", k, amass_raw[k].shape)
    print(amass_raw["mocap_framerate"])

    print("\n\nEXAMINING OUTPUT:")
    arr_out, trans20, offset_trans20, poses_gt = amass_to_diffmimic(amass_raw)  # diffmimic
    print("arr_out.shape:", arr_out.shape)

    print("\n\nEXAMINING RECOVERED SHAPES:")
    trans, poses, rec_offset_trans = diffmimic_to_amass(arr_out)
    print("trans shape", trans.shape)
    print("poses shape", poses.shape)
    print("\n\nSIDE BY SIDE COMPARISON:")
    print("original offsetted trans[:20]\n", offset_trans20)
    print("recovered offsetted trans[:20]\n", rec_offset_trans)
    print("\n\n")
    print("original trans[:20]\n", trans20)
    print("recovered trans[:20]\n", trans[:20])
    print("\n\n")
    print("original poses[0]\n", poses_gt[1])
    print("recovered poses[0]\n", poses[1])
    # print(poses_gt[1]-poses[1])