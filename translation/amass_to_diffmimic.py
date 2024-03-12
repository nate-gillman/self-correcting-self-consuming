# Cannibalized from https://github.com/nate-gillman/diffmimic/blob/main/data/tools/amass_converter.py

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
from utils_diffmimic.conversions import (
    axis_angle_to_matrix,
    matrix_to_euler_angles,
    matrix_to_quaternion,
    euler_angles_to_matrix,
    quaternion_to_euler,
)
from utils_diffmimic.quaternion import (
    quat_identity_like,
    quat_mul_norm,
    quat_inverse,
    quat_angle_axis,
)

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

import matplotlib.pyplot as plt
import io


def get_system_cfg(system_type):
    return {
        "smpl": _SYSTEM_CONFIG_SMPL,
    }[system_type]


def process_system_cfg(cfg):
    return text_format.Parse(cfg, Config())


CFG_SMPL = process_system_cfg(get_system_cfg("smpl"))


def convert_to_qp(ase_poses, ase_vel, ase_ang, pelvis_trans):
    qp_list = []

    body = bodies.Body(CFG_SMPL)
    # set any default qps from the config
    joint_idxs = []
    j_idx = 0
    for j in CFG_SMPL.joints:
        beg = joint_idxs[-1][1][1] if joint_idxs else 0
        dof = len(j.angle_limit)  # dof = degree of freedom
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

    num_joint_dof = sum(len(j.angle_limit) for j in CFG_SMPL.joints)
    num_joints = len(CFG_SMPL.joints)
    takes = []
    for j, (beg, end), _ in joint_idxs:
        arr = list(range(beg, end))
        arr.extend([num_joint_dof] * (3 - len(arr)))
        takes.extend(arr)
    takes = jp.array(takes, dtype=int)

    # build local rot and ang per joint
    joint_rot = jp.array([math.euler_to_quat(vec_to_arr(j.rotation)) for j in joint])
    joint_ref = jp.array(
        [math.euler_to_quat(vec_to_arr(j.reference_rotation)) for j in joint]
    )  # joint_ref shape (17,4),

    fixed = {j.child for j in joint}
    root_idx = {
        b.name: [i]
        for i, b in enumerate(CFG_SMPL.bodies)
        if b.name not in fixed
    }
    for j in joint:
        parent = j.parent
        while parent in lineage:
            parent = lineage[parent]
        if parent in root_idx:
            root_idx[parent].append(body.index[j.child])

    for i in range(ase_poses.shape[0]):
        qp = QP.zero(shape=(len(CFG_SMPL.bodies),))
        # QP is DS that represents body in diffmimic environment
        # pos: Location of center of mass.
        # rot: Rotation about center of mass, represented as a quaternion.
        # vel: Velocity.
        # ang: Angular velocity about center of mass.

        local_rot = ase_poses[i][1:]  # ----
        world_vel = ase_vel[i][1:]
        world_ang = ase_ang[i][1:]

        def init(qp):
            pos = jp.index_update(qp.pos, 0, pelvis_trans[i])
            # pos.shape = (19,3)
            rot = ase_poses[i][0] / jp.norm(ase_poses[i][0])  # important
            # rot now has the normalized root_orientation
            rot = math.quat_mul(math.euler_to_quat(np.array([0.0, -90, 0.0])), rot)
            rot = jp.index_update(qp.rot, 0, rot)
            vel = jp.index_update(qp.vel, 0, ase_vel[i][0])
            ang = jp.index_update(qp.ang, 0, ase_ang[i][0])
            qp = qp.replace(pos=pos, rot=rot, vel=vel, ang=ang)
            return qp

        qp = init(qp)
        amp_rot = local_rot[joint_order]  # shape (17,4)
        world_vel = world_vel[joint_order]  # shape (17,3)
        world_ang = world_ang[joint_order]  # shape (17,3)

        def to_dof(a):
            b = np.zeros([num_joint_dof])
            for idx, (j, (beg, end), _) in enumerate(joint_idxs):
                b[beg:end] = a[idx, : end - beg]
            return b

        def to_3dof(a):
            a = jp.concatenate([a, jp.array([0.0])])
            a = jp.take(a, takes)
            a = jp.reshape(a, (num_joints, 3))
            return a

        def local_rot_ang(_, x):
            angles, vels, rot, ref = x
            axes = jp.vmap(math.rotate, [True, False])(jp.eye(3), rot)
            ang = jp.dot(axes.T, vels).T
            rot = ref
            for axis, angle in zip(axes, angles):
                # these are euler intrinsic rotations, so the axes are rotated too:
                axis = math.rotate(axis, rot)
                next_rot = math.quat_rot_axis(axis, angle)
                rot = math.quat_mul(next_rot, rot)
            return (), (rot, ang)

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

        amp_rot = quaternion_to_euler(amp_rot)
        xs = (amp_rot, world_ang, joint_rot, joint_ref)
        _, (amp_rot, _) = jp.scan(local_rot_ang_inv, (), xs, len(joint))
        amp_rot = quaternion_to_euler(amp_rot)
        amp_rot = to_3dof(to_dof(amp_rot))

        xs = (amp_rot, world_ang, joint_rot, joint_ref)
        _, (amp_rot, _) = jp.scan(local_rot_ang, (), xs, len(joint))

        def set_qp(carry, x):
            (qp,) = carry
            (body_p, body_c), (off_p, off_c), local_rot, world_ang, world_vel = x
            local_rot = local_rot / jp.norm(local_rot)  # important
            world_rot = math.quat_mul(qp.rot[body_p], local_rot)
            world_rot = world_rot / jp.norm(world_rot)  # important
            local_pos = off_p - math.rotate(off_c, local_rot)
            world_pos = qp.pos[body_p] + math.rotate(local_pos, qp.rot[body_p])
            world_vel = qp.vel[body_p] + math.rotate(
                local_pos, math.euler_to_quat(qp.ang[body_p])
            )
            pos = jp.index_update(qp.pos, body_c, world_pos)
            rot = jp.index_update(qp.rot, body_c, world_rot)
            vel = jp.index_update(qp.vel, body_c, world_vel)
            ang = jp.index_update(qp.ang, body_c, world_ang)
            qp = qp.replace(pos=pos, rot=rot, vel=vel, ang=ang)
            return (qp,), ()

        xs = (joint_body, joint_off, amp_rot, world_ang, world_vel)
        (qp,), () = jp.scan(set_qp, (qp,), xs, len(joint))

        for children in root_idx.values():
            zs = jp.array(
                [bodies.min_z(jp.take(qp, c), CFG_SMPL.bodies[c]) for c in children]
            )
            min_z = min(jp.amin(zs), 0)
            children = jp.array(children)
            pos = jp.take(qp.pos, children) - min_z * jp.array([0.0, 0.0, 1.0])
            pos = jp.index_update(qp.pos, children, pos)
            qp = qp.replace(pos=pos)
        qp_list.append(qp)
    return qp_list


def get_rot(rot, dt, target_dt):
    rot = rot[..., [1, 2, 3, 0]]
    nframe = rot.shape[0]
    x = np.arange(nframe) * dt
    x_target = np.arange(int(nframe * dt / target_dt)) * target_dt

    rotations = Rotation.from_quat(rot)
    spline = RotationSpline(x, rotations)

    return spline(x_target, 0).as_quat()[:, [3, 0, 1, 2]]


def _compute_angular_velocity(r, time_delta: float):
    # assume the second last dimension is the time axis
    diff_quat_data = quat_identity_like(r)  # (186, 4) [[0,0,0,1],...]
    diff_quat_data[:-1, :] = quat_mul_norm(
        r[1:, :], quat_inverse(r[:-1, :])
    )  # angle difference at each time step, normalized
    diff_angle, diff_axis = quat_angle_axis(diff_quat_data)
    angular_velocity = diff_axis * diff_angle[..., None] / time_delta

    return angular_velocity


def get_ang_vel(rot, dt, target_dt, gt=None):
    rot = rot[..., [1, 2, 3, 0]]
    x = np.arange(rot.shape[0]) * dt
    x_target = np.arange(int(rot.shape[0] * dt / target_dt)) * target_dt
    rotations = Rotation.from_quat(rot)
    spline = RotationSpline(x, rotations)
    ang = _compute_angular_velocity(
        spline(x_target, 0).as_quat(), target_dt
    )  # [x,y,z,w]
    ang_smoothed = gaussian_filter1d(
        ang, sigma=2 * dt / target_dt, axis=0, mode="nearest"
    )

    if gt is not None:
        plt.plot(x, gt, "x")
        plt.plot(x_target, ang_smoothed, "-")
        plt.show()

    return ang_smoothed


def interpolate(y, dt, target_dt, gt=None):
    x = np.arange(y.shape[0]) * dt
    x_target = np.arange(int(y.shape[0] * dt / target_dt)) * target_dt
    cs = CubicSpline(x, y)
    vel = cs.derivative()(x_target)
    vel_smooth = gaussian_filter1d(vel, sigma=2 * dt / target_dt, axis=0)
    if gt is not None:
        plt.plot(x, gt)
        plt.plot(x_target, vel, "x")
        plt.plot(x_target, vel_smooth, "--")
        plt.show()
    return cs(x_target), vel_smooth


def convert_to_states(qp_list):
    demo_traj = []
    for i in range(len(qp_list)):
        qp = qp_list[i]
        demo_traj.append(
            np.concatenate(
                [
                    qp.pos.reshape(-1),
                    qp.rot.reshape(-1),
                    qp.vel.reshape(-1),
                    qp.ang.reshape(-1),
                ],
                axis=-1,
            )
        )
    demo_traj = np.stack(demo_traj, axis=0)
    return demo_traj


def amass_to_diffmimic(ase_motion):
    # THE ONLY RAW PIECES OF INFORMATION FROM SMPL-H G THAT ARE USED IN AMASS --> DIFFMIMIC...
    # THESE ARE THE VALUES THAT YOU NEED TO RECOVER!! YOU CAN PRINT ALL THESE VALUES TO CHECK THEM...
    # ase_motion['poses']               # shape = (372, 156)
    #   root_orient = ase_motion['poses'][:, 0:3]
    #   pose_body = ase_motion['poses'][:, 3:66]
    # ase_motion['trans']               # shape = (372, 3)
    # ase_motion['mocap_framerate']     # 60.0

    fps = "30"  # target FPS

    # root_orient = ase_motion["poses"][:, 0:3]  # root orientation
    # pose_body = ase_motion["poses"][:, 3:66]  # 22 joints for body w/o hands


    root_orient = ase_motion["root_orient"]
    pose_body = ase_motion["pose_body"]


    ase_poses = np.concatenate([root_orient, pose_body], -1)
    ase_poses = ase_poses.reshape([ase_poses.shape[0], -1, 3])
    # print(ase_poses.shape) is (372, 22, 3)
    print("SMLP2HUMANOID", SMPL2HUMANOID, "len=", len(SMPL2HUMANOID))
    ase_poses = ase_poses[:, SMPL2HUMANOID]  # choose Humanoid joints
    poses_gt = copy.deepcopy(ase_poses)
    # shape is (372, 18, 3)
    #print("axis angle ase_poses 1 ", ase_poses[1])
    ase_poses = axis_angle_to_matrix(torch.from_numpy(ase_poses)).float()

    ase_poses = matrix_to_euler_angles(ase_poses, "ZXY")
    ase_poses = matrix_to_quaternion(euler_angles_to_matrix(ase_poses, "XYZ")).numpy()
    # shape is (372, 18, 4)
    pelvis_trans = ase_motion["trans"]
    # shape = (372, 3)
    pelvis_trans = pelvis_trans[:, [1, 0, 2]]
    pelvis_trans[:, 0] *= -1
    dt = ase_motion['mocap_time_length'] / ase_poses.shape[0]
    # dt = 1 / ase_motion["mocap_framerate"]
    
    ase_ang = np.zeros_like(ase_poses)[..., :-1]
    ase_vel = np.zeros_like(ase_poses)[..., :-1]
    target_dt = {"orig": dt, "16": 0.0625, "30": 0.0333}[fps]
    # dt = 0.016666
    # target_dt = 0.033333
    _qp_list = convert_to_qp(ase_poses, ase_vel, ase_ang, pelvis_trans * 0.0)
    abs_poses = np.stack([qp.rot for qp in _qp_list], axis=0)
    abs_trans = np.stack([qp.pos[0] for qp in _qp_list], axis=0)
    ase_poses_interp = np.stack(
        [get_rot(ase_poses[:, i, :], dt, target_dt) for i in range(ase_poses.shape[1])],
        axis=1,
    )
    ase_ang_interp = np.stack(
        [
            get_ang_vel(abs_poses[:, i, :], dt, target_dt)
            for i in range(ase_poses.shape[1])
        ],
        axis=1,
    )

    
    trans20 = copy.deepcopy(pelvis_trans[:20])
    offset = abs_trans[0, 2] - pelvis_trans[0, 2]
    # print("offset: ", offset)
    pelvis_trans -= 0.05
    pelvis_trans += offset
    offset_trans20 = copy.deepcopy(pelvis_trans[:20])
    pelvis_trans_interp, pelvis_trans_vel_interp = interpolate(
        pelvis_trans, dt, target_dt
    )
    ase_vel_interp = np.zeros_like(ase_ang_interp)
    ase_vel_interp[:, 0] = pelvis_trans_vel_interp
    #print("original interpolated poses shape ", ase_poses_interp.shape)
    #print("original poses shape ", ase_poses.shape)
    #print("original poses[1] ", ase_poses[1])
    #print("original interpolated poses[1] ", ase_poses_interp[1])
    #print("original trans[1]", pelvis_trans_interp)
    qp_list = convert_to_qp(
        ase_poses_interp, ase_vel_interp, ase_ang_interp, pelvis_trans_interp
    )
    demo_traj = convert_to_states(qp_list)  # just a concatenation operation
    # print(demo_traj.shape)

    return demo_traj, trans20, offset_trans20, poses_gt


if __name__ == "__main__":
    # amass_raw_path = "data/amass/CMU/75/75_09_poses.npz"
    # amass_raw_path = "/oscar/data/csun45/nates_stuff/mdm-diffmimic-translation/data/amass/CMU_SMPL_XN/75_09_stageii.npz"
    # amass_raw_path = "/oscar/data/csun45/nates_stuff/mdm-diffmimic-translation/data/amass/CMU_SMPL_HG/75_09_poses.npz"

    # amass_raw_path = "/oscar/data/csun45/nates_stuff/mdm-diffmimic-translation/data/amass/CMU_SMPL_HG/75_09_poses.npz"
    # amass_raw_path = "/oscar/data/csun45/nates_stuff/mdm-diffmimic-translation/data/amass/CMU_SMPL_XN/75_09_stageii.npz"
    # amass_raw_path = "/oscar/data/csun45/nates_stuff/mdm-diffmimic-translation/75_09_stageii.npz"
    amass_raw_path = "/oscar/data/csun45/nates_stuff/mdm-diffmimic-translation/data/amass/CMU_SMPL_XN/75_09_stageii.npz"

    amass_raw = np.load(amass_raw_path, allow_pickle=True)

    print("\n\nEXAMINING INPUT:")
    for k in amass_raw.files:
        print("  ", k, amass_raw[k].shape)
    # print(amass_raw["mocap_framerate"])

    print("\n\nEXAMINING OUTPUT:")
    demo_traj, trans20, offset_trans20, poses_gt = amass_to_diffmimic(amass_raw)
    print("demo_traj.shape:", demo_traj.shape) # should be (seq_len, joint_num*13)

    action = os.path.basename(amass_raw_path).split('.')[0].split("/")[-1]
    file_name = 'data/diffmimic/{}-18-joints.npy'.format(action)
    with open(file_name, 'wb') as f:
        np.save(f, demo_traj)
    print(f"...wrote to {file_name}")
