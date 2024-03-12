import sys, os
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
import time

from utils_mdm.body_model import BodyModel
from utils_mdm.skeleton import Skeleton
from utils_mdm.quaternion import qbetween_np, qrot, qrot_np, qmul_np, qinv, \
    qinv_np, qfix, quaternion_to_cont6d, quaternion_to_cont6d_np
from utils_mdm.paramUtil import t2m_raw_offsets, t2m_kinematic_chain


# ========== Configuration and Environment Setup ==========

# os.environ['PYOPENGL_PLATFORM'] = 'egl'
male_bm_path = './utils_mdm/body_models/smplh/male/model.npz'
male_dmpl_path = './utils_mdm/body_models/dmpls/male/model.npz'
female_bm_path = './utils_mdm/body_models/smplh/female/model.npz'
female_dmpl_path = './utils_mdm/body_models/dmpls/female/model.npz'
n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
kinematic_chain = t2m_kinematic_chain

# Choose the device to run the body model on.
comp_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ========== Main Processing Steps ==========

def raw_pose_processing(amass_raw):

    print("TO DO IN RAW_POSE_PROCESSING: scatter plot before and after... figure out if it's 3D positions, or poses!! Create a general purpose function that inputs a numpy array and a save dir")

    num_betas = 10 # number of body parameters
    num_dmpls = 8 # number of DMPL parameters
    male_bm = BodyModel(bm_fname=male_bm_path, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=male_dmpl_path).to(comp_device)
    female_bm = BodyModel(bm_fname=female_bm_path, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=female_dmpl_path).to(comp_device)

    trans_matrix = np.array([[1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0],
                                [0.0, 1.0, 0.0]])
    ex_fps = 20

    bdata = amass_raw
    
    fps = bdata['mocap_framerate']
    frame_number = bdata['trans'].shape[0]
    print("fps, frame_number:", fps, frame_number)

    fId = 0 # frame id of the mocap sequence
    pose_seq = []
    if bdata['gender'] == 'male':
        bm = male_bm
    else:
        bm = female_bm
    
    down_sample = int(fps / ex_fps)
    print(fps, ex_fps, down_sample)
    
    with torch.no_grad():
        for fId in range(0, frame_number, down_sample):

            root_orient = torch.Tensor(bdata['poses'][fId:fId+1, :3]).to(comp_device)   # (1, 3); controls the global root orientation
            pose_body = torch.Tensor(bdata['poses'][fId:fId+1, 3:66]).to(comp_device)   # (1, 63); controls the body
            pose_hand = torch.Tensor(bdata['poses'][fId:fId+1, 66:]).to(comp_device)    # (1, 90); controls the finger articulation
            betas = torch.Tensor(bdata['betas'][:10][np.newaxis]).to(comp_device)       # (1, 10); controls the body shape
            trans = torch.Tensor(bdata['trans'][fId:fId+1]).to(comp_device)             # (1, 3); controls translation of root position

            body = bm(pose_body=pose_body, pose_hand=pose_hand, betas=betas, root_orient=root_orient)
            joint_loc = body.Jtr[0] + trans # (52, 3); 52 = 21 + 30 + 1

            pose_seq.append(joint_loc.unsqueeze(0))
            
    pose_seq = torch.cat(pose_seq, dim=0) # (124, 52, 3)
    pose_seq_np = pose_seq.detach().cpu().numpy() # (124, 52, 3)
    pose_seq_np_n = np.dot(pose_seq_np, trans_matrix) # (124, 52, 3)
        
    pose_seq_np_n[..., 0] *= -1

    return pose_seq_np_n # (124, 52, 3)


def uniform_skeleton(
        positions, 
        target_offset,
        n_raw_offsets,
        kinematic_chain,
        l_idx1,
        l_idx2,
        face_joint_indx
    ):
    src_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
    src_offset = src_offset.numpy()
    tgt_offset = target_offset.numpy()
    # print(src_offset)
    # print(tgt_offset)
    '''Calculate Scale Ratio as the ratio of legs'''
    src_leg_len = np.abs(src_offset[l_idx1]).max() + np.abs(src_offset[l_idx2]).max()
    tgt_leg_len = np.abs(tgt_offset[l_idx1]).max() + np.abs(tgt_offset[l_idx2]).max()

    scale_rt = tgt_leg_len / src_leg_len
    # print(scale_rt)
    src_root_pos = positions[:, 0]
    tgt_root_pos = src_root_pos * scale_rt

    '''Inverse Kinematics'''
    quat_params = src_skel.inverse_kinematics_np(positions, face_joint_indx)
    # print(quat_params.shape)

    '''Forward Kinematics'''
    src_skel.set_offset(target_offset)
    new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)
    return new_joints

def process_file(
        positions, 
        feet_thre, 
        tgt_offsets, 
        face_joint_indx, 
        fid_l, 
        fid_r,
        n_raw_offsets,
        kinematic_chain,
        l_idx1,
        l_idx2
    ):
    """
    INPUTS:
        positions,          (seq_len, 22, 3) = (123, 263)
        feet_thre,          0.002
        tgt_offsets, 
        face_joint_indx, 
        fid_l, 
        fid_r,
        n_raw_offsets,
        kinematic_chain,
        l_idx1,
        l_idx2
    OUTPUT:
        data:               (seq_len-1, 263) = (123, 263)
        global_positions:
        positions:
        l_velocity:
    """

    '''Uniform Skeleton'''
    positions = uniform_skeleton(
        positions, 
        tgt_offsets,
        n_raw_offsets,
        kinematic_chain,
        l_idx1,
        l_idx2,
        face_joint_indx
    ) # (124, 22, 3)

    '''Put on Floor'''
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height

    '''XZ at origin; i.e., move the first pose to the origin'''
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    '''All initially face Z+'''
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]
    # rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1) # (3,)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis] # (3,)
    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init # (124, 22, 4); each (i, j, :) is equal!!
    # positions_b = positions.copy()
    positions = qrot_np(root_quat_init, positions) # (123, 22, 3), before and after
    # plot_3d_motion("./positions_2.mp4", kinematic_chain, positions, 'title', fps=20)

    '''New ground truth positions'''
    global_positions = positions.copy()

    """ Get Foot Contacts """
    thres = feet_thre
    velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])
    feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
    feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
    feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
    feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

    feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
    feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
    feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
    feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float32)


    '''Quaternion and Cartesian representation'''
    r_rot = None
    # Get continuous 6d parameters
    def get_cont6d_params(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)

        '''Quaternion to continuous 6D'''
        cont_6d_params = quaternion_to_cont6d_np(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        # (seq_len, joints_num, 4)
        return cont_6d_params, r_velocity, velocity, r_rot
    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
    # Get Rifke
    def get_rifke(positions):
        '''Local pose'''
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        '''All pose face Z+'''
        positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
        return positions
    positions = get_rifke(positions) # (124, 22, 3)

    '''Root height'''
    root_y = positions[:, 0, 1:2]               # (seq_len, 1) = (124, 1)
    '''Root rotation and linear velocity'''
    r_velocity = np.arcsin(r_velocity[:, 2:3])  # (seq_len-1, 1) = (123, 1); rotation velocity along y-axis
    l_velocity = velocity[:, [0, 2]]            # (seq_len-1, 2) = (123, 2); linear velocity on xz plane
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1) # (seq_len-1, 4)

    '''Get Joint Rotation Representation; quaternion for skeleton joints'''
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1) # (seq_len, (joints_num-1) *6) = (123, 126)

    '''Get Joint Rotation Invariant Position Represention; local joint position'''
    ric_data = positions[:, 1:].reshape(len(positions), -1) # (seq_len, (joints_num-1)*3) = (124, 63)


    '''Get Joint Velocity Representation'''
    local_vel = qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
                        global_positions[1:] - global_positions[:-1]) # (seq_len-1, joints_num, 3) = (123, 22, 3)
    local_vel = local_vel.reshape(len(local_vel), -1) # (seq_len-1, joints_num*3) = (123, 66)


    '''Concatenate everything!! Get Joint Velocity Representation'''
    # root_data.shape = (116, 4); CAN USE THIS TO RECOVER TRANS VECTOR...
    #       root angular velocity along Y-axis,
    #       root linear velocity on X-plane
    #       root linear velocity on Z-plane,
    #       root height
    data = root_data
    # ric_data[:-1].shape = (116, 63)
    #       Joint Rotation Invariant Position Represention
    #       (seq_len, (joints_num-1) * 3) 
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    # rot_data[:-1].shape = (116, 126)
    #       (seq_len, (joints_num-1) * 6) quaternion for skeleton joints
    #       it's the 6D continuous rotation representation that's widely used
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    # local_vel.shape = (116, 66)
    #       (seq_len-1, joints_num*3)
    #       joint velocity representation
    data = np.concatenate([data, local_vel], axis=-1)
    # feet_l.shape = (116, 2)   
    # feet_r.shape = (116, 2);  # joint velocities to emphasize the foot ground contacts.
    #       (seq_len, 2)
    #       (seq_len, 2)
    #       binary features obtained by thresholding the heel and toe 
    data = np.concatenate([data, feet_l, feet_r], axis=-1)
    # data.shape = (116, 263)
    #       263 = 4 + 63 + 126 + 66 + 2 + 2
    # print(data.shape)

    # see pp 23 here for labeled diagram: https://files.is.tue.mpg.de/black/talks/SMPL-made-simple-FAQs.pdf
    # this is what order the indices correspond to when we're doing the visualization... when it's minus one,
    # that probably corresponds to ignoring the pelvis, since that's the root
    # kinematic_chain = [
    #     [0, 2, 5, 8, 11],       # pelvis, right hip, right knee, right ankle, right foot
    #     [0, 1, 4, 7, 10],       # pelvis, left hip, left knee, left ankle, left foot
    #     [0, 3, 6, 9, 12, 15],   # pelvis, spine1, spine2, spine3, neck, head
    #     [9, 14, 17, 19, 21],    # spine3, right collar, right shoulder, right elbow, right wrist
    #     [9, 13, 16, 18, 20]     # spine3, left collar, left shoulder, left elbow, left wrist
    # ] 

    return data, global_positions, positions, l_velocity


# Recover global angle and positions for rotation data
# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_rot(data, joints_num, skeleton):
    r_rot_quat, r_pos = recover_root_rot_pos(data)

    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)

    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]
    #     print(r_rot_cont6d.shape, cont6d_params.shape, r_pos.shape)
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, joints_num, 6)

    positions = skeleton.forward_kinematics_cont6d(cont6d_params, r_pos)

    return positions


def recover_from_ric(data, joints_num):
    """
    data.shape = (1, 123, 263)
    joints_num = 22
    """

    r_rot_quat, r_pos = recover_root_rot_pos(data) # (1, 123, 4), (1, 123, 3)
    positions = data[..., 4:(joints_num - 1) * 3 + 4] # (1, 123, 63)
    positions = positions.view(positions.shape[:-1] + (-1, 3)) # (1, 123, 21, 3)

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions) # (1, 123, 21, 3)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1] # (1, 123, 21, 3)
    positions[..., 2] += r_pos[..., 2:3] # (1, 123, 21, 3)

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2) # (1, 123, 21, 3)

    return positions


def motion_representation(pose_seq):
    """
    INPUT:
        pose_seq: (124, 52, 3); I THINK this is basically just a re-indexed view of the AMASS data (FALSE!! CUZ THERES SMLP/DMPL NON-INVERTIBLE STUFF THAT HAPPENED...)
    OUTPUT:
        data: (123, 263)
        rec_ric_data: (123, 23, 3)
    """

    l_idx1, l_idx2 = 5, 8 # Lower legs
    fid_r, fid_l = [8, 11], [7, 10] # Right/Left foot
    face_joint_indx = [2, 1, 17, 16] # Face direction, r_hip, l_hip, sdr_r, sdr_l
    r_hip, l_hip = 2, 1 # l_hip, r_hip
    joints_num = 22

    # Get offsets of target skeleton
    example_id = "000021"
    example_data = np.load(os.path.join("utils_mdm", example_id + '.npy')) # (180, 52, 3)
    example_data = torch.from_numpy(example_data)
    tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    tgt_offsets = tgt_skel.get_offsets_joints(example_data[0]) # (22, 3)

    # NOTE: the same tgt_skel for that single example is used across all examples.
    source_data = pose_seq[:, :joints_num] # (124, 22, 3)

    data, ground_positions, positions, l_velocity = process_file(
        source_data, 
        0.002,
        tgt_offsets, 
        face_joint_indx, 
        fid_l, 
        fid_r,
        n_raw_offsets,
        kinematic_chain,
        l_idx1,
        l_idx2
    )
    # data.shape = (123, 263)
    rec_ric_data = recover_from_ric(torch.from_numpy(data).unsqueeze(0).float(), joints_num)[0] # (123, 22, 3)

    return data, rec_ric_data


def amass_to_mdm(amass_raw):

    # THE ONLY RAW PIECES OF INFORMATION FROM SMPL-H G THAT ARE USED IN AMASS --> MDM...

    # amass_raw['poses'][:, 0:3]      # (372, 3),   controls the global root orientation
    # amass_raw['poses'][:, 3:66]     # (372, 63),  controls the body
    # amass_raw['poses'][:, 66:]      # (372, 90),  controls the finger articulation; note we don't need to recover these values
    # amass_raw['betas'][0:10]        # (10,),      controls the body shape; note this wasn't used in DiffMimic, so we don't need to recover these values
    # amass_raw['trans'][:]           # (372, 3),   controls translation

    pose_seq = raw_pose_processing(amass_raw)
    pose_seq, pose_seq_3D = motion_representation(pose_seq)

    return pose_seq, pose_seq_3D




if __name__ == "__main__":


    amass_raw_path = "data/amass/CMU/75/75_09_poses.npz"
    amass_raw = np.load(amass_raw_path)

    pose_seq, pose_seq_3D = amass_to_mdm(amass_raw)
    print("Daksh: pose_seq is 263-dimensional, whereas pose_seq_3D is (22,3)-dimensiona and represents 3D cartesian coordinates. The former is what is input and output from the MDM model, so THATS what we need to work with. But if you look carefully, with a single function call, you can obtain the latter (22, 3) dimensional representation with a single function call; see recover_from_ric. That may or may not be useful. It's possible that the right way to go is actually just doing inverse kinematics from pose_seq_3D; see https://github.com/nghorbani/human_body_prior/blob/master/tutorials/ik_example_joints.py for how inverse kinematics works. It looks like they use the same body model that we do; I predict that this will be the more efficient way to go about doing this, especially because the linear blend skinning step isn't actually invertible. You should google `inverse kinematics' to get a high level understanding though to understand the pros and cons of this approach... this project is a lot more open ended (and AI theoretical) than the diffmimic one.")
    print("pose_seq.shape:", pose_seq.shape)
    print("pose_seq_3D.shape:", pose_seq_3D.shape)

    action = os.path.basename(amass_raw_path).split('.')[0].split("/")[-1]
    save_path = 'data/mdm/{}.npy'.format(action)
    with open(save_path, 'wb') as f:
        np.save(f, pose_seq)
    print(f"Wrote to {save_path}")
