# -*- coding: utf-8 -*-
#
# Code Developed by:
# Nima Ghorbani <https://nghorbani.github.io/>
#
# 2022.10.20
# SMPL-X Solver for MDM: Human Motion Diffusion Model

import os.path as osp
from pathlib import Path
from typing import List, Dict
from typing import Union

import numpy as np
import torch
from colour import Color
from loguru import logger
from scipy.spatial.transform import Rotation as R
from torch import nn

try:
    from human_body_prior.body_model.body_model import BodyModel
    from human_body_prior.models.ik_engine import IK_Engine
    from human_body_prior.tools.omni_tools import copy2cpu as c2c
    from human_body_prior.tools.omni_tools import create_list_chunks
except:
    from translation.utils_mdm_to_amass.human_body_prior.src.human_body_prior.body_model.body_model import \
        BodyModel
    from translation.utils_mdm_to_amass.human_body_prior.src.human_body_prior.models.ik_engine import \
        IK_Engine
    from translation.utils_mdm_to_amass.human_body_prior.src.human_body_prior.tools.omni_tools import \
        copy2cpu as c2c
    from translation.utils_mdm_to_amass.human_body_prior.src.human_body_prior.tools.omni_tools import \
        create_list_chunks
import argparse
from tqdm import tqdm


class SourceKeyPoints(nn.Module):
    def __init__(self,
                 bm: Union[str, BodyModel],
                 n_joints: int = 22,
                 kpts_colors: Union[np.ndarray, None] = None,
                 num_betas=16
                 ):
        super(SourceKeyPoints, self).__init__()

        self.bm = BodyModel(bm, num_betas=num_betas, persistant_buffer=False) if isinstance(bm, str) else bm
        self.bm_f = []  # self.bm.f
        self.n_joints = n_joints
        self.kpts_colors = np.array(
            [Color('grey').rgb for _ in range(n_joints)]) if kpts_colors == None else kpts_colors

    def forward(self, body_parms):
        new_body = self.bm(**body_parms)

        return {'source_kpts': new_body.Jtr[:, :self.n_joints], 'body': new_body}


def transform_smpl_coordinate(bm_fname: Path, trans: np.ndarray,
                              root_orient: np.ndarray, betas: np.ndarray,
                              rotxyz: Union[np.ndarray, List]) -> Dict:
    """
    rotates smpl parameters while taking into account non-zero center of rotation for smpl
    Parameters
    ----------
    bm_fname: body model filename
    trans: Nx3
    root_orient: Nx3
    betas: num_betas
    rotxyz: desired XYZ rotation in degrees

    Returns
    -------

    """
    if isinstance(rotxyz, list):
        rotxyz = np.array(rotxyz).reshape(1, 3)
    if betas.ndim == 1: betas = betas[None]
    if betas.ndim == 2 and betas.shape[0] != 1:
        logger.warning(
            f'betas should be the same for the entire sequence. 2D np.array with 1 x num_betas: {betas.shape}. taking the mean')
        betas = np.mean(betas, keepdims=True, axis=0)
    transformation_euler = np.deg2rad(rotxyz)

    coord_change_matrot = R.from_euler('XYZ', transformation_euler.reshape(1, 3)).as_matrix().reshape(3, 3)
    bm = BodyModel(bm_fname=bm_fname,
                   num_betas=betas.shape[1])
    pelvis_offset = c2c(bm(**{'betas': torch.from_numpy(betas).type(torch.float32)}).Jtr[[0], 0])

    root_matrot = R.from_rotvec(root_orient).as_matrix().reshape([-1, 3, 3])

    transformed_root_orient_matrot = np.matmul(coord_change_matrot, root_matrot.T).T
    transformed_root_orient = R.from_matrix(transformed_root_orient_matrot).as_rotvec()
    transformed_trans = np.matmul(coord_change_matrot, (trans + pelvis_offset).T).T - pelvis_offset

    return {'root_orient': transformed_root_orient.astype(np.float32),
            'trans': transformed_trans.astype(np.float32), }


def pose_seq_3D_to_axis_angle(mdm_data, surface_model_type='smplx', gender='neutral', verbosity=0):
    '''
    :param skeleton_movie_fname:
    :param surface_model_type:
    :param gender:
    :param verbosity: 0: silent, 1: text, 2: visual with psbody.mesh
    :return:
    '''
    support_dir = "body_models"
    # 'TRAINED_MODEL_DIRECTORY'  in this directory the trained model along with the model code exist
    vposer_expr_dir = osp.join(support_dir, 'vposer_v2_05')

    # 'PATH_TO_SMPLX_model.npz'  obtain from https://smpl-x.is.tue.mpg.de/downloads
    bm_fname = osp.join(support_dir, f'{surface_model_type}/{gender}/model.npz')
    motion = mdm_data['motion'][0].transpose(2, 0, 1)  # [nframes, njoints, 3]

    comp_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n_joints = 22
    num_betas = 16

    red = Color("red")
    blue = Color("blue")
    kpts_colors = [c.rgb for c in list(red.range_to(blue, n_joints))]

    # create source and target key points and make sure they are index aligned
    data_loss = torch.nn.MSELoss(reduction='sum')

    stepwise_weights = [
        {'data': 10., 'poZ_body': .01, 'betas': .5},
    ]

    optimizer_args = {'type': 'LBFGS', 'max_iter': 300, 'lr': 1, 'tolerance_change': 1e-4,
                      'history_size': 200}
    ik_engine = IK_Engine(vposer_expr_dir=vposer_expr_dir,
                          verbosity=verbosity,
                          display_rc=(2, 2),
                          data_loss=data_loss,
                          num_betas=num_betas,
                          stepwise_weights=stepwise_weights,
                          optimizer_args=optimizer_args).to(comp_device)

    batch_size = len(motion)
    all_results = {}
    batched_frames = create_list_chunks(np.arange(len(motion)), batch_size, overlap_size=0,
                                        cut_smaller_batches=False)
    if verbosity < 2:
        batched_frames = tqdm(batched_frames, desc='VPoser Advanced IK')
    for cur_frame_ids in batched_frames:

        target_pts = torch.from_numpy(motion[cur_frame_ids, :n_joints]).to(comp_device)
        source_pts = SourceKeyPoints(bm=bm_fname, n_joints=n_joints, kpts_colors=kpts_colors,
                                     num_betas=num_betas).to(
            comp_device)

        ik_res = ik_engine(source_pts, target_pts, {})

        ik_res_detached = {k: c2c(v) for k, v in ik_res.items()}
        # nan_mask = np.isnan(ik_res_detached['trans']).sum(-1) != 0
        # if nan_mask.sum() != 0: raise ValueError('Sum results were NaN!')
        for k, v in ik_res_detached.items():
            if k not in all_results: all_results[k] = []
            all_results[k].append(v)

    d = {k: np.concatenate(v, axis=0) for k, v in all_results.items()}
    # d['betas'] = np.median(d['betas'], axis=0)
    d['betas'] = np.zeros(num_betas)

    transformed_d = transform_smpl_coordinate(bm_fname=bm_fname, trans=d['trans'],
                                              root_orient=d['root_orient'],
                                              betas=d['betas'], rotxyz=[90, 0, 0])
    d.update(transformed_d)
    d['poses'] = np.concatenate([d['root_orient'], d['pose_body'], np.zeros([len(d['pose_body']), 99])],
                                axis=1)

    d['surface_model_type'] = surface_model_type
    d['gender'] = gender
    d['mocap_framerate'] = 30
    d['num_betas'] = num_betas

    del data_loss
    del ik_engine
    del batched_frames
    del ik_res
    del ik_res_detached
    torch.cuda.empty_cache()

    return d


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help='skeleton movie filename that is to be converted into SMPL')
    parser.add_argument("--model_type", type=str, default='smplx', help='model_type; e.g. smplx')
    parser.add_argument("--gender", type=str, default='neutral', help='gender; e.g. neutral')
    parser.add_argument("--verbosity", type=int, default=0, help='gender; e.g. neutral')
    params = parser.parse_args()

    convert_mdm_mp4_to_amass_npz(skeleton_movie_fname=params.input,
                                 surface_model_type=params.model_type,
                                 gender=params.gender,
                                 verbosity=params.verbosity)
