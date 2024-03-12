import sys, os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
import time
from pathlib import Path
from amass_to_mdm import amass_to_mdm
from amass_to_mdm_mod import amass_to_mdm as amass_to_mdm_test
from mdm_to_amass import mdm_to_amass

from utils_mdm_to_amass.rigid_transform_3D import rigid_transform_3D
from numpy import linalg as LA

def parse_args():

    parser = argparse.ArgumentParser(prog='MDM --> AMASS test')

    parser.add_argument(
        "--model-type",
        default='smplh',
        choices=['smplh','smplx'],
        type=str, 
        help="The desired output model type"
    )
    parser.add_argument(
        "--gender",
        default='neutral',
        type=str,
        choices=['neutral','female','male'],
        help="Gender"
    )
    
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_args()

    amass_raw_path = './translation/data/amass/CMU_SMPL_HG/75_09_poses.npz'
    amass_raw = np.load(amass_raw_path)
    (
        pose_seq,   # (123, 263); 263-dim is input and output from the MDM model
        pose_seq_3d # (123, 22, 3); (22, 3) represents 3D cartesian coordinates
    ) = amass_to_mdm(amass_raw)
    
    dirs = os.path.normpath(amass_raw_path).split(os.sep)
    subdir = dirs[4].split('.')[0]
    write_path = os.path.join("translation/data/mdm/", dirs[3], subdir) + "/"
    print("path ", write_path)
    if not os.path.exists(write_path):
        os.makedirs(write_path)
    
    amass_rec, poses, trans = mdm_to_amass(pose_seq_3d, write_path, 20, amass_raw['mocap_framerate'], model_type=args.model_type, gender=args.gender)


    # amass_raw_path = example
    # amass_raw = np.load(amass_raw_path)
    trans = amass_raw['trans']
    _, _, gr = amass_to_mdm_test(amass_raw)
    fps = int(amass_raw['mocap_framerate'])
    tr_down = np.array([trans[i] for i in range(0,gr.shape[0],fps//20)])[1:]
    gr = np.array([gr[i] for i in range(0,gr.shape[0],fps//20)])[1:]
    
    amass_raw_path = write_path+'sample00_rep00.npz'
    amass_raw = np.load(amass_raw_path)
    tr_rec_down = amass_raw['trans']
    _, _, gr_rec = amass_to_mdm_test(amass_raw)
    
    R,t = rigid_transform_3D(gr[0].T, gr_rec[0].T)

    A = np.stack([gr[:,:,i] for i in range(3)])
    A = np.stack([A[0].flatten(),A[1].flatten(),A[2].flatten()])

    B = np.stack([gr_rec[:,:,i] for i in range(3)])
    B = np.stack([B[0].flatten(),B[1].flatten(),B[2].flatten()])

    C = R@A+t-B

    print('Testing on: ', amass_raw_path)
    print(R.shape, A.shape, t.shape, B.shape)
    print("L1 norm for the downsampled poses values: ", LA.norm(C,1))
    print("L2 norm for the downsampled poses values: ", LA.norm(C))

    S,v = rigid_transform_3D(tr_down[0:20].T, tr_rec_down[0:20].T)

    L = S@(tr_down.T)+v-tr_rec_down.T

    print("L1 norm for the downsampled trans values: ", LA.norm(L,1))
    print("L2 norm for the downsampled trans values: ", LA.norm(L))
    
    amass_raw_path = write_path+'upsampled.npz'
    amass_raw = np.load(amass_raw_path)
    print('upsampled frames ', amass_raw['trans'].shape)
    trans = trans[0:-10]
    trans_rec = amass_raw['trans'][0:trans.shape[0]]
    
    sample = trans.shape[0]//4
    S,v = rigid_transform_3D(trans[0:sample].T, trans_rec[0:sample].T)
    
    L = S@(trans.T)+v-trans_rec.T
    
    print("L1 norm for the upsampled trans values: ", LA.norm(L,1))
    print("L2 norm for the upsampled trans values: ", LA.norm(L))
