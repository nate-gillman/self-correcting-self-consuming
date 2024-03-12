"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.

This code is based on https://github.com/openai/guided-diffusion
"""

import os
import numpy as np
from utils_mdm_to_amass.plot_script import plot_3d_motion
import scipy
from amass_to_mdm import amass_to_mdm
from amass_to_mdm_mod import amass_to_mdm as amass_to_mdm_test
from mdm_to_amass import save_as_mdm

def save_point_cloud(motion, i, azim, num): # 0, -30, 0-23

    data = motion[i]
    x = data[:num, 0]
    y = data[:num, 1]
    z = data[:num, 2]

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import proj3d

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.view_init(elev=0, azim=azim)

    # Set the axis limits here
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 1.5])

    ax.scatter(x, y, z)
    plt.savefig("plot%02d.png" % i)


def visualize_samples_and_save_to_disk(
    samples,
    out_path: str,
    file_name: str,
    fps: float,
    caption: str = 'test') -> str:

    # --- visualizations ---
    print(f"saving visualizations to [{out_path}]...")
      
    t2m_kinematic_chain = [
        [0, 2, 5, 8, 11],
        [0, 1, 4, 7, 10],
        [0, 3, 6, 9, 12, 15],
        [9, 14, 17, 19, 21],
        [9, 13, 16, 18, 20]
        ]

    animation_save_dir = out_path
    animation_save_path = os.path.join(animation_save_dir, file_name)

    os.makedirs(animation_save_dir, exist_ok=True)
    # the below actually saves to disk
    plot_3d_motion(
        animation_save_path,
        t2m_kinematic_chain,
        samples,# samples.transpose(0, 2, 1)[:,:,:22], # (61, 3, 24)
        dataset="humanml",
        title=caption,
        fps=fps,
    )

    abs_path = os.path.abspath(animation_save_path)
    print(f"[Done] Results are at [{abs_path}]")

    return abs_path # return sample enclosing folder




def main() -> None:

    fps = 20

    out_path = "./translation/data/mdm/visualize/"

    example = "./translation/data/amass/CMU_SMPL_HG/75_09_poses.npz"

    dirs = os.path.normpath(example).split(os.sep)
    prefix = dirs[-1].split('.')[0]
    
    arr_raw = np.load(example, allow_pickle=True)
    _, pose_seq = amass_to_mdm(arr_raw)
    mdm_path = "./translation/data/mdm/CMU_SMPL_HG/" + prefix + "/"
    save_as_mdm(pose_seq,  mdm_path)
   # mdm_path = "./translation/data/mdm/CMU_SMPL_HG/75_09_poses/results.npy"

    mdm = np.load(mdm_path+"results.npy", allow_pickle=True)[None][0] 
    gt = mdm["motion"][0]   
    gt = np.stack([gt[:,:,i] for i in range(gt.shape[-1])])
    print("", gt.shape)
    file_name = prefix + ".mp4"
    visualize_samples_and_save_to_disk(
        gt,
        out_path,
        file_name,
        fps,
        'Original ' + prefix
        )

    recovered_example = mdm_path + "sample00_rep00.npz"
    arr_raw = np.load(recovered_example, allow_pickle=True)
    _, pose_seq, _ = amass_to_mdm_test(arr_raw)
    mdm_rec_path = mdm_path[:-1] + "_rec/"
    save_as_mdm(pose_seq, mdm_rec_path)

    mdm_rec = np.load(mdm_rec_path+"results.npy", allow_pickle=True)[None][0] 
    gt = mdm_rec["motion"][0]   
    gt = np.stack([gt[:,:,i] for i in range(gt.shape[-1])])
    print("gt ", gt.shape)
    file_name = prefix+"_recovered" + ".mp4"
    visualize_samples_and_save_to_disk(
        gt,
        out_path,
        file_name,
        fps,
        'Recovered ' + prefix
        )



        # --- VISUALIZE SAMPLES AS VIDEOS AND SAVE TO DISK ---




if __name__ == "__main__":
    main()
