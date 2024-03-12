"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.

This code is based on https://github.com/openai/guided-diffusion
"""

import os
import numpy as np
import scipy

from textwrap import wrap
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_3d_motion(
    save_path,
    kinematic_tree,
    joints,
    title,
    dataset,
    figsize=(3, 3),
    fps=120,
    radius=3,
    vis_mode="default",
    gt_frames=[],
):
    matplotlib.use("Agg")

    title = "\n".join(wrap(title, 20))

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3.0, radius * 2 / 3.0])
        # print(title)
        fig.suptitle(title, fontsize=10)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz],
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    #         return ax

    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)

    data[:,:,0], data[:,:,1], data[:,:,2] = data[:,:,1], data[:,:,2], data[:,:,0] # WORKS
    # data[:,:,0], data[:,:,1], data[:,:,2] = data[:,:,2], data[:,:,0], data[:,:,1]

    # preparation related to specific datasets
    if dataset == "kit":
        data *= 0.003  # scale for visualization
    elif dataset == "humanml":
        data *= 1.3  # scale for visualization
    elif dataset in ["humanact12", "uestc"]:
        data *= -1.5  # reverse axes, scale for visualization

    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = p3.Axes3D(fig)
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]  # GT color
    colors_orange = [
        "#DD5A37",
        "#D69E00",
        "#B75A39",
        "#FF6D00",
        "#DDB50E",
    ]  # Generation color
    colors = colors_orange
    if vis_mode == "upper_body":  # lower body taken fixed to input motion
        colors[0] = colors_blue[0]
        colors[1] = colors_blue[1]
    elif vis_mode == "gt":
        colors = colors_blue

    frame_number = data.shape[0]

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    def update(index):
        ax.lines = []
        ax.collections = []
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        plot_xzPlane(
            MINS[0] - trajec[index, 0],
            MAXS[0] - trajec[index, 0],
            0,
            MINS[2] - trajec[index, 1],
            MAXS[2] - trajec[index, 1],
        )
        #         ax.scatter(dataset[index, :22, 0], dataset[index, :22, 1], dataset[index, :22, 2], color='black', s=3)

        # if index > 1:
        #     ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
        #               trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
        #               color='blue')
        # #             ax = plot_xzPlane(ax, MINS[0], MAXS[0], 0, MINS[2], MAXS[2])

        used_colors = colors_blue if index in gt_frames else colors
        for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(
                data[index, chain, 0],
                data[index, chain, 1],
                data[index, chain, 2],
                linewidth=linewidth,
                color=color,
            )

        plt.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(
        fig, update, frames=frame_number, interval=1000 / fps, repeat=False
    )

    # PALM lab note:
    # if you encounter: ValueError: unknown file extension: .mp4, load the oscar module for ffpeg with:
    # module load ffmpeg
    ani.save(save_path, fps=fps)
    plt.close()

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
    file_name : str,
    fps: float,
    kinematic_chain
) -> str:

    # --- visualizations ---
    print(f"saving visualizations to [{out_path}]...")

    t2m_kinematic_chain = kinematic_chain

    caption = "\n".join(out_path.split("/")[-2:])
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

    # uhc_output = np.load("/oscar/data/csun45/nates_stuff/UniversalHumanoidControl/pose_aa--0-ACCAD_MartialArtsWalksTurns_c3d_E9 - side step left_poses.npy", allow_pickle=True)
    fps = 20

    #######################################################################################
    ####### EXAMPLE 1: "0-ACCAD_MartialArtsWalksTurns_c3d_E9 - side step left_poses" ######
    #######################################################################################

    motion_names = [
        "0-ACCAD_Male2General_c3d_A2- Sway_poses",
        "0-ACCAD_Male2Running_c3d_C8 - run backwards to stand_poses",
        "0-ACCAD_MartialArtsWalksTurns_c3d_E9 - side step left_poses",
        "0-BioMotionLab_NTroje_rub008_0025_kicking1_poses",
        "0-BMLhandball_S07_Expert_Trial_upper_left_206_poses",
        "0-BMLmovi_Subject_85_F_MoSh_Subject_85_F_4_poses",
        "0-KIT_10_RightTurn01_poses",
        "0-KIT_314_bend_left06_poses",
        "0-KIT_317_turn_right09_poses",
        "0-Transitions_mocap_mazen_c3d_jumpingjacks_jumpinplace_poses"
    ]

    model_names = ["uhc_explicit", "uhc_implicit", "uhc_implicit_shape"]

    for motion_name in motion_names:
        for model_name in model_names:

            # OUTPUT FROM EXPLICIT MODEL...
            out_path_base = f"/oscar/data/csun45/nates_stuff/motion-diffusion-model/uhc-testing/{motion_name}"
            out_path = os.path.join(out_path_base, model_name)
            uhc_output = np.load(f"/oscar/data/csun45/nates_stuff/motion-diffusion-model/uhc-testing/{motion_name}/{model_name}/motion.npy", allow_pickle=True)[None][0]

            # Ground Truth
            gt = uhc_output["gt_jpos"]      # (69, 72)
            gt = gt.reshape(-1, 24, 3)      # (69, 24, 3)
            file_name = "gt.mp4"
            visualize_samples_and_save_to_disk(gt, out_path, file_name, fps)

            # Pred
            pred = uhc_output["pred_jpos"]  # (69, 72)
            pred = pred.reshape(-1, 24, 3)  # (69, 24, 3)
            file_name = "pred.mp4"
            visualize_samples_and_save_to_disk(pred, out_path, file_name, fps)






if __name__ == "__main__":
    main()