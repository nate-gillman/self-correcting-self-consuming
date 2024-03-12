"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.

This code is based on https://github.com/openai/guided-diffusion
"""

import os
from textwrap import wrap
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np

# from data_loaders.humanml.utils.plot_script import plot_3d_motion
import scipy
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
    gt_frames=[],
):
    matplotlib.use("Agg")

    ELEV = 91



    title = "\n".join(wrap(title, 20))

    def init():

        zoom_factor = 0.5

        ax.set_xlim3d([-radius * zoom_factor, radius * zoom_factor])
        ax.set_ylim3d([- (1/2) * radius * zoom_factor, (3/2) * radius * zoom_factor])
        ax.set_zlim3d([-radius * zoom_factor, radius * zoom_factor])
        fig.suptitle(title, fontsize=10)
        ax.grid(b=False)
        ax.view_init(elev=ELEV, azim=-90) # Set a fixed view
        ax.dist = 7.5 * zoom_factor # Set a fixed camera distance

        # Draw clear XYZ axes in the initial frame
        ax.quiver(0, 0, 0, radius/2, 0, 0, alpha=0.3, arrow_length_ratio=0.1, color="r")  # X-axis
        ax.quiver(0, 0, 0, 0, radius/2, 0, alpha=0.3, arrow_length_ratio=0.1, color="g")  # Y-axis
        ax.quiver(0, 0, 0, 0, 0, radius, alpha=0.3, arrow_length_ratio=0.1, color="b")  # Z-axis

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

    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)

    # if [0,1,2,3,4] in kinematic_tree:
    #     # i.e. if it's UHC... need to change viewpoint
    #     data[:,:,0], data[:,:,1], data[:,:,2] = data[:,:,1], data[:,:,2], data[:,:,0] # WORKS; sideways uhc view
    #     # data[:,:,[0,2]] = data[:,:,[2,0]]
    #     pass

    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = p3.Axes3D(fig)
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors_blue =   ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]  # GT color
    colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]  # Generation color
    colors = colors_orange

    frame_number = data.shape[0]

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    def update(index):
        ax.lines = []
        ax.collections = []
        ax.view_init(elev=ELEV, azim=-90)
        ax.dist = 7.5
        plot_xzPlane(MINS[0], MAXS[0], 0, MINS[2], MAXS[2])
        ax.scatter(data[index, :24, 0], data[index, :24, 1], data[index, :24, 2], color='black', s=3)
        if index > 1:
            ax.plot3D(trajec[:index, 0], np.zeros_like(trajec[:index, 0]),
                    trajec[:index, 1], linewidth=1.0, color='blue')

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

        ax.quiver(0, 0, 0, radius/2, 0, 0, alpha=0.3, arrow_length_ratio=0.1, color="r")  # X-axis
        ax.quiver(0, 0, 0, 0, radius * (2/3), 0, alpha=0.3, arrow_length_ratio=0.1, color="g")  # Y-axis
        ax.quiver(0, 0, 0, 0, 0, radius, alpha=0.3, arrow_length_ratio=0.1, color="b")  # Z-axis

    ani = FuncAnimation(
        fig, update, frames=frame_number, interval=1000 / fps, repeat=False
    )

    # PALM lab note: if you encounter: ValueError: unknown file extension: .mp4,
    # load the oscar module for ffpeg with "module load ffmpeg"
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
    caption: str,
    kinematic_chain: List[List]
) -> str:

    # --- visualizations ---

    t2m_kinematic_chain = kinematic_chain

    caption = caption
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
    print(f"[Done] visualization saved to [{animation_save_path}]")

    return abs_path # return sample enclosing folder









def main() -> None:


    quit()
    fps = 20

    # OUTPUT FROM EXPLICIT MODEL...
    out_path_base = f"/oscar/data/csun45/nates_stuff/motion-diffusion-model/uhc-testing/{motion_name}"
    out_path = os.path.join(out_path_base, model_name)
    uhc_output = np.load(f"/oscar/data/csun45/nates_stuff/motion-diffusion-model/uhc-testing/{motion_name}/{model_name}/motion.npy", allow_pickle=True)[None][0]

    # Ground Truth
    gt = uhc_output["gt_jpos"]      # (69, 72)
    gt = gt.reshape(-1, 24, 3)      # (69, 24, 3)
    file_name = "gt.mp4"
    visualize_samples_and_save_to_disk(gt, out_path, file_name, fps)





if __name__ == "__main__":
    main()