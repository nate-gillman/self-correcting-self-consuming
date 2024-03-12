import os
import functools
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
from PIL import Image, ImageEnhance
import trimesh
from tqdm import tqdm

from body_visualizer.mesh.mesh_viewer import MeshViewer
from body_visualizer.tools.mesh_tools import rotateXYZ
from translation.mdm_to_amass import VposerSkinResult

# sRGB color space
_LIME_GREEN_COLOR = [0.5, 1.0, 0.0, 1.0]
_INDIGO_COLOR = [0.5, 0.0, 0.3, 1.0]

# undo the 90-degree rotation that may be done by rot_body
UNDO_ROT_90 = np.array([
    [1, 0, 0, 0],
    [0, np.cos(np.radians(-90)), -np.sin(np.radians(-90)), 0],
    [0, np.sin(np.radians(-90)), np.cos(np.radians(-90)), 0],
    [0, 0, 0, 1]
])

SMPL_KINEMATIC_TREE = [
    # legs
    (0, 1),
    (0, 2),
    (2, 5),
    (5, 8),
    (8, 11),
    (1, 4),
    (4, 7),
    (7, 10),
    # upper body
    (15, 12),
    (12, 17),
    (12, 16),
    (17, 19),
    (19, 21),
    (16, 18),
    (18, 20),
    (20, 22),
    # spine
    (12, 9),
    (9, 6),
    (6, 3),
    (3, 0),
]

RED = [1.0, 0.0, 0.1, 1.0]
GREEN = [0.0, 1.0, 0.1, 1.0]
BLUE = [0.0, 0.15, 1.0, 1.0]
SILVER_BLUE = [0.0, 0.53, 1.0, 1.0]
YELLOW = [1.0, 0.5, 0.12, 1.0]

JOINT_COLORS = {
    7: RED,
    10: RED,
    8: GREEN,
    11: GREEN,
    2: YELLOW,
    1: YELLOW,
    5: YELLOW,
    4: YELLOW,
    'default': BLUE
}

BONE_COLORS = {
    (0, 1): YELLOW,
    (0, 2): YELLOW,
    (2, 5): YELLOW,
    (5, 8): YELLOW,
    (8, 11): YELLOW,
    (1, 4): YELLOW,
    (4, 7): YELLOW,
    (7, 10): YELLOW,
    'default': SILVER_BLUE
}


def render_smpl_params(
        body_model_result: VposerSkinResult,
        img_width: int = 800,
        img_height: int = 800,
        rot_body=None,
        camera_translation=None,
        person_color_rgb_perc=None,
        floor_color_rgb_perc=None
) -> np.ndarray:
    """
    Code modified from: 
        https://github.com/nghorbani/body_visualizer/blob/5d0ac4615f32563f5dbd2f09c24aee75e3ae62ff/src/body_visualizer/tools/vis_tools.py
    Originally authored by Nima Ghorbani (https://nghorbani.github.io/)
    
    See more on classes called by this function here:
        https://github.com/nghorbani/body_visualizer/blob/5d0ac4615f32563f5dbd2f09c24aee75e3ae62ff/src/body_visualizer/mesh/mesh_viewer.py#L37
        https://github.com/MPI-IS/mesh/blob/master/mesh/meshviewer.py#L159
        https://github.com/MPI-IS/mesh/blob/49e70425cf373ec5269917012bda2944215c5ccd/mesh/meshviewer.py#L645
    """
    vertices = body_model_result['v']
    faces = body_model_result['f']

    import pyrender
    imw, imh = img_width, img_height

    # -- scene setup --
    # On macOS, use_offscreen must be True!
    mv = MeshViewer(width=imw, height=imh, use_offscreen=True)

    # makes objects look a bit more realistic, and without too much render time impact
    mv.use_raymond_lighting()

    # zoom way out on the z plane to give more room to see the full sequence
    if camera_translation is None:
        # use this very zoomed out default
        mv.set_cam_trans([0.0, 2.0, 10.0])
    else:
        mv.set_cam_trans(camera_translation)

    # -- texture setup --
    # lime green
    person_color = person_color_rgb_perc or _LIME_GREEN_COLOR
    # https://pyrender.readthedocs.io/en/latest/generated/pyrender.MetallicRoughnessMaterial.html
    person_texture = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=person_color,
        smooth=False,
        metallicFactor=0.4,
        roughnessFactor=0.0,
    )

    # indigo
    floor_color = floor_color_rgb_perc or _INDIGO_COLOR
    floor_texture = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=floor_color,
        # smooth=False,
        metallicFactor=0.2,
        roughnessFactor=0.3,
    )

    # -- objects setup --
    # Create a plane mesh, but rotate it -90 deg about the x-axis to be consistent with
    # the orientation of the SMPL mesh
    plane_size = 10

    # plane object definition from: https://gist.github.com/wkentaro/c245a9d47b4d6a4c51424c2c144d6e18
    plane = pyrender.Mesh.from_trimesh(
        trimesh.creation.box(extents=(plane_size, 5, 0.01), transform=UNDO_ROT_90),
        smooth=False,
        material=floor_texture
    )

    # prepare dimensions of human meshes
    v = vertices
    t, num_verts = v.shape[:-1]

    # frames
    images = []

    # -- rendering --
    for fIdx in range(t):
        # for each frame in the sequence, get the vertices of that frame
        verts = v[fIdx]

        if rot_body is not None:
            # rotate it if necessary
            verts = rotateXYZ(verts, rot_body)

        # render the person
        human_mesh = pyrender.Mesh.from_trimesh(trimesh.base.Trimesh(
            verts,
            faces,
        ), material=person_texture)

        # put the 3d objects into the scene, which are the person and the floor plane
        mv.set_meshes([human_mesh, plane], 'static')

        # render the image, append the pixels to the image array for later
        images.append(mv.render())

    return np.array(images).reshape(len(images), imw, imh, 3)


def _get_cylinder_transform(joint1, joint2) -> np.ndarray:
    midpoint = (joint1 + joint2) / 2.0  # noqa

    # get the direction vector
    direction = joint2 - joint1

    # height of the cylinder should be the distance between the two joints
    height = np.linalg.norm(direction)

    # make the direction vector a unit vector
    direction /= height

    # get rotation axis and angle
    rotation_axis = np.cross(np.array([0, 0, 1]), direction)  # noqa
    rotation_angle = np.arccos(np.dot(np.array([0, 0, 1]), direction))

    # create rotation matrix, using axis angle
    rotation_matrix = np.eye(4)
    rotation_matrix[:3, :3] = trimesh.transformations.rotation_matrix(rotation_angle, rotation_axis)[:3, :3]

    # create translation matrix, so that joints are connected by the cylinder
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = midpoint

    # the cylinder should connect the two joints
    scale_matrix = np.eye(4)
    scale_matrix[2, 2] = height

    # compose them to a single transform
    transform_matrix = translation_matrix @ rotation_matrix @ scale_matrix

    return transform_matrix


def _combine_layers(foreground, background) -> np.ndarray:
    """Given two images of the same size as numpy arrays in RGBA color space, superimpose foreground over background.

    Source: https://stackoverflow.com/a/59211216/23160579
    """
    result = background.copy()

    # normalize alpha channels from 0-255 to 0-1
    alpha_background = background[:, :, 3] / 255.0
    alpha_foreground = foreground[:, :, 3] / 255.0

    # calculate blended colors
    for color in range(0, 3):
        result[:, :, color] = (
                (1.0 - alpha_foreground) * result[:, :, color] +
                alpha_foreground * foreground[:, :, color]
        )

    # set adjusted alpha and denormalize back to 0-255
    result[:, :, 3] = (1.0 - (1.0 - alpha_foreground) * (1.0 - alpha_background)) * 255.0

    return result.astype(np.uint8)


def _color_correct(image_array) -> np.ndarray:
    # convert to PIL image object
    image = Image.fromarray(image_array, 'RGBA')

    # increase contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.19)

    # increase brightness
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.05)

    # increase saturation
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(1.3)

    # return the color-corrected image as a numpy array
    return np.array(image)


def render_smpl_params_with_joint_colors(
        body_model_result: VposerSkinResult,
        img_width: int = 800,
        img_height: int = 800,
        rot_body=None,
        camera_translation=None,
        person_color_rgb_perc=None,
        floor_color_rgb_perc=None,
        take_snapshots: bool = True,
        snapshot_interval: int = 20,
) -> np.ndarray:
    # extract information we need from vposer
    vertices = body_model_result['v']
    faces = body_model_result['f']
    joints = body_model_result['Jtr']
    v = vertices
    t, num_verts = v.shape[:-1]

    import pyrender
    imw, imh = img_width, img_height

    # -- scene setup --
    # use_offscreen must be True! On macOS
    mv = MeshViewer(width=imw, height=imh, use_offscreen=True)

    # makes objects look a bit more realistic, and without too much render time impact
    mv.use_raymond_lighting()

    # zoom way out on the z plane to give more room to see the full sequence
    if camera_translation is None:
        # use this very zoomed out default
        mv.set_cam_trans([0.0, 2.0, 10.0])
    else:
        mv.set_cam_trans(camera_translation)

    # -- texture setup --
    person_color = person_color_rgb_perc or [0.5, 1.0, 0.0, 1.0]
    # https://pyrender.readthedocs.io/en/latest/generated/pyrender.MetallicRoughnessMaterial.html
    person_texture = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=person_color,
        smooth=False,
        metallicFactor=0.4,
        roughnessFactor=0.0,
    )

    # --- CREATE THE FLOOR ---
    floor_color = floor_color_rgb_perc or [0.3, 0.15, 0.3, 1.0]
    floor_texture = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=floor_color,
        metallicFactor=0.2,
        roughnessFactor=0.3,
    )
    plane_size = 10

    # plane object definition from: https://gist.github.com/wkentaro/c245a9d47b4d6a4c51424c2c144d6e18
    plane = pyrender.Mesh.from_trimesh(
        trimesh.creation.box(extents=(plane_size, 5, 0.01), transform=UNDO_ROT_90),
        smooth=False,
        material=floor_texture
    )

    # limit to how many frames to render, useful in tests - set it to an integer to limit the number
    # of frames this function will render
    limit: Optional[int] = None

    # this gets the 'core' skeleton with no hands.
    joint_limit = 24

    # turn to 1 if we want skeleton fully super-imposed over the human skin
    skeleton_alpha = 0.65
    fade_factor = 0.99  # exponential fall-off
    snapshot_interval = snapshot_interval

    # initialize textures
    joint_textures = {
        x: pyrender.MetallicRoughnessMaterial(
            baseColorFactor=y,
            metallicFactor=0.2,
            roughnessFactor=0.3,
            alphaMode="BLEND",
        ) for x, y in JOINT_COLORS.items()}
    bone_textures = {
        x: pyrender.MetallicRoughnessMaterial(
            baseColorFactor=y,
            metallicFactor=0.2,
            roughnessFactor=0.3,
            alphaMode="BLEND",
        ) for x, y in BONE_COLORS.items()}

    # -- rendering --
    images = []
    prev_snapshots = []
    for fIdx in tqdm(range(t)):
        if limit is not None and (fIdx > limit):
            # useful for debugging, just set limit to an integer
            break

        # --- LAYER 1: The Skeleton ---
        if rot_body:
            # rotate the joints if necessary
            jpos = rotateXYZ(joints[fIdx][:joint_limit], rot_body)
        else:
            jpos = joints[fIdx][:joint_limit]

        # create bone geometry (cylinders)
        bones = []
        for connection in SMPL_KINEMATIC_TREE:
            # using the joint positions, draw cylinders that connect them
            src_joint_idx, dst_joint_idx = connection

            # transform is constructed assuming unit length of cylinder
            transform_matrix = _get_cylinder_transform(jpos[src_joint_idx], jpos[dst_joint_idx])

            # the height should always be 1, otherwise transform is broken
            cylinder_geom = trimesh.creation.cylinder(radius=0.02, height=1)
            cylinder_geom.apply_transform(transform_matrix)

            # make this an object in our scene and add it to the mesh array
            cylinder_mesh = pyrender.Mesh.from_trimesh(
                cylinder_geom,
                smooth=True,
                material=bone_textures.get(connection, bone_textures['default'])
            )
            bones.append(cylinder_mesh)

        # create joint geometry (spheres)
        joints_meshes = []
        for j_idx in range(jpos.shape[0]):
            # just put a sphere at a joint position
            single_joint_position = jpos[j_idx]
            ball_joint_geom = trimesh.creation.icosphere(radius=0.05)
            ball_joint_geom.apply_translation(single_joint_position)
            ball_joint_obj = pyrender.Mesh.from_trimesh(
                ball_joint_geom,
                smooth=True,
                material=joint_textures.get(j_idx, joint_textures['default']),
            )
            joints_meshes.append(ball_joint_obj)

        # add the bone and joint meshes to the scene
        mv.set_meshes([*joints_meshes, *bones], 'static')

        # make the background transparent and render with alpha channel enabled
        mv.set_background_color([0, 0, 0, 0])
        _skeletal_layer = mv.render(RGBA=True)

        skeletal_layer = _skeletal_layer.copy()
        skeletal_layer[:, :, 3] = (_skeletal_layer[:, :, 3] * skeleton_alpha)

        # --- LAYER 0: The Skinned Human Mesh ---
        # for each frame in the sequence, get the human vertices
        verts = v[fIdx]
        if rot_body is not None:
            # rotate it if necessary
            verts = rotateXYZ(verts, rot_body)

        # put the human in the scene and give it a custom texture
        human_trimesh = trimesh.base.Trimesh(
            verts,
            faces,
        )
        human_mesh = pyrender.Mesh.from_trimesh(human_trimesh, material=person_texture)

        # white background
        mv.set_background_color([1.0, 1.0, 1.0, 1.0])
        mv.set_meshes([human_mesh, plane], 'static')
        base_layer = mv.render(RGBA=True)

        # need this to do a trick where we get the floor + shadows, but it doesn't block
        # the previous 'snapshot' images
        human_mesh = pyrender.Mesh.from_trimesh(human_trimesh, material=person_texture)

        # transparent background
        mv.set_background_color([0, 0, 0, 0])
        mv.set_meshes([human_mesh], 'static')
        human_layer = mv.render(RGBA=True)

        # the base frame is the human mesh with the skeleton over it
        frame = _combine_layers(skeletal_layer, human_layer)

        if take_snapshots and (fIdx % snapshot_interval) == 0:
            # flatten the previous snapshot into a frame
            if prev_snapshots:
                prev_snapshots.append(_combine_layers(frame, prev_snapshots.pop(0)))
            else:
                prev_snapshots.append(frame)

        # add exponential decay to previous snapshots by alpha channel
        prev_snapshots[-1][:, :, 3] = np.clip(prev_snapshots[-1][:, :, 3] * fade_factor, 0, 255)

        # combine all the layers together, with left-most (0th index as the top, and last element as the bottom)
        final_frame = functools.reduce(_combine_layers, [skeletal_layer, human_layer, prev_snapshots[-1], base_layer])

        # add color correction like brightness, saturation, contrast
        images.append(_color_correct(final_frame))

    # return as an array of RGBA images in numpy array
    return np.array(images).reshape(len(images), imw, imh, 4)


def image_array2file(img_array: np.ndarray, outpath: Optional[str] = None, fps: int = 30) -> str:
    """
    Code modified from:
        https://github.com/nghorbani/body_visualizer/blob/5d0ac4615f32563f5dbd2f09c24aee75e3ae62ff/src/
        body_visualizer/tools/vis_tools.py

    Originally authored by Nima Ghorbani (https://nghorbani.github.io/)
    
    :param img_array: R x C x T x height x width x 3
    :param outpath: the directory where T images will be dumped for each time point in range T
    :param fps: fps of the gif file
    :return:
        it will return an image list with length T
        if outpath is given as a png file, an image will be saved for each t in T.
        if outpath is given as a gif file, an animated image with T frames will be created.
    """

    if outpath is not None:
        outdir = os.path.dirname(outpath)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

    if not isinstance(img_array, np.ndarray) or img_array.ndim < 6:
        raise ValueError('img_array should be a numpy array of shape R x C x T x height x width x 3')

    R, C, T, img_h, img_w, img_c = img_array.shape  # noqa

    out_images = []
    for tIdx in range(T):
        row_images = []
        for rIdx in range(R):
            col_images = []
            for cIdx in range(C):
                col_images.append(img_array[rIdx, cIdx, tIdx])
            row_images.append(np.hstack(col_images))
        t_image = np.vstack(row_images)
        out_images.append(t_image)

    if outpath is not None:
        ext = outpath.split('.')[-1]
        if ext in ['png', 'jpeg', 'jpg']:
            for tIdx in range(T):
                if T > 1:
                    cur_outpath = outpath.replace('.%s' % ext, '_%03d.%s' % (tIdx, ext))
                else:
                    cur_outpath = outpath
                img = out_images[tIdx]

                # image is of shape: (width, height, color_channels)
                cv2.imwrite(cur_outpath, img)
                while not os.path.exists(cur_outpath):
                    # wait until the snapshot is written to the disk
                    continue
        elif ext == 'gif':
            import imageio

            # more info: https://imageio.readthedocs.io/en/stable/reference/userapi.html#imageio.v2.get_writer
            with imageio.get_writer(outpath, mode='I', fps=fps, loop=0) as writer:
                for tIdx in range(T):
                    img = out_images[tIdx].astype(np.uint8)
                    writer.append_data(img)

    return outpath


def skin_result_to_video(
        body_model_result: VposerSkinResult,
        output_path: str,
        h: int = 800,
        w: int = 800,
        camera_translation=None,
        person_color_rgb_perc=None,
        floor_color_rgb_perc=None,
        with_joint_and_snapshots: bool = False,
        # these do not do anything if super_duper_mode is off
        take_snapshots: bool = True,
        snapshot_interval: int = 20,
) -> None:
    """
    Code modified from: https://github.com/nghorbani/body_visualizer/blob/
        5d0ac4615f32563f5dbd2f09c24aee75e3ae62ff/src/body_visualizer/tools/vis_tools.py

    Originally authored by Nima Ghorbani (https://nghorbani.github.io/)
    """
    # the rot_body specifies, in degrees, how to rotate the mesh. We do this for visualization purposes
    # but not recommended to do it when computing metrics. 
    if not with_joint_and_snapshots:
        images = render_smpl_params(
            body_model_result,
            img_width=w,
            img_height=h,
            rot_body=[-90, 0, 0],
            camera_translation=camera_translation,
            person_color_rgb_perc=person_color_rgb_perc,
            floor_color_rgb_perc=floor_color_rgb_perc
        )
        # dimensions are: R x C x sequence_length x img_height x img_width x num_color_channels (RGB)
        t = images.shape[0]
        reshaped_image = images.reshape(1, 1, t, h, w, 3)
    else:
        images = render_smpl_params_with_joint_colors(
            body_model_result,
            img_width=w,
            img_height=h,
            rot_body=[-90, 0, 0],
            camera_translation=camera_translation,
            person_color_rgb_perc=person_color_rgb_perc,
            floor_color_rgb_perc=floor_color_rgb_perc,
            take_snapshots=take_snapshots,
            snapshot_interval=snapshot_interval
        )
        # dimensions are: R x C x sequence_length x img_height x img_width x num_color_channels (RGBA)
        t = images.shape[0]
        reshaped_image = images.reshape(1, 1, t, h, w, 4)

    # save to disk
    image_array2file(reshaped_image, outpath=output_path)


def skin_result_to_video_test() -> None:
    """Just for testing purposes.
    """
    rel_p = Path(__file__).parent

    # get vposer skin result
    def load_vposer_output() -> VposerSkinResult:
        # unwrap the weird way numpy saves objects
        _f = rel_p / "000062-sampled_skin.npy"
        _loaded_result = np.load(_f, allow_pickle=True)
        return _loaded_result[None][0]

    # load the skin result pickle / numpy
    skin_result: VposerSkinResult = load_vposer_output()

    # use the skin result to render to disk
    skin_result_to_video(
        skin_result,
        str(rel_p / "example_skin_result.gif"),
        800,
        1200,
        camera_translation=[0.0, 1.0, 5.0],
        with_joint_and_snapshots=True
    )


if __name__ == "__main__":
    skin_result_to_video_test()
