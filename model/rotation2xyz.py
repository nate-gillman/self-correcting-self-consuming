# This code is based on https://github.com/Mathux/ACTOR.git
from numbers import Number
from typing import Optional, Tuple, Union

import numpy as np
import torch
from typing_extensions import Literal, get_args

import utils.rotation_conversions as geometry
from model.smpl import JOINTSTYPE_ROOT, SMPL

T_JOINTSTYPE = Literal["a2m", "a2mpl", "smpl", "vibe", "vertices"]
T_POSEREP = Literal["xyz", "rotvec", "rotmat", "rotquat", "rot6d"]


class Rotation2xyz:
    def __init__(self, device, dataset: str = "amass") -> None:
        self.device = device
        self.dataset = dataset
        self.smpl_model = SMPL().eval().to(device)

    def __call__(
        self,
        x,
        mask: Optional[np.ndarray],
        pose_rep: T_POSEREP,
        translation: bool,
        glob: bool,
        jointstype: T_JOINTSTYPE,
        vertstrans: bool,
        betas: Optional[torch.Tensor] = None,
        beta: Number = 0,
        glob_rot: Optional[torch.Tensor] = None,
        get_rotations_back=False,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], torch.Tensor]:
        """
        If the pose_rep = "xyz", this does nothing - simply returns the input x.

        This calls the SMPL model code. Beta(s) and glob/glob_rot are arguments passed
        to its constructor.
        """

        if pose_rep == "xyz":
            # identity
            return x

        if mask is None:
            mask = torch.ones((x.shape[0], x.shape[-1]), dtype=bool, device=x.device)

        if not glob and glob_rot is None:
            raise TypeError("You must specify global rotation if glob is False")

        if jointstype not in get_args(T_JOINTSTYPE):
            raise NotImplementedError(
                f"This jointstype is not implemented. Must be one of: {T_JOINTSTYPE}"
            )

        if pose_rep not in get_args(T_POSEREP):
            raise NotImplementedError(
                f"This pose representation is not implemented. Must be one of: {T_POSEREP}"
            )

        if translation:
            x_translations = x[:, -1, :3]
            x_rotations = x[:, :-1]
        else:
            x_rotations = x

        x_rotations = x_rotations.permute(0, 3, 1, 2)
        nsamples, time, njoints, feats = x_rotations.shape

        # Compute rotations (convert only masked sequences output)
        if pose_rep == "rotvec":
            rotations = geometry.axis_angle_to_matrix(x_rotations[mask])
        elif pose_rep == "rotmat":
            rotations = x_rotations[mask].view(-1, njoints, 3, 3)
        elif pose_rep == "rotquat":
            rotations = geometry.quaternion_to_matrix(x_rotations[mask])
        elif pose_rep == "rot6d":
            rotations = geometry.rotation_6d_to_matrix(x_rotations[mask])
        else:
            raise NotImplementedError("No geometry for this one.")

        if not glob:
            global_orient = torch.tensor(glob_rot, device=x.device)
            global_orient = geometry.axis_angle_to_matrix(global_orient).view(
                1, 1, 3, 3
            )
            global_orient = global_orient.repeat(len(rotations), 1, 1, 1)
        else:
            global_orient = rotations[:, 0]
            rotations = rotations[:, 1:]

        if betas is None:
            # these are the betas passed to the SMPL model, see:
            # https://github.com/vchoutas/smplx/blob/566532a4636d9336403073884dbdd9722833d425/smplx/body_models.py#L59
            betas = torch.zeros(
                [rotations.shape[0], self.smpl_model.num_betas],
                dtype=rotations.dtype,
                device=rotations.device,
            )
            betas[:, 1] = beta

        out = self.smpl_model(
            body_pose=rotations, global_orient=global_orient, betas=betas
        )

        # get the desirable joints
        joints = out[jointstype]

        x_xyz = torch.empty(
            nsamples, time, joints.shape[1], 3, device=x.device, dtype=x.dtype
        )
        x_xyz[~mask] = 0
        x_xyz[mask] = joints

        x_xyz = x_xyz.permute(0, 2, 3, 1).contiguous()

        # the first translation root at the origin on the prediction
        if jointstype != "vertices":
            rootindex = JOINTSTYPE_ROOT[jointstype]
            x_xyz = x_xyz - x_xyz[:, [rootindex], :, :]

        if translation and vertstrans:
            # the first translation root at the origin
            x_translations = x_translations - x_translations[:, :, [0]]

            # add the translation to all the joints
            x_xyz = x_xyz + x_translations[:, None, :, :]

        if get_rotations_back:
            return x_xyz, rotations, global_orient
        else:
            return x_xyz
