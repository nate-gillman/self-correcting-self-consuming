from typing import Tuple
from collections import namedtuple
import torch
import numpy as np

from translation.mdm_to_amass import VposerSkinResult

PhysicsEvalMetricResult = namedtuple('PhysicsEvalMetricResult', ['float', 'penetrate', 'skate', 'phys_err'])


def get_float_penetrate(body_model_result: VposerSkinResult, tolerance: float = 0.005) -> Tuple[float, float]:
    """
    Implementation based on UHC: 
    https://github.com/ZhengyiLuo/UniversalHumanoidControl/blob/d2bec1793858fe5f71ef79c6643d16a3f563ee65/uhc/smpllib/smpl_eval.py#L125
    """
    # tolerance is given in meters
    # vertex xyz positions are in meters
    # but warning, we return the results in mm, which is how they are measured in results of
    # PhysDiff and UHC.
    vertices = body_model_result['v']

    # HumanML3D and UHC use this definition of floor plane in meters
    floor_plane_z_pos = 0.0

    pen = []
    flo = []
    for vert_i in vertices:
        # go through each frame in the sequence and get z positions of the vertices in meters
        vert_z = vert_i[:, 2]

        # --- GROUND PENETRATE --
        # "for ground penetration (penetrate), we compute the distance
        # between the ground and the lowest body mesh vertex below the
        # ground". - PhysDiff
        p_ind = vert_z < (floor_plane_z_pos - tolerance)
        if torch.any(p_ind):
            pen_i = -vert_z[p_ind].min().item() * 1000
        else:
            pen_i = 0.0
        pen.append(pen_i)


        # --- FLOATING ---
        # "for floating we compute the distance between the
        # ground and the lowest body mesh vertex above the ground"
        # - PhysDiff
        f_ind = vert_z > floor_plane_z_pos
        if torch.any(f_ind):
            min_z_vert = vert_z[f_ind].min().item()
            # put the tolerance check here, otherwise, with high tolerance we are filtering
            # on vertices that are higher in z value in the mesh, which results in higher
            # float, instead of lower.
            flo_i = (min_z_vert * 1000) if (min_z_vert > floor_plane_z_pos + tolerance) else 0.0
        else:
            flo_i = 0.0
        flo.append(flo_i)

    # for a given sequence, the overall float and penetrate is characterized by the
    # mean float / penetrate across all frames, including 0 values when neither occur.
    # return as mm instead of meters (x1,000)
    return np.mean(flo), np.mean(pen)


def get_skate(body_model_result: VposerSkinResult, compute_by_vertex: bool = False, l_foot_joint: int = 10, r_foot_joint: int = 11) -> float:
    """
    Implementation based on UHC: 
    https://github.com/ZhengyiLuo/UniversalHumanoidControl/blob/d2bec1793858fe5f71ef79c6643d16a3f563ee65/uhc/smpllib/smpl_eval.py#L138

    There are differences in the UHC implementation and PhysDiff's description of this metric. The UHC implementation uses
    mesh vertices to compute floor contact and displacements, but PhysDiff expressely says they uses joints to compute this.

    Given how the vertex strategy was used to compute these metrics for UHC, for imitated sequences, it makes sense to use
    that strategy.
    """
    vertices, joints = body_model_result['v'], body_model_result['Jtr']

    floor_plane_z_pos = 0.0

    skate = []

    if compute_by_vertex:
        # this is how UHC computes skate
        for t in range(vertices.shape[0] - 1):
            # find VERTICES that are on or beneath the floor on two consecutive frames
            cind = (vertices[t, :, 2] <= floor_plane_z_pos) & (vertices[t + 1, :, 2] <= floor_plane_z_pos)

            if torch.any(cind):
                # this is how UHC computes skate
                offset = vertices[t + 1, cind, :2] - vertices[t, cind, :2]
                # convert from meters to mm
                s = torch.norm(offset, dim=1).mean().item() * 1000

                skate.append(s)
            else:
                # no skating
                skate.append(0.0)

        return np.mean(skate)
    else:
        # compute by joints
        # "For foot sliding (Skate), we find foot joints that contact the ground in two adjacent frames and
        # compute their average horizontal displacement within the frames." - PhysDiff
        for t in range(joints.shape[0] - 1):
            # find JOINTS that are on or beneath the floor on two consecutive frames
            l_cind = (joints[t, l_foot_joint, 2] <= floor_plane_z_pos) & (vertices[t + 1, l_foot_joint, 2] <= floor_plane_z_pos)
            r_cind = (joints[t, r_foot_joint, 2] <= floor_plane_z_pos) & (vertices[t + 1, r_foot_joint, 2] <= floor_plane_z_pos)

            if not torch.any(l_cind) and not torch.any(r_cind):
                # no contact on either foot, cannot skate
                skate.append(0.0)
            else:
                l_foot_d = torch.tensor([])
                if torch.any(l_cind):
                    # get the delta of the left foot joint
                    l_foot_xy_a = joints[t, l_foot_joint, :2]
                    l_foot_xy_b = joints[t + 1, l_foot_joint, :2]
                    l_foot_d = (l_foot_xy_b - l_foot_xy_a).unsqueeze(0)

                r_foot_d = torch.tensor([])
                if torch.any(r_cind):
                    # get the delta of the right foot joint
                    r_foot_xy_a = joints[t, r_foot_joint, :2]
                    r_foot_xy_b = joints[t + 1, r_foot_joint, :2]
                    r_foot_d = (r_foot_xy_b - r_foot_xy_a).unsqueeze(0)

                # concatenate deltas
                both_foot_d = torch.cat((l_foot_d, r_foot_d), 0)

                # define the overall displacement as the mean of dispalcements of both feet, in mm
                s = torch.norm(both_foot_d, dim=1).mean().item() * 1000

                skate.append(s)

        return np.mean(skate)

def compute_physics_metrics_for_skin_result(
        skin_result: VposerSkinResult, l_foot_joint: int = 10, r_foot_joint: int = 11, tolerance: float = 0.005
) -> PhysicsEvalMetricResult:
    # float and penetrate use the vertices of the skinned SMPL model
    m_float, m_penetrate = get_float_penetrate(skin_result, tolerance=tolerance)

    # skate could use either the vertices or joint positions of the skinned SMPL model depending on strategy, but
    # we use the vertex strategy for both since that is what UHC uses.
    m_skate = get_skate(skin_result, compute_by_vertex=True, l_foot_joint=l_foot_joint, r_foot_joint=r_foot_joint)

    # "The overall physics error metric Phys-Err is the sum of Penetrate, Float, and Skate." - PhysDiff.
    phys_err = m_float + m_penetrate + m_skate

    # return the results, name them for clarity
    return PhysicsEvalMetricResult(m_float, m_penetrate, m_skate, phys_err)
