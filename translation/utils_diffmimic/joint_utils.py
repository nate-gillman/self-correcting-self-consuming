# from original repo; this corresponds to the ordering of the indices from the HumanML3D repo
JOINT_NAMES_SMPL = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_index',
    'right_index',
]


# this corresponds to the linear ordering of all the joints in SMPL.py
JOINT_NAMES_HUMANOID = [ 
    'spine1',
    # 'spine2',
    'spine3',
    'neck',
    'right_collar',
    'right_shoulder',
    'right_elbow',
    'right_wrist',
    'left_collar',
    'left_shoulder',
    'left_elbow',
    'left_wrist',
    'right_hip',
    'right_knee',
    'right_ankle',
    'left_hip',
    'left_knee',
    'left_ankle',
    "pelvis"
]

SMPL2IDX = {v: k for k, v in enumerate(JOINT_NAMES_SMPL)}
SMPL2HUMANOID = [SMPL2IDX[j] for j in JOINT_NAMES_HUMANOID]
