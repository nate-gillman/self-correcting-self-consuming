from typing import List, Tuple

import numpy as np


def get_mean_std(
    data_list: List[np.ndarray], joints_num: int = 22
) -> Tuple[np.ndarray, np.ndarray]:
    """
    From here: https://github.com/EricGuo5513/HumanML3D/blob/main/cal_mean_variance.ipynb

      root_rot_velocity (B, seq_len, 1)
      root_linear_velocity (B, seq_len, 2)
      root_y (B, seq_len, 1)
      ric_data (B, seq_len, (joint_num - 1)*3)
      rot_data (B, seq_len, (joint_num - 1)*6)
      local_velocity (B, seq_len, joint_num*3)
      foot contact (B, seq_len, 4)
    """
    data = np.concatenate(data_list, axis=0)
    Mean = data.mean(axis=0)
    Std = data.std(axis=0)
    Std[0:1] = Std[0:1].mean() / 1.0
    Std[1:3] = Std[1:3].mean() / 1.0
    Std[3:4] = Std[3:4].mean() / 1.0
    Std[4 : 4 + (joints_num - 1) * 3] = Std[4 : 4 + (joints_num - 1) * 3].mean() / 1.0
    Std[4 + (joints_num - 1) * 3 : 4 + (joints_num - 1) * 9] = (
        Std[4 + (joints_num - 1) * 3 : 4 + (joints_num - 1) * 9].mean() / 1.0
    )
    Std[4 + (joints_num - 1) * 9 : 4 + (joints_num - 1) * 9 + joints_num * 3] = (
        Std[4 + (joints_num - 1) * 9 : 4 + (joints_num - 1) * 9 + joints_num * 3].mean()
        / 1.0
    )
    Std[4 + (joints_num - 1) * 9 + joints_num * 3 :] = (
        Std[4 + (joints_num - 1) * 9 + joints_num * 3 :].mean() / 1.0
    )

    assert 8 + (joints_num - 1) * 9 + joints_num * 3 == Std.shape[-1]

    return Mean, Std
