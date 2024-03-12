import random
from numbers import Number

import numpy as np
import torch


def fixseed(seed: Number) -> None:
    print(f"Fixing the random seed to: {seed}")
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# SEED = 10
# EVALSEED = 0
# # Provoc warning: not fully functionnal yet
# # torch.set_deterministic(True)
# torch.backends.cudnn.benchmark = False
# fixseed(SEED)
