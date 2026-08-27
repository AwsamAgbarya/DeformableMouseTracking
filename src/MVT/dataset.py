import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class MV_Dataset(Dataset):
    def __init__(self, pose_data, com, projections, n_rotations=18, part_count=10, normalize=True, load_norm_path=None, save_norm_path=None):
        pass