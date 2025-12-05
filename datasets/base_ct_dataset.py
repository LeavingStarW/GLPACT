import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset


class BaseCTDataset(Dataset):
    def __init__(self, data_dir, is_training_set):
        self.data_dir = data_dir
        self.is_training_set = is_training_set


    def __len__(self):
        raise NotImplementedError
    

    def __getitem__(self, item):
        raise NotImplementedError


    