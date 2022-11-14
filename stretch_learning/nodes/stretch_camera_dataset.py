import torch
import numpy as np
import pandas as pd
from PIL import Image
from skimage import io
from torch.utils.data import Dataset


class StretchCameraDataset(Dataset):
    """ Stretch Overhead Camera dataset """

    def __init__(self, csv_file, transform):
        super(StretchCameraDataset, self).__init__()
        self.data = pd.read_csv(csv_file)
        self.transform = transform


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_path = self.data.iloc[idx, -1]
        image = Image.open(img_path)
        image = self.transform(image)
        joint_states = torch.from_numpy(np.array(self.data.iloc[idx, 2:-1], dtype=np.float32))
        key_pressed = torch.from_numpy(np.array([self.data.iloc[idx, 1]], dtype=np.float32))
        sample = {"image": image, "joint_states": joint_states, "key_pressed": key_pressed}

        return sample

        
        