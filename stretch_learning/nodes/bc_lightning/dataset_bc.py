import math
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from skimage import io
from torch.utils.data import Dataset


class VisuomotorDataset(Dataset):
    """ Stretch Overhead Camera dataset """

    def __init__(self, data_dir, csv_file, transform, js_modifications=False):
        super(VisuomotorDataset, self).__init__()
        self.data = pd.read_csv(csv_file)
        self.csv_file = csv_file
        self.data_dir = data_dir
        self.transform = transform
        self.js_modifications = js_modifications
        
        if self.js_modifications:
            print("*************MAKING JS MODIFICATIONS*************")
            print(f"{self.js_modifications=}\n\n")
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = Path(self.data_dir, self.data.loc[idx, "image_path"])
        is_transition = self.data.loc[idx, "is_transition"]

        image = Image.open(img_path)
        image = self.transform(image)
        joint_states = torch.from_numpy(np.array(self.data.loc[idx, "gripper_aperture_pos":"wrist_extension_eff"], dtype=np.float32))
        if self.js_modifications == "only lift essential":
            joint_states = self._only_lift_ess(joint_states)
        elif self.js_modifications == "no velocity":
            joint_states = self._zero_out_vel(joint_states)
        key_pressed = torch.from_numpy(np.array([self.data.loc[idx, "key_pressed"]], dtype=np.float32))

        sample = {"image": image, "joint_states": joint_states, \
            "key_pressed": key_pressed, "is_transition": is_transition}
        return sample

    def _only_lift_ess(self, js):
        new_js = torch.zeros_like(js)
        target_idx = torch.tensor([27, 39])
        new_js[target_idx] = js[target_idx]
        return new_js

    def _only_end_effector_displacement(self, js):
        # gripper is 21cm long, pitch is in radians
        # lift is measured in meters
        # sin(pitch) = delta_y / gripper_len
        new_js = torch.zeros_like(js)
        lift, pitch = js[27].item(), js[30].item()
        gripper_len = 0.21
        gripper_delta_y = gripper_len * math.sin(pitch)
        new_lift = lift + gripper_delta_y
        js[27] = new_lift
        target_idx = torch.tensor([27, 39])
        new_js[target_idx] = js[target_idx]
        return new_js

    def _zero_out_pos_and_vel(self, js):
        target_idx = torch.tensor([i for i in range(2, js.size(0), 3)])
        new_js = torch.zeros_like(js)
        new_js[target_idx] = js[target_idx]
        return new_js

    def _zero_out_vel_and_eff(self, js):
        target_idx = torch.tensor([i for i in range(0, js.size(0), 3)])
        new_js = torch.zeros_like(js)
        new_js[target_idx] = js[target_idx]
        return new_js

    def _zero_out_vel(self, js):
        target_idx = torch.tensor([i for i in range(1, js.size(0), 3)])
        zeros = torch.zeros_like(js)
        js[target_idx] = zeros[target_idx]
        return js
        