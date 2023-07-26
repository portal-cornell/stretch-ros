import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent.parent))
import math
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class VisuomotorDataset(Dataset):
    """Stretch Overhead Camera dataset"""

    def __init__(
        self, data, data_dir, preprocess, augment=None, js_modifications=False
    ):
        super(VisuomotorDataset, self).__init__()
        self.data = data
        self.data_dir = data_dir
        self.js_modifications = js_modifications

        self.preprocess = preprocess
        self.augment = augment

        if self.js_modifications:
            print("*************MAKING JS MODIFICATIONS*************")
            print(f"{self.js_modifications=}\n\n")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        wrist_img_path = Path(self.data_dir, self.data["wrist_image_path"][idx])
        wrist_image = self.preprocess(Image.open(wrist_img_path))
        if self.augment is not None:
            wrist_image = self.augment(wrist_image)
        head_img_path = Path(self.data_dir, self.data["head_image_path"][idx])
        head_image = self.preprocess(Image.open(head_img_path))
        if self.augment is not None:
            head_image = self.augment(head_image)

        joint_states = torch.from_numpy(
            np.array(
                self.data.loc[idx, "gripper_aperture_pos":"wrist_extension_eff"],
                dtype=np.float32,
            )
        )
        if self.js_modifications == "only lift essential":
            joint_states = self._only_lift_ess(joint_states)
        elif self.js_modifications == "no velocity":
            joint_states = self._zero_out_vel(joint_states)
        elif self.js_modifications == "only force":
            joint_states = self._zero_out_pos_and_vel(joint_states)

        key_pressed = torch.from_numpy(
            np.array([self.data.loc[idx, "key_pressed"]], dtype=np.float32)
        )

        sample = {
            "wrist_image": wrist_image,
            "head_image": head_image,
            "joint_states": joint_states,
            "key_pressed": key_pressed,
        }
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
