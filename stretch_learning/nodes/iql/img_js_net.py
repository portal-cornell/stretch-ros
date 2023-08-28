import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent.parent))

import torch
import torch.nn as nn

from torchvision.models import resnet50


JOINT_STATE_DIMS = 14 * 3
END_EFF_DIMS = 3


class ImageJointStateNet(nn.Module):
    def __init__(
        self, img_comp_dims=32, joint_pos=False, joint_vel=False, joint_force=False
    ):
        super().__init__()
        self.img_comp_dims = img_comp_dims
        self.js_ee_dim = JOINT_STATE_DIMS + END_EFF_DIMS
        self.output_dim = self.img_comp_dims + self.js_ee_dim

        conv_net = resnet50(weights=None)
        conv_net.fc = nn.Identity()
        self.img_encoder = nn.Sequential(
            conv_net,
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, img_comp_dims),
            nn.BatchNorm1d(img_comp_dims),
            nn.ReLU(),
        )

        self.joint_pos = joint_pos
        self.joint_vel = joint_vel
        self.joint_force = joint_force

        joint_dims = JOINT_STATE_DIMS // 3
        self.joint_pos_cutoff = joint_dims
        self.joint_vel_cutoff = self.joint_pos_cutoff + joint_dims
        self.joint_force_cutoff = self.joint_vel_cutoff + joint_dims

    def forward(self, img, js_ee):
        img = self.img_encoder(img)
        ee = self.get_end_eff(js_ee)
        if not self.joint_pos:
            js_ee[:, : self.joint_pos_cutoff] = 0
        if not self.joint_vel:
            js_ee[:, self.joint_pos_cutoff : self.joint_vel_cutoff] = 0
        if not self.joint_force:
            js_ee[:, self.joint_vel_cutoff : self.joint_force_cutoff] = 0

        x = torch.cat((img, js_ee), dim=1)
        x = torch.cat((x, ee), dim=1)
        return x

    def get_end_eff(self, js):
        is_single_dim = len(js.shape) == 1
        if is_single_dim:
            js = js.unsqueeze(0)

        lift, pitch = js[:, 27], js[:, 30]
        gripper_len = 0.21
        gripper_delta_y = gripper_len * torch.sin(pitch)
        end_eff_y = lift + gripper_delta_y
        end_eff_x = js[:, 39]
        gripper_left = js[:, 15]
        end_eff_features = torch.cat(
            (end_eff_y.unsqueeze(1), end_eff_x.unsqueeze(1), gripper_left.unsqueeze(1)),
            dim=1,
        )

        return end_eff_features.squeeze() if is_single_dim else end_eff_features

    def load_from_bc(self, bc_state_dict):
        for name, param in self.img_encoder.state_dict().items():
            param.copy_(bc_state_dict[f"img_encoder.{name}"])
