import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent.parent))

from r3m import load_r3m
import torch
import torch.nn as nn
from torchvision import transforms

from .misc_utils import JOINT_STATE_DIMS, END_EFF_DIMS, get_end_eff


class ImageJointStateNet(nn.Module):
    def __init__(self, img_comp_dim=32):
        super().__init__()
        self.img_comp_dim = img_comp_dim
        self.js_ee_dim = JOINT_STATE_DIMS + END_EFF_DIMS
        self.output_dim = self.img_comp_dim + self.js_ee_dim

        self.conv_net = nn.Sequential(
            load_r3m("resnet50"),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )

        self.img_squeeze_linear = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.img_comp_dim),
        )

    def forward(self, img, js_ee):
        if not self.training and js_ee.size(-1) + 3 == self.js_ee_dim:
            # add in end eff data
            ee = get_end_eff(js_ee)
            js_ee = torch.cat((js_ee, ee), dim=-1)

        img = self.conv_net(img)
        img_t = self.img_squeeze_linear(img)
        # js_ee_t = torch.zeros_like(js_ee)  # don't use joint_states
        # js_ee_t[-END_EFF_DIMS:] = js_ee[-END_EFF_DIMS:]
        x = torch.cat((img_t, js_ee), dim=1)
        return x

    def load_from_bc(self, bc_state_dict):
        for name, param in self.conv_net.state_dict().items():
            param.copy_(bc_state_dict[f"conv_net.{name}"])
        for name, param in self.img_squeeze_linear.state_dict().items():
            param.copy_(bc_state_dict[f"img_squeeze_linear.{name}"])
