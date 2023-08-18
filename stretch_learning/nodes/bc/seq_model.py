import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path.cwd().parent.parent))

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
from torchvision.models import resnet18

def get_end_eff(js):
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

class BC_Seq(nn.Module):
    def __init__(
        self,
        skill_name,
        joint_state_dims,
        loss_fn=None,
        accuracy=None,
        lr=1e-3,
        max_epochs=1e5,
        img_comp_dims=32,
        use_wrist_img=True,
        use_head_img=True,
        use_joints=True,
        use_end_eff=True,
        js_modifications="",
        device="cuda",
        o_hor=1,
        a_hor=1,
        reg=0
    ):
        super().__init__()

        # metadata
        self.skill_name = skill_name
        self.device = device

        self.use_wrist_img = use_wrist_img
        self.use_head_img = use_head_img
        self.use_joints = use_joints
        self.use_end_eff = use_end_eff
        self.js_modifications = js_modifications

        # network
        self.o_hor = o_hor
        self.a_hor = a_hor
        self.img_comp_dims = img_comp_dims
        self.joint_state_dims = joint_state_dims * o_hor
        self.end_eff_dims = 3
        self.fc_input_dims = self.img_comp_dims + self.img_comp_dims + self.joint_state_dims + 3

        self.conv_net = resnet18(weights=None)
        self.conv_net.fc = nn.Identity()
        self.conv_net2 = resnet18(weights=None)
        self.conv_net2.fc = nn.Identity()
        self.img_encoder = nn.Sequential(
            self.conv_net,
            # nn.Linear(2048, 512),
            # nn.BatchNorm1d(512),
            # nn.ReLU(),
            # nn.Dropout(p=0.5),
            # nn.Linear(512, 128),
            # nn.BatchNorm1d(128),
            # nn.ReLU(),
            # nn.Dropout(p=0.5),
            # nn.Linear(128, img_comp_dims),

            nn.Linear(512, img_comp_dims),
            nn.LayerNorm(img_comp_dims),
            nn.ReLU(),

            # nn.Linear(512, 128),
            # nn.LayerNorm(128),
            # nn.ReLU(),
            # nn.Dropout(p=0.5),
            # nn.Linear(128, img_comp_dims),
            # nn.ReLU()
        )
        self.img_encoder2 = nn.Sequential(
            self.conv_net2,
            nn.Linear(512, img_comp_dims),
            nn.LayerNorm(img_comp_dims),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dims, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(256, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 4 * a_hor), # was joint_state_dims * a_hor
        )

        # loss/accuracy
        self.loss_fn = loss_fn
        self.accuracy = accuracy
        self.reg = reg
        self.optimizer = optim.AdamW(self.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, max_epochs, 0
        )

    def forward(self, wrist_img, head_img, js_data):
        batch_size = wrist_img.size(0)
        img_t = self.img_encoder(wrist_img)
        img_t2 = self.img_encoder2(head_img)
        device = img_t.device
        if not self.use_wrist_img:
            img_t = torch.zeros((batch_size, self.img_comp_dims))
        if not self.use_head_img:
            img_t2 = torch.zeros((batch_size, self.img_comp_dims))
        if self.use_joints:
            js_t = js_data
        else:
            js_t = torch.zeros((batch_size, self.joint_state_dims))
        if self.use_end_eff:
            ee_t = get_end_eff(js_data)
        else:
            ee_t = torch.zeros((batch_size, self.end_eff_dims))
        img_t = img_t.to(device)
        img_t2 = img_t2.to(device)
        js_t = js_t.to(device)
        ee_t = ee_t.to(device)

        x = torch.cat((js_t, ee_t), dim=1)
        x = torch.cat((img_t, x), dim=1)
        x = torch.cat((img_t2, x), dim=1)
        x = self.fc(x)
        return x

    def integrate_deltas(self, d, isFlat=True):
        B = d.size(0)
        if isFlat:
            d = d.reshape(B, self.a_hor, d.size(1) // self.a_hor)
        d = d.cumsum(axis=-2)
        if isFlat:
            d = d.reshape(B, d.size(1) * d.size(2))
        return d

    def train_loop(self, train_dataloader):
        bad_yaws = 0
        losses, accuracy = [], []
        for batch in tqdm(
            train_dataloader,
            unit_scale=True,
            total=len(train_dataloader),
            position=1,
            desc="train",
            leave=False,
        ):
            self.batch_to_device(batch, self.device)
            wrist_img, head_img, obs_joints, action_joints = batch.values()
            predicted_joints = self(wrist_img, head_img, obs_joints)

            int_pred, int_act = self.integrate_deltas(predicted_joints), self.integrate_deltas(action_joints)
            loss = self.loss_fn(int_pred, int_act) + self.reg * torch.norm(predicted_joints, p=1)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.mean().item())
            acc = self.accuracy(int_pred, int_act)
            accuracy.append(acc.mean().item())
        self.scheduler.step()
        log_dict = {
            "avg_train_loss": np.mean(losses),
            "avg_train_acc": np.mean(accuracy),
            "lr": self.scheduler.get_last_lr()[0],
        }
        return log_dict

    @torch.no_grad()
    def evaluate_loop(self, val_dataloader):
        losses, accuracy = [], []
        for batch in tqdm(
            val_dataloader,
            unit_scale=True,
            total=len(val_dataloader),
            position=1,
            desc="validation",
            leave=False,
        ):
            self.batch_to_device(batch, self.device)
            wrist_img, head_img, obs_joints, action_joints = batch.values()
            predicted_joints = self(wrist_img, head_img, obs_joints)

            int_pred, int_act = self.integrate_deltas(predicted_joints), self.integrate_deltas(action_joints)
            loss = self.loss_fn(int_pred, int_act) + self.reg * torch.norm(predicted_joints, p=1)
            losses.append(loss.mean().item())
            acc = self.accuracy(int_pred, int_act)
            accuracy.append(acc.mean().item())
        log_dict = {
            "avg_val_loss": np.mean(losses),
            "avg_val_acc": np.mean(accuracy),
        }
        return log_dict

    def batch_to_device(self, batch, device):
        for key, value in batch.items():
            batch[key] = value.to(device)