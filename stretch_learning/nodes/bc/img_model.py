import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path.cwd().parent.parent))

import numpy as np
from tqdm import tqdm
import torchvision
import torch
import torch.nn as nn
from torch import optim
from torchvision.models import resnet18,convnext_tiny
import torchmetrics
# from utils.common import get_end_eff,get_end_eff_yaw_ext
gripper_len = 0.22
base_gripper_yaw = -0.09
def get_end_eff_yaw_ext(js):
    is_single_dim = len(js.shape) == 1
    if is_single_dim:
        js = js.unsqueeze(0)

    yaw, ext = js[:, 36], js[:, 39]
    yaw_delta = -(yaw - base_gripper_yaw)  # going right is more negative
    yaw_delta = yaw_delta
    y = gripper_len * torch.cos(yaw_delta) + ext
    x = gripper_len * torch.sin(yaw_delta)

    return x, y
# class MLP(nn.Module):
#     def __init__(self,input_size,output_size,hidden_dim, num_layers, dropout = 0.5, norm = nn.LayerNorm, activation= nn.GELU):
#         super().__init__()
#         assert num_layers >= 2
#         self.layers = []
#         self.layers.append(nn.Linear(input_size,hidden_dim))
#         self.layers.append(norm(hidden_dim))
#         self.layers.append(nn.Dropout(dropout))
#         self.layers.append(activation())

#         for i in range(num_layers - 2):
#             self.layers.append(nn.Linear(hidden_dim,hidden_dim))
#             self.layers.append(norm(hidden_dim))
#             self.layers.append(nn.Dropout(dropout))
#             self.layers.append(activation())
#         self.layers.append(nn.Linear(hidden_dim,output_size))
#         self.model = nn.Sequential(*self.layers)
#     def forward(self,x):
#         return self.model(x)
class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            hidden_dim = 100
            self.num_classes = 4
            self.fc_input_dims = 4
            self.linear1 = nn.Linear(self.fc_input_dims, hidden_dim)
            self.linear2 = nn.Linear(hidden_dim, hidden_dim)
            self.linear3 = nn.Linear(hidden_dim, self.num_classes)
            self.dropout1 = nn.Dropout(p=0.5)
            self.dropout2 = nn.Dropout(p=0.5)
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.norm2 = nn.LayerNorm(hidden_dim)
            self.activation = nn.GELU()

            self.fc = nn.Sequential(
                self.linear1,
                self.norm1,
                self.dropout1,
                self.activation,
                self.linear2,
                self.norm2,
                self.dropout2,
                self.activation,
                self.linear3,
            )
        def forward(self,x):
            return self.fc(x)


import math

def adjust_learning_rate(optimizer, epoch ,lr,min_lr, num_epochs, warmup_epochs):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < warmup_epochs:
        lr = lr * epoch / warmup_epochs 
    else:
        lr = min_lr + (lr - min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


class BC(nn.Module):
    def __init__(
        self,
        skill_name,
        joint_state_dims,
        num_classes,
        lr=1e-3,
        max_epochs=1e5,
        img_comp_dims=32,
        use_wrist_img=True,
        use_head_img=True,
        use_end_eff=True,
        device="cuda",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_classes, top_k=1
        )
        self.max_epochs = max_epochs
        self.warmup_epochs = max_epochs // 10
        # metadata
        self.skill_name = skill_name
        self.device = device
        self.lr = lr
        self.min_lr = lr/1e3
        self.use_wrist_img = use_wrist_img
        self.use_head_img = use_head_img
        self.use_end_eff = use_end_eff

        # network
        self.img_comp_dims = img_comp_dims
        self.joint_state_dims = joint_state_dims
        self.end_eff_dims = 2
        self.fc_input_dims = img_comp_dims + img_comp_dims

        self.conv_net = convnext_tiny(torchvision.models.convnext.ConvNeXt_Tiny_Weights.DEFAULT)
        self.conv_net.fc = nn.Identity()
        self.conv_net2 = convnext_tiny(torchvision.models.convnext.ConvNeXt_Tiny_Weights.DEFAULT)
        self.conv_net2.fc = nn.Identity()

        self.fc = nn.Sequential(
            nn.Linear(2000, 100),
            nn.LayerNorm(100),
            nn.GELU(),
            nn.Linear(100, 100),
            nn.LayerNorm(100),
            nn.GELU(),
            nn.Linear(100, 2),
        )
        config = {
            "input_size": 4,
            "output_size": num_classes,
            "hidden_dim": 100,
            "num_layers": 3,
            "dropout": 0.5,
            "norm": nn.LayerNorm,
            "activation": nn.GELU
        }
        # self.fc_last = MLP(**config)
        self.fc_last = MLP()
        # state_dict = torch.load("/share/portal/jlr429/hal-skill-repo/point_and_shoot_debug/ckpts/20230905-180606_use_delta/epoch=900_mean_deltas=0.021.pt", map_location=torch.device(device))
        # Iterate over the weights dictionary
        

        # self.fc_last.load_state_dict(state_dict)
        # loss/accuracy
        self.optimizer = optim.AdamW([
                {'params': self.conv_net.parameters()},
                {'params': self.conv_net2.parameters()},
                {'params': self.fc.parameters()}
            ], lr=lr)

    def forward(self, wrist_img, head_img, js_data):
        batch_size = wrist_img.size(0)
        device = wrist_img.device
        if self.use_wrist_img:
            img_t = self.conv_net(wrist_img)
        else:
            img_t = torch.zeros((batch_size, self.img_comp_dims))
        if self.use_head_img:
            img_t2 = self.conv_net2(head_img)
        else:
            img_t2 = torch.zeros((batch_size, self.img_comp_dims))
        if self.use_end_eff:
            curr_x, curr_y = get_end_eff_yaw_ext(js_data)
        else:
            curr_x = torch.zeros((batch_size, 1))
            curr_y = torch.zeros((batch_size, 1))
        img_t = img_t.to(device)
        img_t2 = img_t2.to(device)
        curr_x = curr_x.to(device).unsqueeze(1)
        curr_y = curr_y.to(device).unsqueeze(1)

        x = torch.cat((img_t, img_t2), dim=1)
        x = self.fc(x)
        goal_x,goal_y = x[:,0].unsqueeze(1),x[:,1].unsqueeze(1)
        x = torch.cat((curr_x,curr_y,(goal_x-curr_x), (goal_y-curr_y)), dim=1)
        x = self.fc_last(x)
        return x

    def train_loop(self, train_dataloader,epoch,ctx, scaler):
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
            wrist_img, head_img, joint_state, key_pressed = batch.values()
            idx = good_yaw_only(joint_state)
            # import pdb; pdb.set_trace()
            wrist_img, head_img, joint_state, key_pressed = wrist_img[idx], head_img[idx], joint_state[idx], key_pressed[idx]
            if joint_state.size == 0:
                print("Unlikely event of all bad yaw")
                continue
            with ctx:
                predicted_action = self(wrist_img, head_img, joint_state)
                actual_action = key_pressed.to(torch.int64).squeeze()
                loss = self.loss_fn(predicted_action, actual_action)

            self.optimizer.zero_grad()
            scaler.scale(loss).backward()
            # loss.backward()
            scaler.step(self.optimizer)
            # self.optimizer.step()
            scaler.update()    

            losses.append(loss.mean().item())
            acc = self.accuracy(predicted_action, actual_action)
            accuracy.append(acc.mean().item())
        lr = adjust_learning_rate(self.optimizer,epoch,self.lr,self.min_lr,self.max_epochs,self.warmup_epochs)
        log_dict = {
            "avg_train_loss": np.mean(losses),
            "avg_train_acc": np.mean(accuracy),
            "lr": lr,
        }
        return log_dict

    @torch.no_grad()
    def evaluate_loop(self, val_dataloader,ctx):
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
            wrist_img, head_img, joint_state, key_pressed = batch.values()
            idx = good_yaw_only(joint_state)
            wrist_img, head_img, joint_state, key_pressed = wrist_img[idx], head_img[idx], joint_state[idx], key_pressed[idx]
            if joint_state.size == 0:
                print("Unlikely event of all bad yaw")
                continue
            # predicted_action = self(wrist_img, joint_state, image2=head_img)
            with ctx:
                predicted_action = self(wrist_img, head_img, joint_state)
                actual_action = key_pressed.to(torch.int64).squeeze()
                loss = self.loss_fn(predicted_action, actual_action)
            losses.append(loss.mean().item())
            acc = self.accuracy(predicted_action, actual_action)
            accuracy.append(acc.mean().item())
        log_dict = {
            "avg_val_loss": np.mean(losses),
            "avg_val_acc": np.mean(accuracy),
        }
        return log_dict

    def batch_to_device(self, batch, device):
        for key, value in batch.items():
            batch[key] = value.to(device)

def good_yaw_only(joint_states):
    from dataset import joint_labels
    yaw = joint_states[:,joint_labels.index("joint_wrist_yaw_pos")]
    return yaw < 1.5