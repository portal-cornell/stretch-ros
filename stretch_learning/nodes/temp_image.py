import sys
from pathlib import Path
from typing import Any, Mapping
from collections import Counter
sys.path.insert(0, str(Path.cwd().parent.parent))

import numpy as np
from tqdm import tqdm
import torchvision
import torch
import torch.nn as nn
from torch import optim
from torchvision.models import resnet18,convnext_tiny
import torchmetrics
from utils.common import get_end_eff,get_end_eff_yaw_ext
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import random
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


class MLPBlock(nn.Module):
    def __init__(self,input_size,output_size,info_size = 0,dropout = 0.5, norm = nn.LayerNorm, activation= nn.GELU):
        super().__init__()
        self.layers = []
        self.layers.append(nn.Linear(input_size+info_size,output_size))
        self.layers.append(norm(output_size))
        self.layers.append(nn.Dropout(dropout))
        self.layers.append(activation())
        self.model = nn.Sequential(*self.layers)
        self.use_residual = input_size == output_size
         
    def forward(self,x,info=None):
        if info is not None:
            x_new = torch.cat((x,info),dim = 1)
            if self.use_residual:
                return x + self.model(x_new)
            else:
                return self.model(x_new)
        else:
            if self.use_residual:
                return x + self.model(x)
            else:
                return self.model(x)
class MLPAdvanced(nn.Module):
    def __init__(self,input_size,output_size,hidden_dim, num_layers, info_size = 0,dropout = 0.5, norm = nn.LayerNorm, activation= nn.GELU):
        super().__init__()
        assert num_layers >= 2
        self.layers = []

        self.pre_fc = MLPBlock(input_size,hidden_dim,info_size=0,dropout=dropout,norm=norm,activation=activation)
        for i in range(num_layers - 2):

            self.layers.append(MLPBlock(hidden_dim,hidden_dim,info_size,dropout=dropout,norm=norm,activation=activation))
        self.layers = nn.ModuleList(self.layers)

        self.fc_last = nn.Linear(hidden_dim+info_size,output_size)
        
    def forward(self,x,info):
        x = self.pre_fc(x)
        for layer in self.layers:
            x = layer(x,info)
        x = torch.cat((x,info),dim=1)
        return self.fc_last(x)
class MLP(nn.Module):
    def __init__(self,input_size,output_size,hidden_dim, num_layers, info_size = 0,dropout = 0.5, norm = nn.LayerNorm, activation= nn.GELU):
        super().__init__()
        assert num_layers >= 2
        self.layers = []
        self.layers.append(nn.Linear(input_size,hidden_dim))
        self.layers.append(norm(hidden_dim))
        self.layers.append(nn.Dropout(dropout))
        self.layers.append(activation())
        

        for i in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim,hidden_dim))
            self.layers.append(norm(hidden_dim))
            self.layers.append(nn.Dropout(dropout))
            self.layers.append(activation())
        self.layers.append(nn.Linear(hidden_dim,output_size))
        self.model = nn.Sequential(*self.layers)
    def forward(self,x):
        return self.model(x)
# class MLP(nn.Module):
#         def __init__(self):
#             super().__init__()
#             hidden_dim = 100
#             self.num_classes = 4
#             self.fc_input_dims = 4
#             self.linear1 = nn.Linear(self.fc_input_dims, hidden_dim)
#             self.linear2 = nn.Linear(hidden_dim, hidden_dim)
#             self.linear3 = nn.Linear(hidden_dim, self.num_classes)
#             self.dropout1 = nn.Dropout(p=0.5)
#             self.dropout2 = nn.Dropout(p=0.5)
#             self.norm1 = nn.LayerNorm(hidden_dim)
#             self.norm2 = nn.LayerNorm(hidden_dim)
#             self.activation = nn.GELU()

#             self.fc = nn.Sequential(
#                 self.linear1,
#                 self.norm1,
#                 self.dropout1,
#                 self.activation,
#                 self.linear2,
#                 self.norm2,
#                 self.dropout2,
#                 self.activation,
#                 self.linear3,
#             )
#         def forward(self,x):
#             return self.fc(x)


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
        weights = torch.FloatTensor([0.43317524,0.0,3.39238095,2.52087757])
        self.loss_fn = nn.CrossEntropyLoss(weight=weights)
        self.goal_loss = nn.SmoothL1Loss()
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
            # nn.Linear(2000,100),
            # nn.LayerNorm(100),
            # nn.GELU(),
            # nn.Linear(100,100),
            # nn.LayerNorm(100),
            # nn.GELU(),
            # nn.Linear(100,2)
            MLPBlock(2000,100,dropout=0),
            MLPBlock(100,100,dropout=0),
            MLPBlock(100,100,dropout=0),
            nn.Linear(100,2)
        )
        config = {
            "input_size": 2,
            "output_size": num_classes,
            "hidden_dim": 100,
            "num_layers": 3,
            "dropout": 0.5,
            "norm": nn.LayerNorm,
            "activation": nn.GELU
        }
        self.fc_last = MLP(**config)
        # self.fc_last1 = MLPBlock(100,100,2,dropout=0.2)
        # self.fc_last2 = MLPBlock(100,100,2,dropout=0.2)
        # self.fc_last3 = nn.Linear(102,4)
        # self.fc_last = MLP()
        # state_dict = torch.load("/share/portal/jlr429/hal-skill-repo/point_and_shoot_debug/ckpts/20230905-180606_use_delta/epoch=900_mean_deltas=0.021.pt", map_location=torch.device(device))
        state_dict = torch.load("/share/portal/nlc62/hal-skill-repo/point_and_shoot_debug/ckpts/20230922-013559_2d/epoch=300_success=1.000.pt", map_location=torch.device(device))
        for key in list(state_dict.keys()):
                state_dict[key.replace('fc.', '')] = state_dict.pop(key)
        miss_keys, unexpected_keys = self.fc_last.load_state_dict(state_dict,strict=False)
        print(miss_keys)
        print(unexpected_keys)
        # loss/accuracy
        self.optimizer = optim.AdamW([
                {'params': self.conv_net.parameters()},
                {'params': self.conv_net2.parameters()},
                {'params': self.fc.parameters()}
            ], lr=lr)
        # self.optimizer = optim.AdamW(
        #        self.parameters()
        #     , lr=lr)

    def forward(self, wrist_img, head_img,curr_x,curr_y):##js_data,goal_pos,return_loss = True):
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
        # if self.use_end_eff:
        #     curr_x, curr_y = get_end_eff_yaw_ext(js_data)
        # else:
        #     curr_x = torch.zeros((batch_size, 1))
        #     curr_y = torch.zeros((batch_size, 1))
        img_t = img_t.to(device)
        img_t2 = img_t2.to(device)
        curr_x = curr_x.to(device).unsqueeze(1)
        curr_y = curr_y.to(device).unsqueeze(1)
        # info  = torch.cat((curr_x,curr_y),dim = 1)

        x = torch.cat((img_t, img_t2), dim=1)
        x = self.fc(x)
        # gt = torch.cat(((goal_pos[:,0].unsqueeze(1)-curr_x), (goal_pos[:,1].unsqueeze(1)-curr_y)), dim=1)

        # if return_loss:
        #     loss_goal = self.goal_loss(x,goal_pos)
        g_x,g_y = x[:,0].unsqueeze(1),x[:,1].unsqueeze(1)

        # inp_x = torch.cat((curr_x,curr_y,(g_x-curr_x),(g_y-curr_y)), dim=1).float()
        inp_x = torch.cat(((g_x-curr_x),(g_y-curr_y)), dim=1).float()
        x = self.fc_last(inp_x)
        # if return_loss:
        #     return x, loss_goal
        # else:
        return x, torch.cat((g_x,g_y),dim=1)

    def train_loop(self, train_dataloader,epoch,ctx, scaler):
        bad_yaws = 0
        losses, losses_ce, accuracy = [], [], []
        for batch in tqdm(
            train_dataloader,
            unit_scale=True,
            total=len(train_dataloader),
            position=1,
            desc="train",
            leave=False,
        ):
            self.batch_to_device(batch, self.device)
            wrist_img, head_img, joint_state, goal_pos, key_pressed = batch.values()
            idx = good_yaw_only(joint_state)
            # import pdb; pdb.set_trace()
            wrist_img, head_img, joint_state, goal_pos, key_pressed = wrist_img[idx], head_img[idx], joint_state[idx], goal_pos[idx], key_pressed[idx]
            if joint_state.size == 0:
                print("Unlikely event of all bad yaw")
                continue
            with ctx:
                predicted_action, loss  = self(wrist_img, head_img, joint_state,goal_pos)
                actual_action = key_pressed.to(torch.int64).squeeze()
                # loss_goal = self.goal_loss(predicted_delta, gt)
                loss_action = self.loss_fn(predicted_action,actual_action)
            # loss += loss_action
            self.optimizer.zero_grad()
            scaler.scale(loss).backward()
            # loss.backward()
            scaler.step(self.optimizer)
            # self.optimizer.step()
            scaler.update()    

            losses.append(loss.mean().item())
            losses_ce.append(loss_action.mean().item())
            acc = self.accuracy(predicted_action, actual_action)
            accuracy.append(acc.mean().item())
        lr = adjust_learning_rate(self.optimizer,epoch,self.lr,self.min_lr,self.max_epochs,self.warmup_epochs)
        log_dict = {
            "avg_train_goal_loss": np.mean(losses),
            "avg_train_action_loss": np.mean(losses_ce),
            "avg_train_acc": np.mean(accuracy),
            "lr": lr,
        }
        return log_dict

    @torch.no_grad()
    def evaluate_loop(self, val_dataloader,ctx):
        self.eval()
        losses, losses_ce, accuracy = [], [], []
        for batch in tqdm(
            val_dataloader,
            unit_scale=True,
            total=len(val_dataloader),
            position=1,
            desc="validation",
            leave=False,
        ):
            self.batch_to_device(batch, self.device)
            wrist_img, head_img, joint_state, goal_pos, key_pressed = batch.values()
            idx = good_yaw_only(joint_state)
            wrist_img, head_img, joint_state, goal_pos, key_pressed = wrist_img[idx], head_img[idx], joint_state[idx], goal_pos[idx], key_pressed[idx]
            if joint_state.size == 0:
                print("Unlikely event of all bad yaw")
                continue
            # predicted_action = self(wrist_img, joint_state, image2=head_img)
            with ctx:
                predicted_action,loss  = self(wrist_img, head_img, joint_state,goal_pos)
                actual_action = key_pressed.to(torch.int64).squeeze()
                # loss_goal = self.goal_loss(predicted_delta, gt)
                loss_action = self.loss_fn(predicted_action,actual_action)
            # loss += loss_action
            losses.append(loss.mean().item())
            losses_ce.append(loss_action.mean().item())
            acc = self.accuracy(predicted_action, actual_action)
            accuracy.append(acc.mean().item())
        log_dict = {
            "avg_val_goal_loss": np.mean(losses),
            "avg_val_action_loss": np.mean(losses_ce),
            "avg_val_acc": np.mean(accuracy),
        }
        return log_dict

    @torch.no_grad()
    def evaluate_total(self, val_dataloader,ctx):
        self.eval()
        preds = []
        actual = []
        for batch in tqdm(
            val_dataloader,
            unit_scale=True,
            total=len(val_dataloader),
            position=1,
            desc="validation",
            leave=False,
        ):
            self.batch_to_device(batch, self.device)
            wrist_img, head_img, joint_state, goal_pos, key_pressed = batch.values()
            idx = good_yaw_only(joint_state)
            wrist_img, head_img, joint_state, goal_pos, key_pressed = wrist_img[idx], head_img[idx], joint_state[idx], goal_pos[idx], key_pressed[idx]
            if joint_state.size == 0:
                print("Unlikely event of all bad yaw")
                continue
            # predicted_action = self(wrist_img, joint_state, image2=head_img)
            with ctx:
                predicted_action,loss = self(wrist_img, head_img, joint_state,goal_pos)
                predicted_action  = torch.argmax(predicted_action,dim = 1)
                actual_action = key_pressed.to(torch.int64).squeeze()
            preds.extend(list(predicted_action.cpu()))
            actual.extend(list(actual_action.cpu()))
        
        log_dict = {
            "Accuracy Val": accuracy_score(actual,preds),
            "Balanced Accuracy Val": balanced_accuracy_score(actual,preds),
        }
        preds = [x.item() for x in preds]
        actual = [x.item() for x in actual]
        distribution = dict(Counter(preds))
        total_kp = sum(list(distribution.values()))
        for key in distribution.keys():
            distribution[key] = distribution[key]/total_kp
        print(distribution)


        num_classes = 4

        # Calculate per-class accuracy
        class_correct = list(0. for i in range(num_classes))
        class_total = list(0. for i in range(num_classes))

        for i in range(len(actual)):
            label = actual[i]
            pred = preds[i]  
            if label == pred:
                class_correct[label] += 1
            class_total[label] += 1

        for i in range(num_classes):
            if class_total[i] == 0:
                acc = 0
            else:
                acc = 100 * class_correct[i] / class_total[i]
            # print(f'Accuracy of class {i}: {acc:.2f}%')
            log_dict[f"Accuracy of class {i}"] = acc

        return log_dict

    def batch_to_device(self, batch, device):
        for key, value in batch.items():
            batch[key] = value.to(device)

def good_yaw_only(joint_states):
    from dataset import joint_labels
    yaw = joint_states[:,joint_labels.index("joint_wrist_yaw_pos")]
    return yaw < 1.5
