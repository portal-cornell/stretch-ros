import sys
from pathlib import Path
from typing import Any, Mapping
from collections import Counter
import pdb

sys.path.insert(0, str(Path.cwd().parent.parent))

import numpy as np
from tqdm import tqdm
import torchvision
import torch
import torch.nn as nn
from torch import optim
import open_clip
from torchvision.models import resnet18, convnext_tiny
from torchvision.transforms.functional import crop
import torchmetrics
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import random
import bitsandbytes as bnb
from einops import rearrange
from einops.layers.torch import Rearrange
from torchvision.transforms import Resize

import torch.nn.functional as F


# from transformers import OwlViTForObjectDetection
from transformers import Owlv2ForObjectDetection

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


class MLPBlock(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        info_size=0,
        dropout=0.5,
        norm=nn.LayerNorm,
        activation=nn.GELU,
    ):
        super().__init__()
        self.layers = []
        self.layers.append(nn.Linear(input_size + info_size, output_size))
        self.layers.append(norm(output_size))
        self.layers.append(nn.Dropout(dropout))
        self.layers.append(activation())
        self.model = nn.Sequential(*self.layers)
        self.use_residual = input_size == output_size

    def forward(self, x, info=None):
        if info is not None:
            x_new = torch.cat((x, info), dim=1)
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
    def __init__(
        self,
        input_size,
        output_size,
        hidden_dim,
        num_layers,
        info_size=0,
        dropout=0.5,
        norm=nn.LayerNorm,
        activation=nn.GELU,
    ):
        super().__init__()
        assert num_layers >= 2
        self.layers = []

        self.pre_fc = MLPBlock(
            input_size,
            hidden_dim,
            info_size=0,
            dropout=dropout,
            norm=norm,
            activation=activation,
        )
        for i in range(num_layers - 2):
            self.layers.append(
                MLPBlock(
                    hidden_dim,
                    hidden_dim,
                    info_size,
                    dropout=dropout,
                    norm=norm,
                    activation=activation,
                )
            )
        self.layers = nn.ModuleList(self.layers)

        self.fc_last = nn.Linear(hidden_dim + info_size, output_size)

    def forward(self, x, info):
        x = self.pre_fc(x)
        for layer in self.layers:
            x = layer(x, info)
        x = torch.cat((x, info), dim=1)
        return self.fc_last(x)


class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_dim,
        num_layers,
        info_size=0,
        dropout=0.5,
        norm=nn.LayerNorm,
        activation=nn.GELU,
    ):
        super().__init__()
        assert num_layers >= 2
        self.layers = []
        self.layers.append(nn.Linear(input_size, hidden_dim))
        self.layers.append(norm(hidden_dim))
        self.layers.append(nn.Dropout(dropout))
        self.layers.append(activation())

        for i in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(norm(hidden_dim))
            self.layers.append(nn.Dropout(dropout))
            self.layers.append(activation())
        self.layers.append(nn.Linear(hidden_dim, output_size))
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)


import math


def adjust_learning_rate(optimizer, epoch, lr, min_lr, num_epochs, warmup_epochs):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < warmup_epochs:
        lr = lr * epoch / warmup_epochs
    else:
        lr = min_lr + (lr - min_lr) * 0.5 * (
            1.0
            + math.cos(math.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs))
        )
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


class BC(nn.Module):
    def __init__(
        self,
        num_classes,
        lr=1e-3,
        max_epochs=1e5,
        device="cpu",
    ):
        super().__init__()
        self.num_classes = num_classes
        # TODO: calculate new weights on our data
        weights = torch.FloatTensor([0.43317524, 0.0, 3.39238095, 2.52087757])
        self.loss_fn = nn.CrossEntropyLoss(weight=weights)
        self.goal_loss = nn.SmoothL1Loss(reduction="none")
        self.goal_weights = torch.tensor([1, 1]).to("cpu")
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_classes, top_k=1
        )
        self.max_epochs = max_epochs
        self.warmup_epochs = max_epochs // 10
        # metadata
        self.device = device
        self.lr = lr
        self.min_lr = lr / 1e3

        # network
        self.end_eff_dims = 2
        self.model, _, _ = open_clip.create_model_and_transforms(
            "EVA02-B-16", pretrained="merged2b_s8b_b131k"
        )
        # self.object_model = OwlViTForObjectDetection.from_pretrained(
        #     "google/owlvit-base-patch32"
        # )
        self.object_model = Owlv2ForObjectDetection.from_pretrained(
            "google/owlv2-base-patch16-ensemble"
        )

        self.fc = nn.Sequential(
            MLPBlock(512, 512, dropout=0.1),
            MLPBlock(512, 512, dropout=0.1),
            MLPBlock(512, 512, dropout=0.1),
            nn.Linear(512, 2),
        )

        config = {
            "input_size": 2,
            "output_size": num_classes,
            "hidden_dim": 100,
            "num_layers": 3,
            "dropout": 0.5,
            "norm": nn.LayerNorm,
            "activation": nn.GELU,
        }
        self.fc_last = MLP(**config)

        # loss/accuracy
        self.box_linear = nn.Sequential(
            MLPBlock(4, 512, dropout=0.1),
            MLPBlock(512, 512, dropout=0.1),
            MLPBlock(512, 512, dropout=0.1),
            nn.Linear(512, 2),
        )
        self.optimizer = bnb.optim.AdamW8bit(
            [{"params": self.box_linear.parameters()}],
            lr=lr,
        )
        # self.optimizer = optim.AdamW(
        #        self.parameters()
        #     , lr=lr)

    def get_top_k_boxes(self, predicted_boxes, scores, k):
        # Sort the scores tensor in descending order along the num_boxes dimension
        sorted_scores = torch.argsort(scores, dim=1, descending=True)

        # Take the top 4 indices from each batch
        top_k_indices = sorted_scores[:, :k]
        top_k_boxes = torch.gather(
            predicted_boxes,
            1,
            top_k_indices.unsqueeze(-1).expand(-1, -1, predicted_boxes.size(-1)),
        )
        return top_k_boxes

    def posemb_sincos_2d(
        self, h, w, dim, temperature: int = 10000, dtype=torch.float32
    ):
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
        omega = torch.arange(dim // 4) / (dim // 4 - 1)
        omega = 1.0 / (temperature**omega)

        y = y.flatten()[:, None] * omega[None, :]
        x = x.flatten()[:, None] * omega[None, :]
        pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
        return pe.type(dtype)

    def forward(self, inputs, head_img, ref_text, js_data, goal_pos, return_loss=True):
        k = 4
        with torch.no_grad():
            outputs = self.object_model(**inputs)
            logits = torch.max(outputs["logits"], dim=-1)
            scores = torch.sigmoid(logits.values)
            boxes = self.get_top_k_boxes(outputs["pred_boxes"], scores, k)
            cx, cy, w, h = (
                boxes[:, :, 0],
                boxes[:, :, 1],
                boxes[:, :, 2],
                boxes[:, :, 3],
            )
            x = (
                (cx * head_img.shape[2] - w * head_img.shape[2] / 2)
                .to(torch.int)
                .squeeze(0)
                .cpu()
                .detach()
                .tolist()
            )
            y = (
                (cy * head_img.shape[3] - h * head_img.shape[3] / 2)
                .to(torch.int)
                .squeeze(0)
                .cpu()
                .detach()
                .tolist()
            )
            w = (w * head_img.shape[2]).to(torch.int).squeeze(0).cpu().detach().tolist()
            h = (h * head_img.shape[3]).to(torch.int).squeeze(0).cpu().detach().tolist()

            cropped_imgs = []
            for i in range(k):
                cropped_img = crop(head_img, y[i], x[i], h[i], w[i])
                cropped_imgs.append(Resize((224, 224))(cropped_img))
            cropped_imgs = torch.stack(cropped_imgs).squeeze(1)
            img_embeddings = self.model.visual(cropped_imgs)
            text_embeddings = self.model.encode_text(ref_text)
            img_embeddings /= img_embeddings.norm(dim=-1, keepdim=True)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            dot_product = img_embeddings @ text_embeddings.T
            dot_soft = F.softmax(dot_product, dim=0)
            amax = int(torch.argmax(dot_soft).item())

        best_box = boxes[:, amax]
        x = self.box_linear(best_box)

        if return_loss:
            loss_goal = self.goal_loss(x, goal_pos)
            loss_goal = loss_goal * self.goal_weights
            loss_goal = torch.mean(loss_goal)
        g_x, g_y = x[:, 0].unsqueeze(1), x[:, 1].unsqueeze(1)

        # inp_x = torch.cat((curr_x,curr_y,(g_x-curr_x),(g_y-curr_y)), dim=1).float()
        # curr_x, curr_y = get_end_eff_yaw_ext(js_data)

        curr_x = curr_x.to(self.device).unsqueeze(1)
        curr_y = curr_y.to(self.device).unsqueeze(1)
        inp_x = torch.cat(((g_x - curr_x), (g_y - curr_y)), dim=1).float()
        x = self.fc_last(inp_x)
        if return_loss:
            return x, loss_goal
        else:
            return x  # torch.cat((g_x,g_y),dim=1)

    def forward_special(self, inputs, head_img, ref_text, js_data):
        k = 4
        with torch.no_grad():
            outputs = self.object_model(**inputs)
            logits = torch.max(outputs["logits"], dim=-1)
            scores = torch.sigmoid(logits.values)
            boxes = self.get_top_k_boxes(outputs["pred_boxes"], scores, k)
            cx, cy, w, h = (
                boxes[:, :, 0],
                boxes[:, :, 1],
                boxes[:, :, 2],
                boxes[:, :, 3],
            )
            x = (
                (cx * head_img.shape[2] - w * head_img.shape[2] / 2)
                .to(torch.int)
                .squeeze(0)
                .cpu()
                .detach()
                .tolist()
            )
            y = (
                (cy * head_img.shape[3] - h * head_img.shape[3] / 2)
                .to(torch.int)
                .squeeze(0)
                .cpu()
                .detach()
                .tolist()
            )
            w = (w * head_img.shape[2]).to(torch.int).squeeze(0).cpu().detach().tolist()
            h = (h * head_img.shape[3]).to(torch.int).squeeze(0).cpu().detach().tolist()

            cropped_imgs = []
            for i in range(k):
                cropped_img = crop(head_img, y[i], x[i], h[i], w[i])
                cropped_imgs.append(Resize((224, 224))(cropped_img))
            cropped_imgs = torch.stack(cropped_imgs).squeeze(1)
            img_embeddings = self.model.visual(cropped_imgs)
            text_embeddings = self.model.encode_text(ref_text)
            img_embeddings /= img_embeddings.norm(dim=-1, keepdim=True)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            dot_product = img_embeddings @ text_embeddings.T
            dot_soft = F.softmax(dot_product, dim=0)
            amax = int(torch.argmax(dot_soft).item())

        best_box = boxes[:, amax]
        x = self.box_linear(best_box)
        return x, best_box

    def train_loop(self, train_dataloader, epoch, ctx, scaler):
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
            (
                inputs,
                head_img,
                ref_text_tokenized,
                joint_state,
                goal_pos,
                key_pressed,
            ) = batch.values()
            # idx = good_yaw_only(joint_state)
            # wrist_img, head_img, joint_state, ref_text_tokenized, goal_pos, key_pressed = wrist_img[idx], head_img[idx], joint_state[idx], ref_text_tokenized[idx],  goal_pos[idx], key_pressed[idx]
            if joint_state.size == 0:
                print("Unlikely event of all bad yaw")
                continue
            with ctx:
                predicted_action, loss = self(
                    inputs, head_img, ref_text_tokenized, joint_state, goal_pos
                )
                actual_action = key_pressed.to(torch.int64)
                # loss_goal = self.goal_loss(predicted_delta, gt)
                # pdb.set_trace()
                loss_action = self.loss_fn(predicted_action, actual_action)
            # loss += loss_action
            self.optimizer.zero_grad()
            scaler.scale(loss).backward()
            # loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            scaler.step(self.optimizer)
            # self.optimizer.step()
            scaler.update()

            losses.append(loss.mean().item())
            losses_ce.append(loss_action.mean().item())
            acc = self.accuracy(predicted_action, actual_action)
            accuracy.append(acc.mean().item())
        lr = adjust_learning_rate(
            self.optimizer,
            epoch,
            self.lr,
            self.min_lr,
            self.max_epochs,
            self.warmup_epochs,
        )
        log_dict = {
            "avg_train_goal_loss": np.mean(losses),
            "avg_train_action_loss": np.mean(losses_ce),
            "avg_train_acc": np.mean(accuracy),
            "lr": lr,
        }
        return log_dict

    @torch.no_grad()
    def evaluate_loop(self, val_dataloader, ctx):
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
            (
                inputs,
                head_img,
                ref_text_tokenized,
                joint_state,
                goal_pos,
                key_pressed,
            ) = batch.values()
            # idx = good_yaw_only(joint_state)
            # wrist_img, head_img, joint_state, ref_text_tokenized, goal_pos, key_pressed = wrist_img[idx], head_img[idx], joint_state[idx], ref_text_tokenized[idx], goal_pos[idx], key_pressed[idx]
            if joint_state.size == 0:
                print("Unlikely event of all bad yaw")
                continue
            # predicted_action = self(wrist_img, joint_state, image2=head_img)
            with ctx:
                predicted_action, loss = self(
                    inputs, head_img, ref_text_tokenized, joint_state, goal_pos
                )
                actual_action = key_pressed.to(torch.int64)
                # loss_goal = self.goal_loss(predicted_delta, gt)
                loss_action = self.loss_fn(predicted_action, actual_action)
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
    def evaluate_total(self, val_dataloader, ctx):
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
            (
                wrist_img,
                head_img,
                ref_text_tokenized,
                joint_state,
                goal_pos,
                key_pressed,
            ) = batch.values()
            # idx = good_yaw_only(joint_state)
            # wrist_img, head_img, joint_state, ref_text_tokenized, goal_pos, key_pressed = wrist_img[idx], head_img[idx], joint_state[idx], ref_text_tokenized[idx], goal_pos[idx], key_pressed[idx]
            if joint_state.size == 0:
                print("Unlikely event of all bad yaw")
                continue
            # predicted_action = self(wrist_img, joint_state, image2=head_img)
            with ctx:
                predicted_action, _ = self(
                    wrist_img, head_img, ref_text_tokenized, joint_state, goal_pos
                )
                predicted_action = torch.argmax(predicted_action, dim=1)
                actual_action = key_pressed.to(torch.int64).squeeze()
            preds.extend(list(predicted_action.cpu()))
            actual.extend(list(actual_action.cpu()))

        log_dict = {
            "Accuracy Val": accuracy_score(actual, preds),
            "Balanced Accuracy Val": balanced_accuracy_score(actual, preds),
        }
        preds = [x.item() for x in preds]
        actual = [x.item() for x in actual]
        distribution = dict(Counter(preds))
        total_kp = sum(list(distribution.values()))
        for key in distribution.keys():
            distribution[key] = distribution[key] / total_kp
        print(distribution)

        num_classes = 4

        # Calculate per-class accuracy
        class_correct = list(0.0 for i in range(num_classes))
        class_total = list(0.0 for i in range(num_classes))

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

    yaw = joint_states[:, joint_labels.index("joint_wrist_yaw_pos")]
    return yaw < 1.5
