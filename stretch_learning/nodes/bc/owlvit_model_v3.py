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
import open_clip
from torchvision.models import resnet18, convnext_tiny
import torchmetrics

# from utils.common import get_end_eff, get_end_eff_yaw_ext
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import random
import bitsandbytes as bnb
from einops import rearrange
from einops.layers.torch import Rearrange

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
from transformers import OwlViTForObjectDetection


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
        device="cuda",
    ):
        super().__init__()
        self.num_classes = num_classes
        # TODO: calculate new weights on our data
        weights = torch.FloatTensor([0.43317524, 0.0, 3.39238095, 2.52087757])
        self.loss_fn = nn.CrossEntropyLoss(weight=weights)
        self.goal_loss = nn.SmoothL1Loss(reduction="none")
        self.goal_weights = torch.tensor([2, 0.5]).to("cpu")
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
        self.model.eval()
        self.object_model = OwlViTForObjectDetection.from_pretrained(
            "google/owlvit-base-patch32"
        )
        self.object_model.eval()
        self.box_embed = nn.Linear(4, 512)
        # encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8,batch_first= True, activation="gelu")
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=512, nhead=8, batch_first=True, activation="gelu", dropout=0
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=8)
        patch_height = 16
        patch_width = 16
        patch_dim = 768
        dim = 512
        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        # self.conv_net = convnext_tiny(torchvision.models.convnext.ConvNeXt_Tiny_Weights.DEFAULT)
        # self.conv_net.fc = nn.Identity()
        # self.conv_net2 = convnext_tiny(torchvision.models.convnext.ConvNeXt_Tiny_Weights.DEFAULT)
        # self.conv_net2.fc = nn.Identity()

        # self.fc = MLPAdvanced(512,2,512,4,info_size=14,dropout=0.1)
        self.fc = nn.Sequential(
            MLPBlock(512, 512, dropout=0),
            MLPBlock(512, 512, dropout=0),
            MLPBlock(512, 512, dropout=0),
            nn.Linear(512, 2),
        )
        self.tokenizer = open_clip.get_tokenizer("EVA02-B-16")
        self.pos_embedding = self.posemb_sincos_2d(
            h=224 // patch_height,
            w=224 // patch_width,
            dim=dim,
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
        # state_dict = torch.load(
        #     "/share/portal/nlc62/hal-skill-repo/point_and_shoot_debug/ckpts/20230930-205318_2d/epoch=30_success=1.000.pt",
        #     map_location=torch.device(device),
        # )
        # for key in list(state_dict.keys()):
        #     state_dict[key.replace("fc.", "")] = state_dict.pop(key)
        # miss_keys, unexpected_keys = self.fc_last.load_state_dict(
        #     state_dict, strict=False
        # )
        # print(miss_keys)
        # print(unexpected_keys)
        # loss/accuracy
        self.optimizer = bnb.optim.AdamW8bit(
            [
                # {'params': self.conv_net.parameters()},
                # {'params': self.model.text.parameters()},
                {"params": self.fc.parameters()},
                # {'params': self.transformer_encoder.parameters()},
                {"params": self.transformer_decoder.parameters()},
                {"params": self.to_patch_embedding.parameters()},
                {"params": self.box_embed.parameters()},
            ],
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

    def forward(
        self,
        inputs_head,
        inputs_wrist,
        wrist_img,
        head_img,
        ref_text,
        js_data,
        goal_pos,
        return_loss=True,
    ):
        with torch.no_grad():
            outputs_head = self.object_model(**inputs_head)
            logits_head = torch.max(outputs_head["logits"], dim=-1)
            scores_head = torch.sigmoid(logits_head.values)
            top_boxes_head = self.get_top_k_boxes(
                outputs_head["pred_boxes"], scores_head, 4
            )
            print(f"{top_boxes_head=}")
            outputs_wrist = self.object_model(**inputs_wrist)
            logits_wrist = torch.max(outputs_wrist["logits"], dim=-1)
            scores_wrist = torch.sigmoid(logits_wrist.values)
            top_boxes_wrist = self.get_top_k_boxes(
                outputs_wrist["pred_boxes"], scores_wrist, 4
            )
            print(f"{top_boxes_wrist=}")
            text_features = self.model.encode_text(ref_text).unsqueeze(1)
            text_features = torch.zeros_like(text_features)
        img_tokens_wrist = self.to_patch_embedding(wrist_img)
        img_tokens_wrist += self.pos_embedding.to(
            self.device, dtype=img_tokens_wrist.dtype
        )
        img_tokens_head = self.to_patch_embedding(head_img)
        img_tokens_head += self.pos_embedding.to(
            self.device, dtype=img_tokens_head.dtype
        )
        img_tokens = torch.cat((img_tokens_head, img_tokens_wrist), dim=1)
        # img_tokens = self.transformer_encoder(img_tokens)
        box_tokens_head = self.box_embed(top_boxes_head)
        box_tokens_wrist = self.box_embed(top_boxes_wrist)
        input_tokens = torch.cat(
            (box_tokens_head, box_tokens_wrist, text_features), dim=1
        )

        curr_x, curr_y = get_end_eff_yaw_ext(js_data)

        curr_x = curr_x.to(self.device).unsqueeze(1)
        curr_y = curr_y.to(self.device).unsqueeze(1)
        # info = js_data.to(self.device)

        # x = torch.cat((image_features_head, img_features_wrist), dim=1)
        x = self.transformer_decoder(input_tokens, img_tokens)
        x = torch.mean(x, dim=1)
        x = self.fc(x)
        # gt = torch.cat(((goal_pos[:,0].unsqueeze(1)-curr_x), (goal_pos[:,1].unsqueeze(1)-curr_y)), dim=1)

        if return_loss:
            loss_goal = self.goal_loss(x, goal_pos)
            loss_goal = loss_goal * self.goal_weights
            loss_goal = torch.mean(loss_goal)
        # g_x,g_y = x[:,0].unsqueeze(1),x[:,1].unsqueeze(1)

        # inp_x = torch.cat((curr_x,curr_y,(g_x-curr_x),(g_y-curr_y)), dim=1).float()
        # inp_x = torch.cat(((g_x-curr_x),(g_y-curr_y)), dim=1).float()
        x = self.fc_last(x)
        if return_loss:
            return x, loss_goal
        else:
            return x  # torch.cat((g_x,g_y),dim=1)

    def forward_special(
        self, inputs_head, inputs_wrist, wrist_img, head_img, ref_text, js_data
    ):
        with torch.no_grad():
            outputs_head = self.object_model(**inputs_head)
            outputs_head["logits"] = torch.zeros_like(outputs_head["logits"])
            logits_head = torch.max(outputs_head["logits"], dim=-1)
            scores_head = torch.sigmoid(logits_head.values)
            top_boxes_head = self.get_top_k_boxes(
                outputs_head["pred_boxes"], scores_head, 4
            )
            outputs_wrist = self.object_model(**inputs_wrist)
            outputs_wrist["logits"] = torch.zeros_like(outputs_wrist["logits"])
            logits_wrist = torch.max(outputs_wrist["logits"], dim=-1)
            scores_wrist = torch.sigmoid(logits_wrist.values)
            top_boxes_wrist = self.get_top_k_boxes(
                outputs_wrist["pred_boxes"], scores_wrist, 4
            )

            text_features = self.model.encode_text(ref_text).unsqueeze(1)
            text_features = torch.zeros_like(text_features)
        img_tokens_wrist = self.to_patch_embedding(wrist_img)
        # img_tokens_wrist += self.pos_embedding.to(
        #     self.device, dtype=img_tokens_wrist.dtype
        # )
        img_tokens_head = self.to_patch_embedding(head_img)
        # img_tokens_head += self.pos_embedding.to(
        #     self.device, dtype=img_tokens_head.dtype
        # )
        img_tokens = torch.cat((img_tokens_head, img_tokens_wrist), dim=1)
        box_tokens_head = self.box_embed(top_boxes_head)
        box_tokens_wrist = self.box_embed(top_boxes_wrist)
        input_tokens = torch.cat(
            (box_tokens_head, box_tokens_wrist, text_features), dim=1
        )
        # input_tokens = torch.zeros_like(input_tokens)
        # img_tokens = torch.zeros_like(img_tokens)

        curr_x, curr_y = get_end_eff_yaw_ext(js_data)

        curr_x = curr_x.to(self.device).unsqueeze(1)
        curr_y = curr_y.to(self.device).unsqueeze(1)
        info = js_data.to(self.device)

        x = self.transformer_decoder(input_tokens, img_tokens)
        x = torch.mean(x, dim=1)
        x = self.fc(x)
        g_x, g_y = x[:, 0].unsqueeze(1), x[:, 1].unsqueeze(1)
        x = self.fc_last(x)
        return x, torch.cat((g_x, g_y), dim=1)

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
                inputs_head,
                inputs_wrist,
                wrist_img,
                head_img,
                ref_text,
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
                    inputs_head,
                    inputs_wrist,
                    wrist_img,
                    head_img,
                    ref_text,
                    joint_state,
                    goal_pos,
                )
                actual_action = key_pressed.to(torch.int64).squeeze()
                # loss_goal = self.goal_loss(predicted_delta, gt)
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
                inputs_head,
                inputs_wrist,
                wrist_img,
                head_img,
                ref_text,
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
                    inputs_head,
                    inputs_wrist,
                    wrist_img,
                    head_img,
                    ref_text,
                    joint_state,
                    goal_pos,
                )
                actual_action = key_pressed.to(torch.int64).squeeze()
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


base_gripper_yaw = -0.09
gripper_len = 0.22


def get_end_eff_yaw_ext(js):
    is_single_dim = len(js.shape) == 1
    if is_single_dim:
        js = js.unsqueeze(0)
    yaw, ext = js[:, -1], js[:, -2]
    yaw_delta = -(yaw - base_gripper_yaw)  # going right is more negative
    yaw_delta = yaw_delta
    y = gripper_len * torch.cos(yaw_delta) + ext
    x = gripper_len * torch.sin(yaw_delta)
    return x, y
