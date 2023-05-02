import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent.parent))

from typing import Callable

import uuid

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from tqdm import tqdm

# print utils
HYPHENS_80 = "-" * 80

# dim constants
END_EFF_DIMS = 3
JOINT_STATE_DIMS = 14 * 3
KEYBOARD_ACTIONS = 17
RESNET_IMG_DIM = (3, 224, 224)

# train type
IQL = "iql"
CQL = "cql"
AWC = "awc"
TD3 = "td3"


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


@torch.no_grad()
def eval_actor(
    actor,
    img_js_net,
    val_dataloader,
    batch_size,
    device,
    train_type,
    train_id=None,
):
    accuracy = torchmetrics.Accuracy(
        task="multiclass", num_classes=KEYBOARD_ACTIONS, top_k=1
    ).to(device)

    acc_accum = []
    loss_accum = []
    for i, batch in tqdm(
        enumerate(val_dataloader), unit_scale=True, maxinterval=len(val_dataloader)
    ):
        images = batch["images"]
        joint_states = batch["joint_states"]
        actual_actions = batch["actions"]

        images, joint_states = images.to(device), joint_states.to(device)
        observations = img_js_net(images, joint_states)
        if train_type == IQL:
            predicted_actions = (
                actor.act(observations, deterministic=True).detach().cpu()
            )
        elif train_type == TD3 or train_type == CQL:
            predicted_actions = actor.act(observations).detach().cpu()

        loss = torch.mean(
            F.cross_entropy(predicted_actions, actual_actions.squeeze(1))
        ).item()
        loss_accum.append(loss)
        acc = accuracy(predicted_actions, actual_actions.squeeze(1)).item()
        acc_accum.append(acc)

        if train_id and i == len(val_dataloader) - 1:
            pac = torch.cat((predicted_actions, actual_actions), axis=1)
            pac = torch.cat(
                (pac, torch.argmax(predicted_actions, dim=1).unsqueeze(1)), axis=1
            )
            pac_np = pac.numpy()
            df = pd.DataFrame(pac_np)
            df.to_csv(
                f"validation_debug/{train_id}_predicted_v_actual.csv", index=False
            )

    return np.mean(acc_accum), np.mean(loss_accum)


class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


class MLP(nn.Module):
    def __init__(
        self,
        dims,
        activation_fn: Callable[[], nn.Module] = nn.ReLU,
        output_activation_fn: Callable[[], nn.Module] = None,
        squeeze_output: bool = False,
    ):
        super().__init__()
        n_dims = len(dims)
        if n_dims < 2:
            raise ValueError("MLP requires at least two dims (input and output)")

        layers = []
        for i in range(n_dims - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(activation_fn())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
