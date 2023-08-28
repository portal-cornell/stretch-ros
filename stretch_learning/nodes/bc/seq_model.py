import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim


class BC_Seq(nn.Module):
    def __init__(
        self,
        action_horizon,
        loss_fn,
        accuracy,
        lr=1e-5,
        max_epochs=1e5,
        device="cuda",
    ):
        super().__init__()

        # metadata
        joint_state_dim = 3
        final_coord_dim = 3
        self.device = device
        self.fc_input_dims = joint_state_dim + final_coord_dim
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dims, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(256, action_horizon * joint_state_dim),
        )

        # loss/accuracy
        self.loss_fn = loss_fn
        self.accuracy = accuracy
        self.optimizer = optim.AdamW(self.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, max_epochs, 0
        )

    def forward(self, x):
        return self.fc(x)

    def train_loop(self, train_dataloader):
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
            start_js, actual_deltas = batch.values()
            predicted_deltas = self(start_js)
            predicted_deltas, actual_deltas = self.reshape_then_sum(
                predicted_deltas, actual_deltas
            )

            self.optimizer.zero_grad()
            loss = self.loss_fn(predicted_deltas, actual_deltas)
            loss.backward()
            self.optimizer.step()

            losses.append(loss.mean().item())
            acc = self.accuracy(predicted_deltas, actual_deltas)
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
        self.eval()
        for batch in tqdm(
            val_dataloader,
            unit_scale=True,
            total=len(val_dataloader),
            position=1,
            desc="validation",
            leave=False,
        ):
            self.batch_to_device(batch, self.device)
            start_js, actual_deltas = batch.values()
            predicted_deltas = self(start_js)
            predicted_deltas, actual_deltas = self.reshape_then_sum(
                predicted_deltas, actual_deltas
            )

            loss = self.loss_fn(predicted_deltas, actual_deltas)
            losses.append(loss.mean().item())
            acc = self.accuracy(predicted_deltas, actual_deltas)
            accuracy.append(acc.mean().item())
        self.train()
        log_dict = {
            "avg_val_loss": np.mean(losses),
            "avg_val_acc": np.mean(accuracy),
        }
        return log_dict

    def batch_to_device(self, batch, device):
        for key, value in batch.items():
            batch[key] = value.to(device)

    def reshape_then_sum(self, predicted_deltas, actual_deltas):
        predicted_deltas = predicted_deltas.view(*actual_deltas.size())
        # predicted_deltas = torch.cumsum(predicted_deltas, dim=-2)
        # actual_deltas = torch.cumsum(actual_deltas, dim=-2)
        return predicted_deltas, actual_deltas