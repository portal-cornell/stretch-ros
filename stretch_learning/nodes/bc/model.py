import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torchmetrics
from torch import optim


class BC(nn.Module):
    def __init__(
        self,
        lr=1e-3,
        max_epochs=1e5,
        device="cuda",
    ):
        super().__init__()

        # metadata
        self.device = device
        self.fc_input_dims = 3 + 3  # joint_states_pos + final_coord
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dims, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(256, 17),
        )

        # loss/accuracy
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=17, top_k=1
        )
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
            inp, key_pressed = batch.values()
            predicted_action = self(inp)
            actual_action = key_pressed.to(torch.int64).squeeze()

            self.optimizer.zero_grad()
            loss = self.loss_fn(predicted_action, actual_action)
            loss.backward()
            self.optimizer.step()

            losses.append(loss.mean().item())
            acc = self.accuracy(predicted_action, actual_action)
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
            inp, key_pressed = batch.values()
            predicted_action = self(inp)
            actual_action = key_pressed.to(torch.int64).squeeze()

            loss = self.loss_fn(predicted_action, actual_action)
            losses.append(loss.mean().item())
            acc = self.accuracy(predicted_action, actual_action)
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