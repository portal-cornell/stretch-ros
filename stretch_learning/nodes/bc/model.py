import math
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torch import optim


# from dataset import gripper_len, base_gripper_yaw
gripper_len = 0.22
base_gripper_yaw = -0.09

def convert_js_xy(extension, yaw):
    yaw_delta = -(yaw - base_gripper_yaw)  # going right is more negative
    yaw_delta = yaw_delta.cpu()
    y = gripper_len * np.cos(yaw_delta) + extension
    x = gripper_len * np.sin(yaw_delta)
    return x, y


class BC(nn.Module):
    def __init__(
        self,
        is_2d,
        use_delta,
        lr=1e-3,
        max_epochs=1e3,
        device="cuda",
    ):
        super().__init__()

        # metadata
        self.device = device
        self.num_classes = 4
        self.fc_input_dims = 2 if is_2d else 2 + 2  # joint_states_pos + final_coord

        hidden_dim = 100
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

        self.is_2d = is_2d
        self.use_delta = use_delta

        # loss/accuracy
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_classes, top_k=1
        )
        self.optimizer = optim.AdamW(self.parameters(), lr=lr)

        self.max_epochs = max_epochs
        self.warmup_epochs = max_epochs // 10
        self.lr = lr
        self.min_lr = lr / 1e3

        self.kp_delta_mapping = {
            # arm out
            0: [0.04, 0],
            # arm in
            1: [-0.04, 0],
            # gripper right
            2: [0, 0.010472],
            # gripper left
            3: [0, -0.010472],
        }

        print(self.fc)

    def forward(self, x):
        return self.fc(x)

    def train_loop(self, train_dataloader, epoch):
        losses, accuracy = [], []
        avg_prediction_magnitude = []
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

            prediction_magnitude = torch.norm(predicted_action)
            avg_prediction_magnitude.append(prediction_magnitude.item())

            self.optimizer.zero_grad()
            loss = self.loss_fn(predicted_action, actual_action)
            loss.backward()
            self.optimizer.step()

            losses.append(loss.mean().item())
            acc = self.accuracy(predicted_action, actual_action)
            accuracy.append(acc.mean().item())

        log_dict = {
            "avg_train_loss": np.mean(losses),
            "avg_train_acc": np.mean(accuracy),
            "avg_train_norm": np.mean(avg_prediction_magnitude),
            "lr": self.optimizer.param_groups[0]["lr"],
        }
        return log_dict

    def eval_on_simulator(self, test_points):
        self.eval()
        deltas = []
        for test_point in tqdm(
            test_points, desc="Eval on sim", total=len(test_points), leave=False
        ):
            with torch.no_grad():
                test_point = (
                    torch.from_numpy(np.array(test_point)).unsqueeze(0).to(self.device)
                )
                delta_min = self.simulate(test_point, 500)
                deltas.append(delta_min)
        self.train()
        log_dict = {
            "val_deltas_max": max(deltas),
            "val_deltas_min": min(deltas),
            "val_deltas_mean": np.mean(deltas),
        }
        return log_dict

    def simulate(self, inp, iterations):
        """
        inp: [wrist extension, joint wrist yaw, goal wrist extension, goal joint wrist yaw] torch tensor
        model: takes inp and outputs [delta wrist extension, delta wrist yaw]
        wrist extension: lower (0.0025) upper (0.457)
        wrist yaw: (-1.3837785024051723, 4.585963397116449)
        """
        delta_list = []
        goal_ext = inp[0, 2] + inp[0, 0]
        goal_yaw = inp[0, 3] + inp[0, 1]
        curr_x, curr_y = convert_js_xy(inp[0, 0], inp[0, 1])
        goal_x, goal_y = convert_js_xy(goal_ext, goal_yaw)
        for _ in range(iterations):
            if not self.is_2d:
                temp_inp = torch.tensor(
                    [[curr_x, curr_y, (goal_x - curr_x), (goal_y - curr_y)]],
                    device=self.device,
                )
            else:
                temp_inp = inp
            prediction = self(temp_inp)
            prediction = prediction.flatten()

            predicted_kp = torch.argmax(prediction).item()
            deltas = torch.Tensor(self.kp_delta_mapping[predicted_kp]).to(self.device)
            if not self.is_2d:
                inp[0, :2] += deltas
                inp[0, 2:] -= deltas
                curr_x, curr_y = convert_js_xy(inp[0, 0], inp[0, 1])
                delta_list.append(torch.norm(inp[0, 2:]).item())
            else:
                inp[0, :] -= deltas
                delta_list.append(torch.norm(inp[0, :]).item())

        return min(delta_list)

    @torch.no_grad()
    def evaluate_loop(self, val_dataloader):
        losses, accuracy = [], []
        avg_prediction_magnitude = []
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

            avg_prediction_magnitude.append(torch.norm(predicted_action).item())
            loss = self.loss_fn(predicted_action, actual_action) + torch.norm(
                predicted_action
            )
            losses.append(loss.mean().item())
            acc = self.accuracy(predicted_action, actual_action)
            accuracy.append(acc.mean().item())
        self.train()
        log_dict = {
            "avg_val_loss": np.mean(losses),
            "avg_val_acc": np.mean(accuracy),
            "avg_val_norm": np.mean(avg_prediction_magnitude),
        }
        return log_dict

    def batch_to_device(self, batch, device):
        for key, value in batch.items():
            batch[key] = value.to(device)