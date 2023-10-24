import math
import numpy as np
from tqdm import tqdm
import random

from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torch import optim
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# from dataset import gripper_len, base_gripper_yaw

gripper_len = 0.22
base_gripper_yaw = -0.09


def convert_js_xy(extension, yaw):
    yaw_delta = -(yaw - base_gripper_yaw)  # going right is more negative
    yaw_delta = yaw_delta.cpu()
    y = gripper_len * np.cos(yaw_delta) + extension
    x = gripper_len * np.sin(yaw_delta)
    return x, y
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

class MLP(nn.Module):
    def __init__(self,input_size,output_size,hidden_dim, num_layers, dropout = 0.5, norm = nn.LayerNorm, activation= nn.GELU):
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

        # self.fc = nn.Sequential(
        #     nn.Linear(self.fc_input_dims, 256),
        #     nn.BatchNorm1d(256),
        #     nn.Dropout(p=0.5),
        #     nn.ReLU(),
        #     nn.Linear(256,256),
        #     nn.BatchNorm1d(256),
        #     nn.Dropout(p=0.5),
        #     nn.ReLU(),
        #     nn.Linear(256, self.num_classes),
        # )

    def forward(self,x):
        return self.model(x)

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

        # self.fc = nn.Sequential(
        #     nn.Linear(self.fc_input_dims, 256),
        #     nn.BatchNorm1d(256),
        #     nn.Dropout(p=0.5),
        #     nn.ReLU(),
        #     nn.Linear(256,256),
        #     nn.BatchNorm1d(256),
        #     nn.Dropout(p=0.5),
        #     nn.ReLU(),
        #     nn.Linear(256, self.num_classes),
        # )

        config = {
            "input_size": self.fc_input_dims,
            "output_size": self.num_classes,
            "hidden_dim": 100,
            "num_layers": 3,
            "dropout": 0.5,
            "norm": nn.LayerNorm,
            "activation": nn.GELU
        }

        self.fc = MLP(**config)

        self.is_2d = is_2d
        self.use_delta = use_delta
        self.l1_lambda = 10

        # loss/accuracy
        # weight=torch.tensor([0.49957591, 1.53986928, 1.48737374, 1.47804266])
        self.loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.49957591, 1.53986928, 1.48737374, 1.47804266]))
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
            2: [0, -0.010472],
            # gripper left
            3: [0, 0.010472],
        }


    def forward(self, x):
        # x[:, 2] = -x[:, 2] 
        # x[:, 3] = -x[:, 3]
        if self.is_2d:
            return self.fc(x[:, 2:])
        return self.fc(x)

    def train_loop(self, train_dataloader,epoch):
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
        lr = adjust_learning_rate(self.optimizer,epoch,self.lr,self.min_lr,self.max_epochs,self.warmup_epochs)
        log_dict = {
            "avg_train_loss": np.mean(losses),
            "avg_train_acc": np.mean(accuracy),
            "avg_train_norm": np.mean(avg_prediction_magnitude),
            "lr": lr
        }
        return log_dict

    def eval_on_simulator(self, test_points):
        self.eval()
        deltas = []
        iterations = []
        threshold = 0.175
        begin_points = [
            [0.10000246696812026, -0.5253884198508321], # 1.1  
            [ 0.1000, 4.0 ], 
            [0.35, -0.53], 
            [0.35, 2.7],  #[0.35, 3.52]
            [0.100, -0.27], 
            [0.100, 2.33], 
            [0.37, -0.50], 
            [0.37, 1.42],
        ] # in form of [ [ext1, yaw1], [ext2, yaw2], ...] 
        goal_points = [ 
            [0.25, 1.3], 
            [ 0.25, -0.5254], 
            [ 0.02, 2.03], 
            [0.02, 1.74], #[0.02, -0.74]
            [0.37, -0.87], 
            [0.37, 4.0], 
            [0.11, -1.0], 
            [0.11, 2.79]
        ] # similar as before 
        for i in tqdm(range(len(begin_points))):

            if self.use_delta:
                inp = torch.tensor([[begin_points[i][0], begin_points[i][1], goal_points[i][0] - begin_points[i][0], goal_points[i][1] - begin_points[i][1]]]).to(self.device)
            else:
                inp = torch.tensor([[begin_points[i][0], begin_points[i][1], goal_points[i][0], goal_points[i][1]]]).to(self.device)

            with torch.no_grad():
                # test_point = (
                #     torch.from_numpy(np.array(test_point)).unsqueeze(0).to(self.device)
                # )
                delta_min, iteration, total_steps = self.simulate(inp, 350)
                deltas.append(delta_min)
                if delta_min < threshold:
                    iterations.append(iteration - total_steps)
                else:
                    iterations.append(350 - total_steps)

        self.train()
        log_dict = {
            "val_deltas_min": min(deltas),
            "val_deltas_mean": np.mean(deltas),
            'success': np.count_nonzero(np.array(deltas) < threshold) / len(begin_points), 
            'time steps to goal': np.mean(iterations), 
            'mean_list_dist': np.mean(deltas[len(deltas)-10 :])
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

        goal = [inp[0, 2] + inp[0, 0], inp[0, 3] + inp[0, 1]]
        self.eval()
        # goal_x, goal_y = 
        if self.is_2d:
            c_ext, c_yaw = -inp[0, 2] + goal[0], -inp[0, 3] + goal[1]
            c_x, c_y = convert_js_xy(c_ext, c_yaw)
            g_x, g_y = convert_js_xy(goal[0], goal[1])
            d_x, d_y = g_x - c_x, g_y - c_y

            # d_x, d_y = convert_js_xy(inp[0, 0], inp[0, 1])
        else:
            c_x, c_y = convert_js_xy(inp[0, 0], inp[0, 1])
            g_x, g_y = convert_js_xy(goal[0], goal[1])
            d_x, d_y = g_x - c_x, g_y - c_y
            
        delta_list.append(torch.norm(torch.tensor([d_x,d_y])).item())

        # calculate normalizing factor
        ext_steps = np.abs(inp[0, 0].cpu() / 0.04)
        yaw_steps = np.abs(inp[0, 1].cpu() / 0.010472)
        total_steps = ext_steps + yaw_steps

        temp_inp = inp.clone()


        # curr_x, curr_y = convert_js_xy(inp[0, 0], inp[0, 1])
        for _ in range(iterations):
            if self.use_delta:
                c_x, c_y = convert_js_xy(inp[0, 0], inp[0, 1])
                g_x, g_y = convert_js_xy(goal[0], goal[1])
                d_x, d_y = g_x - c_x, g_y - c_y
                temp_inp = torch.tensor([[c_x,c_y,d_x,d_y]],device=self.device)
            else:
                c_x, c_y = convert_js_xy(inp[0, 0], inp[0, 1])
                g_x, g_y = convert_js_xy(goal[0], goal[1])
                temp_inp = torch.tensor([[c_x,c_y,g_x,g_y]],device=self.device)

         
            prediction = self(temp_inp) # predict from input
            prediction = prediction.flatten() # flatten prediction

            predicted_kp = torch.argmax(prediction).item() 
            deltas = torch.Tensor(self.kp_delta_mapping[predicted_kp]).to(self.device) 

            if (-inp[0, 2] + goal[0])+ deltas[0] < 0.457 and \
                (-inp[0, 2] + goal[0]) + deltas[0] > 0.0025 and \
                (-inp[0, 3] + goal[1]) + deltas[1] < 4.5859 and \
                (-inp[0, 3] + goal[1]) + deltas[1] > -1.3837:
                    inp[0, :2] += deltas
                    inp[0, 2:] -= deltas
            delta_list.append(torch.norm(torch.Tensor([inp[0, 2],inp[0, 3]])).item())
            

        return min(delta_list), np.argmin(np.array(delta_list)), total_steps # return the closest it gets to goal

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

    @torch.no_grad()
    def evaluate_total(self, val_dataloader):
        preds = []
        actual = []
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
            actual_action = key_pressed.to(torch.int64).squeeze()          
            
            predicted_action  = torch.argmax(self(inp),dim = 1)
            actual_action = key_pressed.to(torch.int64).squeeze()
            preds.extend(list(predicted_action.cpu()))
            actual.extend(list(actual_action.cpu()))
        log_dict = {
            "Accuracy Val Total": accuracy_score(actual,preds),
            "Balanced Accuracy Val Total": balanced_accuracy_score(actual,preds),
            "F1 Score": f1_score(actual, preds, average='weighted')
        }
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
            log_dict[f'Accuracy class {i}'] = acc

        return log_dict

    def batch_to_device(self, batch, device):
        for key, value in batch.items():
            batch[key] = value.to(device)

    def keypressed_to_index(self, keypressed):
        _keypressed_to_index = {
            # noop
            "_": 0,
            # arm up
            "8": 1,
            # arm down
            "2": 2,
            # arm out
            "w": 3,
            # arm in
            "x": 4,
            # base forward
            "4": 5,
            # base back
            "6": 6,
            # base rotate left
            "7": 7,
            # base rotate right
            "9": 8,
            # gripper right
            "a": 9,
            # gripper left
            "d": 10,
            # gripper down
            "c": 11,
            # gripper up
            "v": 12,
            # gripper roll right
            "o": 13,
            # gripper roll left
            "p": 14,
            # gripper open
            "0": 15,
            # gripper close
            "5": 16,
        }
        self.kp_delta_mapping = {
            # arm out
            3: [0.04, 0],
            # arm in
            1: [-0.04, 0],
            # gripper right
            2: [0, 0.010472],
            # gripper left
            3: [0, -0.010472],
        }

        return _keypressed_to_index.get(keypressed, 0)
