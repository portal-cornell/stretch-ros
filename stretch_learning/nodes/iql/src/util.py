from pathlib import Path
import sys
sys.path.insert(0, Path.cwd().parent)
import csv
from datetime import datetime
import json
import random
import string

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from itertools import chain
from common import DATA_DIR, OVERSHOOT_ROLLOUTS_DIR, LEARNER_ROLLOUTS_DIR
from PIL import Image
from common import calculate_y_pos

DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Squeeze(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(dim=self.dim)

def mlp(dims, activation=nn.ReLU, output_activation=None, squeeze_output=False):
    n_dims = len(dims)
    assert n_dims >= 2, 'MLP requires at least two dims (input and output)'

    layers = []
    for i in range(n_dims - 2):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        layers.append(activation())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    if output_activation is not None:
        layers.append(output_activation())
    if squeeze_output:
        assert dims[-1] == 1
        layers.append(Squeeze(-1))
    net = nn.Sequential(*layers)
    net.to(dtype=torch.float32)
    return net

def compute_batched(f, xs):
    return f(torch.cat(xs, dim=0)).split([len(x) for x in xs])

def update_exponential_moving_average(target, source, alpha):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1. - alpha).add_(source_param.data, alpha=alpha)

def torchify(x):
    x = torch.from_numpy(x)
    if x.dtype is torch.float64:
        x = x.float()
    x = x.to(device=DEFAULT_DEVICE)
    return x

def return_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0., 0
    for r, d in zip(dataset['rewards'], dataset['terminals']):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0., 0
    # returns.append(ep_ret)    # incomplete trajectory
    lengths.append(ep_len)      # but still keep track of number of steps
    assert sum(lengths) == len(dataset['rewards'])
    return min(returns), max(returns)

# dataset is a dict, values of which are tensors of same first dimension
def sample_batch(dataset, batch_size):
    k = list(dataset.keys())[0]
    n, device = len(dataset[k]), dataset[k].device
    for v in dataset.values():
        assert len(v) == n, 'Dataset values must have same length'
    indices = torch.randint(low=0, high=n, size=(batch_size,), device=device)
    return {k: v[indices] for k, v in dataset.items()}

def evaluate_policy(policy, n_simul, x_pos_avg, y_pos_avg, csv_to_imgs):
    # take first time key is pressed
    csvs = [csv for csv in LEARNER_ROLLOUTS_DIR.glob("*.csv")]
    rewards = []    
    sim_rounds, keep_simulating = 0, True
    while keep_simulating:
        for csv in csvs:
            is_overshoot = "high_open" in csv.stem
            if is_overshoot:
                target_pos = y_pos_avg
                target_key = 6
            else:
                target_pos = x_pos_avg
                target_key = 5

            dataset = pd.read_csv(csv)
            transformed_images = csv_to_imgs[csv]

            predicted_pos = 0
            for i, row in dataset.iterrows():
                with torch.no_grad():
                    joint_state = row[2:44].to_numpy(dtype=np.float32)
                    transformed_image = transformed_images[i]
                    obs = np.concatenate((joint_state, transformed_image))
                    obs = torchify(obs)
                    action_idx = torch.argmax(policy.act(obs)).item()
                    kp = int(action_idx + 4)
                    if kp == target_key: break
            else:
                predicted_pos = -1

            if predicted_pos != -1:
                if is_overshoot:
                    predicted_pos = calculate_y_pos(dataset.iloc[i])
                else:
                    predicted_pos = dataset.iloc[i]["wrist_extension_pos"]
            rewards.append(-abs(predicted_pos - target_pos)*100)

            sim_rounds += 1
            if sim_rounds >= n_simul:
                keep_simulating = False
                break

    return rewards

def set_seed(seed, env=None):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if env is not None:
        env.seed(seed)


def _gen_dir_name():
    now_str = datetime.now().strftime('%m-%d-%y_%H.%M.%S')
    rand_str = ''.join(random.choices(string.ascii_lowercase, k=4))
    return f'{now_str}_{rand_str}'

class Log:
    def __init__(self, root_log_dir, cfg_dict,
                 txt_filename='log.txt',
                 csv_filename='progress.csv',
                 cfg_filename='config.json',
                 flush=True):
        self.dir = Path(root_log_dir)/_gen_dir_name()
        self.dir.mkdir(parents=True)
        self.txt_file = open(self.dir/txt_filename, 'w')
        self.csv_file = None
        (self.dir/cfg_filename).write_text(json.dumps(cfg_dict))
        self.txt_filename = txt_filename
        self.csv_filename = csv_filename
        self.cfg_filename = cfg_filename
        self.flush = flush

    def write(self, message, end='\n'):
        now_str = datetime.now().strftime('%H:%M:%S')
        message = f'[{now_str}] ' + message
        for f in [sys.stdout, self.txt_file]:
            print(message, end=end, file=f, flush=self.flush)

    def __call__(self, *args, **kwargs):
        self.write(*args, **kwargs)

    def row(self, dict):
        if self.csv_file is None:
            self.csv_file = open(self.dir/self.csv_filename, 'w', newline='')
            self.csv_writer = csv.DictWriter(self.csv_file, list(dict.keys()))
            self.csv_writer.writeheader()

        self(str(dict))
        self.csv_writer.writerow(dict)
        if self.flush:
            self.csv_file.flush()

    def close(self):
        self.txt_file.close()
        if self.csv_file is not None:
            self.csv_file.close()
