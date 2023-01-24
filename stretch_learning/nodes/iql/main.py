from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd().parent))
sys.path.insert(0, str(Path.cwd().parent.parent))

import wandb
from PIL import Image
from torchvision import transforms
from r3m import load_r3m
from utils.common import *
from src.util import return_range, set_seed, Log, sample_batch, torchify, evaluate_policy
from src.value_functions import TwinQ, ValueFunction
from src.policy import GaussianPolicy, DeterministicPolicy
from src.iql import ImplicitQLearning
from tqdm import trange
from itertools import chain
import torch
import numpy as np
import pandas as pd
import math
import gym


wandb.init(project="iql_open_drawer")
device = "cuda" if torch.cuda.is_available else "cpu"


@progress_alerts(func_name="iql in stretch_behavioral_cloning")
def main(args):
    torch.set_num_threads(1)
    log = Log(Path(args.log_dir)/args.env_name, vars(args))
    log(f'Log dir: {log.dir}')

    r3m = load_r3m("resnet18")
    resnet_transform = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset = get_subdom_cost(r3m, resnet_transform)
    for k, v in dataset.items():
        dataset[k] = torchify(v)
        
    obs_dim = dataset['observations'].shape[1]
    act_dim = dataset['actions'].shape[1]   # this assume continuous actions
    # set_seed(args.seed, env=env)

    if args.deterministic_policy:
        policy = DeterministicPolicy(
            obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden)
    else:
        policy = GaussianPolicy(
            obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden)

    x_pos_avg, y_pos_avg, csv_to_imgs = get_metadata(r3m, resnet_transform)
    def eval_policy():
        eval_returns = np.array(evaluate_policy(
            policy, args.n_eval_episodes, x_pos_avg, y_pos_avg, csv_to_imgs))
        # normalized_returns = d4rl.get_normalized_score(args.env_name, eval_returns) * 100.0
        log.row({
            'return mean': eval_returns.mean(),
            'return std': eval_returns.std(),
        })
        wandb.log({
            'return mean': eval_returns.mean(),
            'return std': eval_returns.std(),
        })
        return eval_returns.mean()

    iql = ImplicitQLearning(
        qf=TwinQ(obs_dim, act_dim, hidden_dim=args.hidden_dim,
                 n_hidden=args.n_hidden),
        vf=ValueFunction(obs_dim, hidden_dim=args.hidden_dim,
                         n_hidden=args.n_hidden),
        policy=policy,
        optimizer_factory=lambda params: torch.optim.Adam(
            params, lr=args.learning_rate),
        max_steps=args.n_steps,
        tau=args.tau,
        beta=args.beta,
        alpha=args.alpha,
        discount=args.discount
    )

    highest_reward = -math.inf
    train_type = "iql_subdom"
    save_dir = Path(ALL_CKPTS_DIR, train_type)
    if not save_dir.exists():
        save_dir.mkdir()
    for step in trange(args.n_steps):
        iql.update(**sample_batch(dataset, args.batch_size))
        if (step+1) % args.eval_period == 0:
            mean_reward = eval_policy()
            if mean_reward > highest_reward:
                torch.save(iql.state_dict(), str(
                    Path(save_dir, f"{train_type}_{step}_{mean_reward}.pth")))
                highest_reward = mean_reward
            torch.save(iql.state_dict(), str(
                Path(save_dir, f"{train_type}_last.pth")))
    log.close()

def get_learner_csv_to_imgs(r3m, resnet_transform):
    csvs = [csv for csv in LEARNER_ROLLOUTS_DIR.glob("*.csv")]
    overshoot_csv_to_imgs = {}
    for csv in csvs:
        dataset = pd.read_csv(csv)
        img_paths = dataset[dataset.columns[44]].values.tolist()
        transformed_images = []
        for img_path in img_paths:
            img_path = Path(DATA_DIR, img_path)
            image = Image.open(img_path)
            image = resnet_transform(image).cpu().tolist()
            transformed_images.append(image)
        transformed_images = torch.tensor(np.array(transformed_images), device=device)
        transformed_images = r3m(transformed_images).cpu().tolist()
        overshoot_csv_to_imgs[csv] = transformed_images
    return overshoot_csv_to_imgs

def get_metadata(r3m, resnet_transform):
    far_open_csvs = [csv for csv in OVERSHOOT_ROLLOUTS_DIR.glob("far_open_*.csv")]
    x_pos_avg = 0
    for csv in far_open_csvs:
        dataset = pd.read_csv(csv)
        transition_row = dataset.loc[dataset["key_pressed"] == 6].iloc[0]
        x_pos_avg += calculate_y_pos(transition_row)
    high_open_csvs = [csv for csv in OVERSHOOT_ROLLOUTS_DIR.glob("high_open_*.csv")]
    y_pos_avg = 0
    for csv in high_open_csvs:
        if csv.stem == "/share/cuvl/jlr429/bc/data/csv/high_open_drawer_2022-11-16-20-29-26.csv":
            # bad rollout
            continue
        dataset = pd.read_csv(csv)
        transition_row = dataset.loc[dataset["key_pressed"] == 5].iloc[0]
        y_pos_avg += transition_row["wrist_extension_pos"]

    x_pos_avg = x_pos_avg/len(far_open_csvs)
    y_pos_avg = y_pos_avg/(len(high_open_csvs)-1)
    
    csvs = [csv for csv in LEARNER_ROLLOUTS_DIR.glob("*.csv")]
    csv_to_imgs = {}
    for csv in csvs:
        dataset = pd.read_csv(csv)
        img_paths = dataset[dataset.columns[44]].values.tolist()
        transformed_images = []
        for img_path in img_paths:
            img_path = Path(DATA_DIR, img_path)
            image = Image.open(img_path)
            image = resnet_transform(image).cpu().tolist()
            transformed_images.append(image)
        transformed_images = torch.tensor(np.array(transformed_images), device=device)
        transformed_images = r3m(transformed_images).cpu().tolist()
        csv_to_imgs[csv] = transformed_images
    return x_pos_avg, y_pos_avg, csv_to_imgs

def get_subdom_cost(r3m, resnet_transform):
    csvs = [csv for csv in chain(LEARNER_ROLLOUTS_DIR.glob("*.csv"), REG_CSVS_DIR.glob("*.csv"))]
    
    longest_traj = 0
    for csv in csvs:
        longest_traj = max(longest_traj, len(pd.read_csv(csv)))

    trajs = []
    for csv in csvs:
        dataset = pd.read_csv(csv)

        # actions
        key_presses = dataset[dataset.columns[1]].values.tolist()
        actions = []
        for action in key_presses:
            # 4=up, 6=forward, 5=down, 7=back
            action_vec = [0., 0., 0., 0.]
            action_vec[int(action)-4] = 1.
            actions.append(action_vec)
        actions = actions[:-1]

        # obs/next_obs
        joint_states = dataset[dataset.columns[2:44]].values.tolist()
        img_paths = dataset[dataset.columns[44]].values.tolist()
        transformed_images = []
        for img_path in img_paths:
            img_path = Path(DATA_DIR, img_path)
            image = Image.open(img_path)
            image = resnet_transform(image).cpu().tolist()
            transformed_images.append(image)
        transformed_images = torch.tensor(np.array(transformed_images), device=device)
        transformed_images = r3m(transformed_images).cpu().tolist()

        obs, next_obs = [], []
        for idx, (img, js) in enumerate(zip(transformed_images, joint_states)):
            if idx == len(joint_states)-1:
                break
            curr_img, curr_js = img.copy(), js.copy()
            curr_img.extend(curr_js)
            obs.append(curr_img)
            next_img, next_js = transformed_images[idx + 1].copy(), joint_states[idx+1].copy()
            next_img.extend(next_js)
            next_obs.append(next_img)

        # terminals
        terminals = np.zeros(len(key_presses)-1)
        terminals[-1] = 1.0

        # rewards/subcosts
        effort_cost = len(dataset) / longest_traj
        success_cost = 1 if "learner" in str(csv) else 0    # if learner traj, cost of 1 = unsuccessful

        # turn discrete actions into continuous
        trajs.append(dict(
            obs=np.array(obs, dtype=np.float32),
            actions=np.array(actions, dtype=np.float32),
            next_obs=np.array(next_obs, dtype=np.float32),
            terminals=terminals,
            subcosts=np.array([effort_cost, success_cost])))

    # calculate subdom
    for target_idx, target_traj in enumerate(trajs):
        subdom = 0
        target_subcosts = target_traj["subcosts"]
        for other_idx, other_traj in enumerate(trajs):
            if target_idx == other_idx:
                continue
            other_subcosts = other_traj["subcosts"]
            for subcost_idx in range(len(target_subcosts)):
                subdom += max(0, target_subcosts[subcost_idx] -
                              other_subcosts[subcost_idx])
        # lower subdom = higher reward
        target_traj["rewards"] = np.ones(len(target_traj["obs"])) * (-subdom)

    # concat all trajs
    obs, actions, rewards, next_obs, terminals = [], [], [], [], []
    for traj in trajs:
        obs.extend(traj["obs"])
        actions.extend(traj["actions"])
        rewards.extend(traj["rewards"])
        next_obs.extend(traj["next_obs"])
        terminals.extend(traj["terminals"])

    return dict(
        observations=np.array(obs, dtype=np.float32),
        actions=np.array(actions, dtype=np.float32),
        rewards=np.array(rewards, dtype=np.float32),
        next_observations=np.array(next_obs, dtype=np.float32),
        terminals=np.array(terminals, dtype=np.float32),
    )


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--env-name', default="LunarLander-v2")
    parser.add_argument('--log-dir', default="logs/")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--n-hidden', type=int, default=2)
    # parser.add_argument('--n-steps', type=int, default=3*10**5)
    parser.add_argument('--n-steps', type=int, default=10**6)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--alpha', type=float, default=0.005)
    parser.add_argument('--tau', type=float, default=0.7)
    parser.add_argument('--beta', type=float, default=3.0)
    parser.add_argument('--deterministic-policy', action='store_true')
    parser.add_argument('--eval-period', type=int, default=5000)
    parser.add_argument('--n-eval-episodes', type=int, default=20)
    parser.add_argument('--max-episode-steps', type=int, default=1000)
    # parser.add_argument('--subdom', action='store_true', default=False)
    main(parser.parse_args())