# -*- coding: utf-8 -*-
"""hal_controller_sim_v2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1jx6kMtN-5j0FXn5nNBkhOCN7qwk3c1MF

HAL Controller Simulation Environment
"""

# import packages
import sys
import numpy as np
import torch
import gymnasium
from gymnasium import core, spaces
# from gym import spaces,core
import matplotlib.pyplot as plt
import torch as th

class HalControllerEnv(gymnasium.Env):

  def __init__(self,max_steps=400):

    # joints
    self.s_high = np.array([0.457, 1.5, 1.1]) # 4.586
    self.s_low  = np.array([0.0025, -1.3837, 0.1538])

    # xyz
    self.s_xyz_high = np.array([0.22, 0.68, 1.1])
    self.s_xyz_low = np.array([-0.22, 0.005, 0.15])
    self.g_high = np.array([0.1, 0.64, 1])
    self.g_low = np.array([-0.1, 0.1, 0.16])

    self.s_dims = 3 # ext, yaw, lift
    self.s_xyz_dims = 3 # xyz
    self.u_dims = 1 # keypress
    self.g_dims = 3 # xyz

    self.sparse_reward = False
    self.fixed_start = False
    
    self.eps = 0.03
    self.prop_steps = 1

    # triangle rewards
    self.goal_reward = 1
    self.t_delta_x = 0.05
    self.t_delta_y = 0.1
    self.t_delta_z = 0.05
    self.triangle_reward = -0.2
    self.out_triangle_reward = -10

    # trapezoid rewards
    self.t_delta_in_x = 0.015
    self.t_delta_out_x = 0.06
    

    self.triangle_reward_sparse = -0.1
    self.goal_reward_sparse = 0

    self.kp_delta_mapping = {
        # arm out
        0: [0.04, 0, 0],
        # arm in
        1: [-0.04, 0, 0],
        # gripper right
        2: [0, -0.010472, 0],
        # gripper left
        3: [0, 0.010472, 0],
        # arm up
        4: [0, 0, 0.04],
        # arm down
        5: [0, 0, -0.04],
    }

    self.kp_mapping = {
        0: "Arm out",
        1: "Arm in",
        2: "Gripper right",
        3: "Gripper left",
        4: "Arm up",
        5: "Arm down",
    }

    xyz_bound = self.s_xyz_high - self.s_xyz_low
    self.observation_space = spaces.Box(low=-xyz_bound, high=xyz_bound, shape=(3, ), dtype=np.float32)

    self.action_space = spaces.Discrete(6)

    self.max_steps = max_steps
    self.reset()

  def _get_delta_state(self):
    return self.delta_state.astype(np.float32)


  def reset(self, seed=0, start = None, goal = None):
    self.steps = 0

    self.state = np.zeros((self.s_dims,), dtype=np.float32)
    self.state_xyz = np.zeros((self.s_xyz_dims,), dtype=np.float32)
    self.goal  = np.zeros((self.g_dims,), dtype=np.float32)
    self.delta_state = self.goal - self.state_xyz

    if start is None:
      for i in range(self.s_dims):
        if not self.fixed_start:
          self.state[i] = np.random.uniform(self.s_low[i],self.s_high[i])
        self.goal[i]  = np.random.uniform(self.g_low[i],self.g_high[i])
      self.state_xyz = self.convert_js_xyz(np.copy(self.state))
      self.delta_state = self.goal - self.state_xyz
    else:
      self.state = np.array(start,dtype = np.float32)
      self.state_xyz = self.convert_js_xyz(self.state)
      self.goal = np.array(goal,dtype = np.float32)
      self.delta_state = self.goal - self.state_xyz

    return (self._get_delta_state(),{})


  def _enforce_bounds(self,s):
    for i in range(self.s_dims):
      s[i] = np.clip(s[i],self.s_low[i],self.s_high[i])
    return s

  def is_valid(self, js, delta):
    """
    js: current [ext, yaw, lift]
    delta: delta [ext, yaw, lift]
    """

    if js[0] + delta[0] <= self.s_high[0] and \
      js[0] + delta[0] >= self.s_low[0] and \
      js[1] + delta[1] <= self.s_high[1] and \
      js[1] + delta[1] >= self.s_low[1] and \
      js[2] + delta[2] <= self.s_high[2] and \
      js[2] + delta[2] >= self.s_low[2]:
      return True
    else:
      return False


  def update_dynamics(self, js, delta):
    """
    js: [curr_ext, curr_yaw, curr_lift]
    deltas: [delta_ext, delta_yaw, delta_lift]
    returns: updated state
    """
    valid_action = False
    if self.is_valid(js, delta):
        valid_action = True
        js[0] = js[0] + delta[0]
        js[1] = js[1] + delta[1]
        js[2] = js[2] + delta[2]

    return js, valid_action

  def convert_js_xyz(self, joint_state):
    extension = joint_state[0]
    yaw = joint_state[1]
    lift = joint_state[2]

    gripper_len = 0.22
    base_gripper_yaw = -0.09
    yaw_delta = -(yaw - base_gripper_yaw)  # going right is more negative
    y = gripper_len * torch.cos(torch.tensor([yaw_delta])) + extension
    x = gripper_len * torch.sin(torch.tensor([yaw_delta]))
    z = lift

    return np.array([x.item(),y.item(),z.item()])

  def _terminal(self,s_xyz,goal):
    diff = s_xyz-goal
    if np.linalg.norm(diff) <= self.eps: return True, True
    if self.steps >= self.max_steps: return True, False
    return False, False

  def calculate_area(self, p1, p2, p3):
    return 0.5 * abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))

  def in_trapezoid(self, vertices, point):
    
    p1, p2, p3, p4 = vertices
    area_trapezoid = self.calculate_area(p1, p2, p3) + self.calculate_area(p1, p3, p4)

    area_triangle1 = self.calculate_area(p1, p2, point)
    area_triangle2 = self.calculate_area(p2, p3, point)
    area_triangle3 = self.calculate_area(p3, p4, point)
    area_triangle4 = self.calculate_area(p1, p4, point)

    return abs(area_triangle1 + area_triangle2 - area_trapezoid) < 1e-6

  def get_trapezoid_vertices(self, goal):
    goal_x = goal[0]
    goal_y = goal[1]
    goal_z = goal[2]

    x1 = goal_x - self.t_delta_in_x
    y1 = goal_y
    x2 = goal_x + self.t_delta_in_x
    y2 = goal_y
    x3 = goal_x - self.t_delta_out_x
    y3 = goal_y - self.t_delta_y
    x4 = goal_x + self.t_delta_out_x
    y4 = goal_y - self.t_delta_y

    return x1, y1, x2, y2, x3, y3, x4, y4

  def in_trap_reward_area(self, test_point, goal):
    x1, y1, x2, y2, x3, y3, x4, y4 = self.get_trapezoid_vertices(goal)
    trapezoid_vertices = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    test_point_xy = [test_point[0], test_point[1]]
    if (
      self.in_trapezoid(trapezoid_vertices, test_point_xy) and 
      test_point[2] <= min(goal[2] + self.t_delta_z, self.g_high[2]) and
      test_point[2] >= max(goal[2] - self.t_delta_z, self.g_low[2])
    ):
      return True
    else:
      return False

  def compute_reward_trapezoid(self, achieved_goal, goal):
    if self.sparse_reward:
      if (np.linalg.norm(achieved_goal - goal) <= self.eps):
        reward = self.goal_reward_sparse
      elif self.in_trap_reward_area(achieved_goal, goal):
        reward = self.triangle_reward_sparse
      else:
        reward = -1
      return reward
    else:
      dist_reward = -np.linalg.norm(achieved_goal - goal)
      goal_reward = self.goal_reward*(np.linalg.norm(achieved_goal - goal) <= self.eps)
      triangle_reward = 0
      if self.in_trap_reward_area(achieved_goal, goal):
        triangle_reward = self.triangle_reward
      else:
        triangle_reward = self.out_triangle_reward
      return dist_reward + goal_reward + triangle_reward

  def step(self,a):
    s = self.state

    u = np.copy(a)
    u = self.kp_delta_mapping[int(u)]

    for i in range(self.prop_steps):
      s, valid_action = self.update_dynamics(s,u)


    self.state = s
    self.state_xyz = self.convert_js_xyz(self.state)
    self.delta_state = self.goal - self.state_xyz

    terminal, success = self._terminal(self.state_xyz,self.goal)

    # reward = self.compute_reward_special(self.state_xyz, self.goal)
    reward = self.compute_reward_trapezoid(self.state_xyz, self.goal)


    self.steps += 1

    return (self._get_delta_state(),reward,terminal,terminal,{'is_success':success})

  def render(self, obs, action):
    """

    """
    state_xyz = obs['achieved_goal']
    goal_xyz = obs['desired_goal']
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.grid()
    sim_scatter = ax.scatter([state_xyz[0]], [state_xyz[1]], [state_xyz[2]], s=5, c=action, cmap='viridis', alpha=1)
    handles, _ = sim_scatter.legend_elements()

    filtered_kp_mapping = [self.kp_mapping[i] for i in np.unique(action)]
    plt.legend(handles, filtered_kp_mapping, title="Key Presses", loc='best', bbox_to_anchor=(0.8, 0., 0.5, 0.5))

    plt.plot(
      [self.goal[0]],
      [self.goal[1]],
      [self.goal[2]],
      marker="*",
      markersize=15,
      color="red",
      label="goal",
    )

    self.s_xyz_high = np.array([0.20, 0.65, 1])
    self.s_xyz_low = np.array([-0.20, 0.005, 0.16])

    plt.xlim(-0.25,0.25)
    plt.ylim(0.68,0.)
    plt.zlim(0.15, 1.1)

    plt.xlabel("Position X")
    plt.ylabel("Position Y")

    plt.title("HAL Controller Sim with PPO")

    plt.savefig("PPO Training")
    # plt.show()
    plt.close()





def plot_traj(state_list, action_list, goal):
  kp_mapping = {
    0: "Arm out",
    1: "Arm in",
    2: "Gripper right",
    3: "Gripper left",
    4: "Arm up",
    5: "Arm down",
  }
  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')
  ax.grid()

  s = np.array(state_list)
  plot_x = s[:, 0]
  plot_y = s[:, 1]
  plot_z = s[:, 2]
  sim_scatter = ax.scatter(plot_x[:-1], plot_y[:-1], plot_z[:-1], s=5, c=action_list, cmap='viridis', alpha=1)
  handles, _ = sim_scatter.legend_elements()

  filtered_kp_mapping = [kp_mapping[i] for i in np.unique(action_list)]
  plt.legend(handles, filtered_kp_mapping, title="Key Presses")

  # Plot the start
  plt.plot(
    [plot_x[0]],
    [plot_y[0]],
    [plot_z[0]],
    marker=".",
    markersize=15,
    color="pink",
    label="start",
  )

  # Plot the goal
  plt.plot(
    [goal[0]],
    [goal[1]],
    [goal[2]],
    marker="*",
    markersize=15,
    color="red",
    label="goal",
  )

  # Plot the end of trajectory with no action
  plt.plot(
    [plot_x[-1]],
    [plot_y[-1]],
    [plot_z[-1]],
    marker=".",
    markersize=5,
    color="orange",
    label="end",
  )

  ax.set_xlim(-0.25, 0.25)
  ax.set_ylim(0.68, 0.)
  ax.set_zlim(0.15, 1.1)

  ax.set_xlabel('Position X (m)')
  ax.set_ylabel('Position Y (m)')
  ax.set_zlabel('Position Z (m)')
  plt.title("HAL Controller Sim with PPO")

  plt.savefig("PPO_eval")
  plt.show()
  plt.close()

def evaluate(model, env, num_episodes=1):
  all_episode_successes = []
  state_list, action_list = [], []
  for i in range(num_episodes):
    episode_rewards = []
    done = False
    obs = env.reset()[0]
    # obs = obs[0]
    state_list.append(obs)
    # goal = obs['desired_goal']
    goal = env.goal
    while not done:
      action, _ = model.predict(obs)
      action_list.append(action)
      obs, reward, done, _, info = env.step(action)
      state_list.append(obs)
      # @aravind: There has to be a way of getting the success directly from the environment.
    all_episode_successes.append(np.linalg.norm(np.copy(env.state_xyz) - env.goal) < env.eps)

  success_rate = 1.0 * sum(all_episode_successes)/num_episodes
  return success_rate, state_list, action_list, goal

from gymnasium.envs.registration import register
register(id="HalControllerEnv",entry_point=HalControllerEnv,max_episode_steps=400)

"""Training Policy"""

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
import optuna

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances
from typing import Any, Dict
import torch
import torch.nn as nn
from stable_baselines3 import PPO

N_TRIALS = 100  # Maximum number of trials
N_JOBS = 1 # Number of jobs to run in parallel
N_STARTUP_TRIALS = 5  # Stop random sampling after N_STARTUP_TRIALS
N_EVALUATIONS = 4  # Number of evaluations during the training 4
N_TIMESTEPS = int(600000)  # Training budget # 900,000
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_ENVS = 5
N_EVAL_EPISODES = 10
TIMEOUT = int(60 * 1000)  # 15 minutes

ENV_ID = "HalControllerEnv"

DEFAULT_HYPERPARAMS = {
    "policy": "MlpPolicy",
    "env": ENV_ID,
}



def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for A2C hyperparameters.

    :param trial: Optuna trial object
    :return: The sampled hyperparameters for the given trial.
    """
    # Discount factor between 0.9 and 0.9999
    gamma = 1.0 - trial.suggest_float("gamma", 0.0001, 0.1, log=True)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 5.0, log=True)
    # 8, 16, 32, ... 1024
    n_steps = 2 ** trial.suggest_int("exponent_n_steps", 3, 8)

    ### YOUR CODE HERE
    # TODO:
    # - define the learning rate search space [1e-5, 1] (log) -> `suggest_float`
    # - define the network architecture search space ["tiny", "small"] -> `suggest_categorical`
    # - define the activation function search space ["tanh", "relu"]
    learning_rate = trial.suggest_float("learning_rate", 0.00001, 1, log=True)
    gae_lambda = 1.0 - trial.suggest_float("gae_lambda", 0.0001, 0.1, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.2)

    net_arch = trial.suggest_categorical("net_arch", ["tiny", "small"])

    ### END OF YOUR CODE

    # Display true values
    trial.set_user_attr("gamma", gamma)
    trial.set_user_attr("n_steps", n_steps)
    trial.set_user_attr("gae_lambda", gae_lambda)
    trial.set_user_attr("ent_coef", ent_coef)

    net_arch = {"pi": [64], "vf": [64]} if net_arch == "tiny" else {"pi": [64, 64], "vf": [64, 64]}
    

    return {
        "n_steps": n_steps,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "max_grad_norm": max_grad_norm,
        "policy_kwargs": {
            "net_arch": net_arch
        },
        "gae_lambda": gae_lambda,
        "ent_coef": ent_coef
    }

from stable_baselines3.common.callbacks import EvalCallback

class TrialEvalCallback(EvalCallback):
    """
    Callback used for evaluating and reporting a trial.

    :param eval_env: Evaluation environement
    :param trial: Optuna trial object
    :param n_eval_episodes: Number of evaluation episodes
    :param eval_freq:   Evaluate the agent every ``eval_freq`` call of the callback.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic policy.
    :param verbose:
    """

    def __init__(
        self,
        eval_env: gymnasium.Env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
    ):

        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Evaluate policy (done in the parent class)
            super()._on_step()
            self.eval_idx += 1
            # Send report to Optuna
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True

def objective(trial: optuna.Trial) -> float:
    """
    Objective function using by Optuna to evaluate
    one configuration (i.e., one set of hyperparameters).

    Given a trial object, it will sample hyperparameters,
    evaluate it and report the result (mean episodic reward after training)

    :param trial: Optuna trial object
    :return: Mean episodic reward after training
    """

    kwargs = DEFAULT_HYPERPARAMS.copy()
    ### YOUR CODE HERE
    # TODO:
    # 1. Sample hyperparameters and update the default keyword arguments: `kwargs.update(other_params)`
    # 2. Create the evaluation envs
    # 3. Create the `TrialEvalCallback`


    # 1. Sample hyperparameters and update the keyword arguments
    kwargs.update(sample_ppo_params(trial))
    # Create the RL model
    model = PPO(**kwargs)

    # 2. Create envs used for evaluation using `make_vec_env`, `ENV_ID` and `N_EVAL_ENVS`
    env_id = "HalControllerEnv"
    num_cpu = 8
    eval_envs = make_vec_env(env_id, n_envs=num_cpu)
    # 3. Create the `TrialEvalCallback` callback defined above that will periodically evaluate
    # and report the performance using `N_EVAL_EPISODES` every `EVAL_FREQ`
    # TrialEvalCallback signature:
    # TrialEvalCallback(eval_env, trial, n_eval_episodes, eval_freq, deterministic, verbose)


    eval_callback = TrialEvalCallback(eval_envs, trial, N_EVAL_EPISODES, EVAL_FREQ, deterministic=True, verbose=False)

    ### END OF YOUR CODE

    nan_encountered = False
    try:
        # Train the model
        model.learn(N_TIMESTEPS, callback=eval_callback)
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN
        print(e)
        nan_encountered = True
    finally:
        # Free memory
        model.env.close()
        eval_envs.close()

    # Tell the optimizer that the trial failed
    if nan_encountered:
        return float("nan")

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.last_mean_reward



# Set pytorch num threads to 1 for faster training
th.set_num_threads(1)
# Select the sampler, can be random, TPESampler, CMAES, ...
sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
# Do not prune before 1/3 of the max budget is used
pruner = MedianPruner(
    n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3
)
# Create the study and start the hyperparameter optimization
study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")

try:
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_JOBS, timeout=TIMEOUT)
except KeyboardInterrupt:
    pass

print("Number of finished trials: ", len(study.trials))

print("Best trial:")
trial = study.best_trial

print(f"  Value: {trial.value}")

print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

print("  User attrs:")
for key, value in trial.user_attrs.items():
    print(f"    {key}: {value}")

with open("./hyperparameters2.txt", 'w') as file:
	file.write("Best trial:\n")
	file.write(f"  Value: {trial.value}\n")
	file.write("  Params: \n")
	for key, value in trial.params.items():
		file.write(f"    {key}: {value}\n")

	file.write("  User attrs:\n")
	for key, value in trial.user_attrs.items():
		file.write(f"    {key}: {value}\n")

# Write report
study.trials_dataframe().to_csv("study_results_ppo_halcontroller.csv")

fig1 = plot_optimization_history(study)
fig2 = plot_param_importances(study)

fig1.show()
fig2.show()
fig1.savefig("fig1.png")
fig2.savefig("fig2.png")
# from stable_baselines3 import PPO
# from google.colab import drive
# drive.mount('/content/drive')
# num_cpu = 8
# env_id = "HalControllerEnv"
# # train_env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
# vec_env = make_vec_env(env_id, n_envs=num_cpu)
# # env = gymnasium.make("HalControllerEnv")
# model = PPO("MultiInputPolicy", vec_env, verbose=1,device = "cuda")
# model.learn(total_timesteps=900000)
# model.save("/content/drive/My Drive/PPO_models/sparse_1")
# # model = model.load("/content/drive/My Drive/PPO_models/ppo_hal_controller_sparse")

# env = gymnasium.make("HalControllerEnv")
# success_rate, state_list, action_list, goal = evaluate(model, env, 100)
# print("Success Rate: ",success_rate)
# # plot_traj(state_list, action_list, goal)

# from google.colab import drive
# drive.mount('/content/drive')
# !mkdir -p "/content/drive/My Drive/PPO_models"
# model.save("/content/drive/My Drive/PPO_models/ppo_hal_controller")

# !ls