# -*- coding: utf-8 -*-

# import packages
import numpy as np
import gymnasium
from gymnasium import spaces
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import yaml
from typing import Callable

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.envs.registration import register

from stable_baselines3 import PPO

import env_utils
import evaluation_utils

from tqdm.auto import tqdm
from agent import HAL_Agent
class HalControllerEnv(gymnasium.Env):

  def __init__(self,max_steps=500):

    # joints
    self.s_high = np.array([0.457, 1.5, 1.1, 0, 0, 0]) 
    self.s_low  = np.array([0.0025, -1.3837,  0.1538, 0, 0, 0])

    self.s_narrow_high = np.array([0.02, 0.01, 1.1, 0, 0, 0]) 
    self.s_narrow_low = np.array([0.0025, -0.01,  0.1538, 0, 0, 0])

    # xyz
    self.s_xyz_high = np.array([0.5, 0.75, 1.1])
    self.s_xyz_low = np.array([-0.5, -0.75, 0.15])

    self.g_high = np.array([0.43, 1.4, 1.0, 0.2, 0.,  0.]) 
    self.g_low  = np.array([0.164, -1.3,  0.16, -0.2, 0., 0.])

    self.s_dims = 6 # ext, yaw, lift, curr_x, curr_y, base angle
    self.s_xyz_dims = 3 # xyz
    self.u_dims = 1 # keypress
    self.g_dims = 6 # joint state

    self.eps_z = 0.023
    self.eps_x = 0.02
    self.eps_y = 0.015

    # bin rewards
    self.blinder_y = 0.1
    self.blinder_x = 0.020
    self.hit_blinder = -5

    # shelf
    self.shelf_x_offset = 0.25
    self.shelf_y_offset = 0.14 # 0.14
    self.shelf_z_offset = 0.05
    self.hit_shelf = -5

    # ext, yaw, lift, base_trans, base_angle
    self.kp_delta_mapping = {
        # arm out
        0: [0.04, 0, 0, 0, 0],
        # arm in
        1: [-0.04, 0, 0, 0, 0],
        # gripper right
        2: [0, -0.10472, 0, 0, 0],
        # gripper left
        3: [0, 0.10472, 0, 0, 0],
        # arm up
        4: [0, 0, 0.04, 0, 0],
        # arm down
        5: [0, 0, -0.04, 0, 0],
        # base forward (left)
        6: [0, 0, 0, 0.04, 0],
        # base backward (right)
        7: [0, 0, 0, -0.04, 0], 
        # base rotate left, 
        8: [0, 0, 0, 0, 0.10472], 
        # base rotate right, 
        9: [0, 0, 0, 0, -0.10472]
    }

    self.kp_mapping = {
        0: "Arm out",
        1: "Arm in",
        2: "Gripper right",
        3: "Gripper left",
        4: "Arm up",
        5: "Arm down",
        6: "Base Forward (left)", 
        7: "Base Backward (right)",
        8: "Base Rotate Left", 
        9: "Base Rotate Right"
    }

    state_low = np.zeros(4)
    state_high = np.zeros(4)
    xyz_bound = self.s_xyz_high - self.s_xyz_low
    state_low[:3] = -xyz_bound
    state_low[3] = -np.pi / 10
    state_high[:3] = xyz_bound
    state_high[3] = np.pi / 10
    
    self.observation_space = spaces.Box(low=state_low, high=state_high, shape=(4, ), dtype=np.float32)

    self.action_space = spaces.Discrete(10)

    self.max_steps = max_steps
    self.reset()

  def _get_delta_state(self):
    obs_arr = np.zeros(4, dtype=np.float32)
    obs_arr[:3] = self.delta_state.astype(np.float32)
    obs_arr[3] = self.state_js[5]
    return obs_arr

  def reset(self, seed=None, mode="standard", start=None, goal=None):
    """
    mode (str): [recovery, fixed, narrow, standard]
    """
    self.steps = 0
    print(mode)
    if mode == "recovery":
      state_js, state_xyz, goal_xyz, delta_state = env_utils.recovery_reset(self.s_low, self.s_high, self.g_low, self.g_high, seed)
    elif mode == "fixed":
      state_js, state_xyz, goal_xyz, delta_state = env_utils.fixed_reset(start, goal)
    elif mode == "narrow":
      state_js, state_xyz, goal_xyz, delta_state = env_utils.standard_reset(self.s_narrow_low, self.s_narrow_high, self.g_low, self.g_high, seed)
    elif mode == "standard":
      state_js, state_xyz, goal_xyz, delta_state = env_utils.standard_reset(self.s_low, self.s_high, self.g_low, self.g_high, seed)

    self.state_js = state_js
    self.state_xyz = state_xyz
    self.goal_xyz = goal_xyz
    self.delta_state = delta_state

    return (self._get_delta_state(),{})
    
  def _enforce_bounds(self,s):
    for i in range(self.s_dims):
      s[i] = np.clip(s[i],self.s_low[i],self.s_high[i])
    return s
  
  def update_dynamics(self, js, delta):
    """
	  js: [curr_ext, curr_yaw, curr_lift, curr_x, curr_y, base_rot]
	  deltas: [delta_ext, delta_yaw, delta_lift, delta_base, base_rot]
	  returns: updated state
	  """
    js_valid = False
    grid_valid = False
    shelf_valid = False
    js_new = np.copy(js)
    js_final = np.copy(js)

	  # update the joint states with delta
    js_new[0] = js[0] + delta[0]
    js_new[1] = js[1] + delta[1]
    js_new[2] = js[2] + delta[2]
    js_new[3] = js[3] - delta[3] * np.cos(js[5]).item() # - because positive base trans goes left
    js_new[4] = js[4] - delta[3] * np.sin(js[5]).item() # - because positive base trans goes left
    js_new[5] = (js[5] + delta[4] + np.pi) % (2 * np.pi) - np.pi # constrains angle to be within -pi ~ pi

    if env_utils.is_valid(js_new, self.s_low, self.s_high):
      js_valid = True
			
      if env_utils.blinder_valid(self.goal_xyz, js_new[5], js, js_new, self.blinder_x, self.blinder_y):
        grid_valid = True

        if env_utils.shelf_valid(self.goal_xyz, js, js_new, self.shelf_x_offset, self.shelf_y_offset, self.shelf_z_offset):
          shelf_valid = True
          js_final = js_new
    return js_final, js_valid, grid_valid, shelf_valid
    
  def _terminal(self, s_xyz, goal_xyz):
    in_z = np.abs(goal_xyz[2] - s_xyz[2]) <= self.eps_z
    in_y = np.abs(goal_xyz[1] - s_xyz[1]) <= self.eps_y
    in_x = np.abs(goal_xyz[0] - s_xyz[0]) <= self.eps_x

    no_steps_left = self.steps >= self.max_steps
    if in_x and in_y and in_z: return True, True
    if no_steps_left: return True, False
    return False, False

  def compute_reward_graph(self, achieved_goal, goal, grid_valid, shelf_valid):
    # distance rewards
    dist = np.linalg.norm(achieved_goal - goal)
    dist_reward = (np.exp(-dist) - 1)

    # penalize hitting a shelf or blinder
    invalid_reward = 0
    if (not grid_valid) or (not shelf_valid):
       invalid_reward = self.hit_blinder

    in_z = np.abs(goal[2] - achieved_goal[2]) <= self.eps_z
    in_y = np.abs(goal[1] - achieved_goal[1]) <= self.eps_y
    in_x = np.abs(goal[0] - achieved_goal[0]) <= self.eps_x

    if in_z and in_y and in_x:
       return 1

    return dist_reward + invalid_reward
    
  def step(self,a):
    s = self.state_js
    
    u = np.copy(a)
    u = np.array(self.kp_delta_mapping[int(u)]) * (3/5)
    goal_xyz = np.copy(self.goal_xyz)
    
    new_js, _, grid_valid, shelf_valid = self.update_dynamics(s,u)

    self.state_js = new_js
    self.state_xyz = env_utils.convert_js_xyz(self.state_js)
    self.delta_state = goal_xyz - self.state_xyz

    terminal, success = self._terminal(self.state_xyz, goal_xyz)
    
    reward = self.compute_reward_graph(np.copy(self.state_xyz), goal_xyz, grid_valid, shelf_valid)

    self.steps += 1

    return (self._get_delta_state(),reward,terminal,terminal,{'is_success':success})

  def simulate_step(self, a):
    s = self.state_js

    u = np.copy(a)
    u = np.array(self.kp_delta_mapping[int(u)]) * (3/5)
    goal_xyz = np.copy(self.goal_xyz)
    
    state_js, _, _, _ = self.update_dynamics(s,u)

    state_xyz = env_utils.convert_js_xyz(state_js)

    delta_state = goal_xyz - state_xyz

    return delta_state

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
      [self.goal_xyz[0]],
      [self.goal_xyz[1]],
      [self.goal_xyz[2]],
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

class RewardCallback(BaseCallback):
    def __init__(self, eval_env, num_episodes=10, plot_interval=10000, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.rewards = []
        self.success_time_steps = []
        self.reward_time_steps = []
        self.success_rates = []
        self.eval_env = eval_env
        self.plot_interval = plot_interval
        self.num_episodes = num_episodes


    def _on_step(self) -> bool:
        # Get the current time step and reward
        current_time_step = self.num_timesteps
        self.reward_time_steps.append(current_time_step)

        episode_info = self.model.ep_info_buffer
        rewards = [ep_info['r'] for ep_info in episode_info]

        self.rewards.append(np.mean(rewards))

        if self.n_calls % self.plot_interval == 0:
          success_rate = self._evaluate_success_rate()
          self.success_rates.append(success_rate)
          self.success_time_steps.append(self.num_timesteps)
          
        return True

    def _on_training_end(self) -> None:
        # Plot the rewards over time steps
        self._plot_success_rates()

        plt.plot(self.reward_time_steps, self.rewards)
        plt.xlabel('Time Steps')
        plt.ylabel('Rewards')
        plt.title('Rewards over Time Steps')
      
        plt.savefig('./ppo_rewards/ppo_rewards_standard_4_2')
        plt.close()

    def _evaluate_success_rate(self):
        all_episode_successes = []

        for i in range(self.num_episodes):
            done = False
            obs = env.reset()[0]
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, info = env.step(action)
            all_episode_successes.append(info['is_success'])
        success_rate = 1.0 * sum(all_episode_successes) / self.num_episodes
        return success_rate

    def _plot_success_rates(self):
        plt.plot(self.success_time_steps, self.success_rates)
        plt.xlabel('Timesteps')
        plt.ylabel('Success Rate')
        plt.title('Success Rate over Timesteps')
        plt.grid(True)
        plt.savefig('./ppo_success/ppo_success_standard_4_2')
        plt.close()

class ProgressBarCallback(BaseCallback):
    """
    :param pbar: (tqdm.pbar) Progress bar object
    """

    def __init__(self, pbar):
        super().__init__()
        self._pbar = pbar

    def _on_step(self):
        # Update the progress bar:
        self._pbar.n = self.num_timesteps
        self._pbar.update(0)


# this callback uses the 'with' block, allowing for correct initialisation and destruction
class ProgressBarManager(object):
    def __init__(self, total_timesteps):  # init object with total timesteps
        self.pbar = None
        self.total_timesteps = total_timesteps

    def __enter__(self):  # create the progress bar and callback, return the callback
        self.pbar = tqdm(total=self.total_timesteps)

        return ProgressBarCallback(self.pbar)

    def __exit__(self, exc_type, exc_val, exc_tb):  # close the callback
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

def load_ik_agent():
    env_id = "HalControllerEnv"
    register(id=env_id,entry_point=HalControllerEnv,max_episode_steps=500)

    # Create the RewardCallback
    env = gymnasium.make("HalControllerEnv")

    model = HAL_Agent(ppo_agent=None, env=env)
    return model

# if __name__ == "__main__":

#     env_id = "HalControllerEnv"
#     register(id=env_id,entry_point=HalControllerEnv,max_episode_steps=500)

#     # Create the RewardCallback
#     env = gymnasium.make("HalControllerEnv")

#     model = HAL_Agent(ppo_agent=None, env=env)
#     # model=None
#     plot_save_dir = args.plot_save_dir
#     eval_episodes = 100
#     reset_mode = "recovery"
#     success_rate, state_list, action_list, goal = evaluation_utils.evaluate(model, env, eval_episodes, plot_save_dir, reset_mode)
#     # print(save_path)
#     print("Success Rate: ", success_rate)

