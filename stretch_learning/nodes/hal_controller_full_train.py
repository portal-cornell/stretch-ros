# -*- coding: utf-8 -*-

# import packages
import sys
import numpy as np
import math
import torch
import gymnasium
from gymnasium import core, spaces
import matplotlib.pyplot as plt
from shapely.geometry import LineString

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
import optuna

from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances
from typing import Any, Dict
import torch.nn as nn
from stable_baselines3 import PPO

class HalControllerEnv(gymnasium.Env):

  def __init__(self,max_steps=500):

    # joints
    self.s_high = np.array([0.457, 1.5, 1.1, 0, 0, 0]) 
    self.s_low  = np.array([0.0025, -1.3837,  0.1538, 0, 0, 0])

    self.base_high = np.array([0.35, 0.12, np.pi / 8])
    self.base_low = np.array([-0.35, -0.12, -np.pi / 8])

    # xyz
    self.s_xyz_high = np.array([0.5, 0.75, 1.1])
    self.s_xyz_low = np.array([-0.5, -0.75, 0.15])
    # self.s_xyz_high = np.array([1.22, 0.76, 1.1])
    # self.s_xyz_low = np.array([-1.22, -0.76, 0.15])

    self.g_high = np.array([0.44, 1.4, 1.0, 0.2, 0.075,  np.pi / 10])
    self.g_low  = np.array([0.01, -1.3,  0.16, -0.2, -0.075, -np.pi / 10])
    # self.g_high = np.array([0.44, 1.4, 1, 0, 0, 0])
    # self.g_low  = np.array([0.01, -1.3, 0.16, 0, 0, 0])


    self.s_dims = 6 # ext, yaw, lift, curr_x, curr_y, base angle
    self.s_xyz_dims = 3 # xyz
    self.u_dims = 1 # keypress
    self.g_dims = 6 # joint state

    self.sparse_reward = False
    self.fixed_start = False
    self.eps = 0.02
    self.eps_circle = 0.01
    self.eps_z = 0.02
    self.prop_steps = 1
    self.center_offset = 0

    # bin rewards
    self.t_delta_y = 0.05
    self.t_delta_in_x = 0.020
    self.t_delta_out_x = 0.020

    # bad rect rewards
    self.bad_x_delta = 0.04
    self.reward_bad_rect = -15
    
    # trapezoid rewards
    self.t_delta_y_trap = 0.20

    self.reward_sparse_goal = 100
    self.reward_sparse_z = -10
    self.reward_sparse_rect = -1
    self.reward_sparse_default = -50
    self.reward_dist = 25

    self.reward_exceed_rotation = -20
    self.max_base_angle = np.pi / 10

    self.dist_threshold = 0.085
    self.base_threshold = 0.1

    self.base_reward = -25
    self.grid_reward = -25

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

    xyz_bound = self.s_xyz_high - self.s_xyz_low
    self.observation_space = spaces.Box(low=-xyz_bound, high=xyz_bound, shape=(3, ), dtype=np.float32)

    self.action_space = spaces.Discrete(10)

    self.max_steps = max_steps
    self.reset()

  def _get_delta_state(self):
    return self.delta_state.astype(np.float32)


  def reset(self, seed=0, start = None, goal = None):
    self.steps = 0

    self.state_js = np.zeros((self.s_dims,), dtype=np.float32)
    self.state_xyz = np.zeros((self.s_xyz_dims,), dtype=np.float32)
    self.goal = np.zeros((self.g_dims,), dtype=np.float32)
    self.goal_xyz = self.convert_js_xyz(np.copy(self.goal))
    self.delta_state = self.goal_xyz - self.state_xyz

    max_tries = 10
    tries = 0
    if start is None:
      self.state_js = np.random.uniform(self.s_low, self.s_high)
      self.goal = np.random.uniform(self.g_low, self.g_high)
      self.state_xyz = self.convert_js_xyz(np.copy(self.state_js))
      self.goal_xyz = self.convert_js_xyz(np.copy(self.goal))

      self.state_js = np.float32(self.state_js)
      self.goal = np.float32(self.goal)
      self.state_xyz = np.float32(self.state_xyz)
      self.goal_xyz = np.float32(self.goal_xyz)
      
      while self.goal_xyz[1] <= self.state_xyz[1] and tries < max_tries:
        self.state_js = np.random.uniform(self.s_low, self.s_high)
        self.goal = np.random.uniform(self.g_low, self.g_high)
        self.state_xyz = self.convert_js_xyz(np.copy(self.state_js))
        self.goal_xyz = self.convert_js_xyz(np.copy(self.goal))

        self.state_js = np.float32(self.state_js)
        self.goal = np.float32(self.goal)
        self.state_xyz = np.float32(self.state_xyz)
        self.goal_xyz = np.float32(self.goal_xyz)
        
        tries += 1

      self.delta_state = self.goal_xyz - self.state_xyz
    else:
      self.state_js = np.array(start,dtype = np.float32)
      self.state_xyz = self.convert_js_xyz(self.state_js)
      self.goal = np.array(goal,dtype = np.float32)
      self.goal_xyz = self.convert_js_xyz(np.copy(self.goal))
      self.delta_state = self.goal_xyz - self.state_xyz

    return (self._get_delta_state(),{})


  def _enforce_bounds(self,s):
    for i in range(self.s_dims):
      s[i] = np.clip(s[i],self.s_low[i],self.s_high[i])
    return s

  def get_rect_line_segments(self, coords):
    a = LineString([coords[0], coords[1]])
    b = LineString([coords[0], coords[2]])
    c = LineString([coords[1], coords[3]])
    return [a, b, c]

  def get_rect_coords(self, goal):
    goal_x = goal[0]
    goal_y = goal[1]
    goal_z = goal[2]

    x1 = goal_x - self.t_delta_in_x
    y1 = goal_y + self.t_delta_y
    x2 = goal_x + self.t_delta_in_x
    y2 = goal_y + self.t_delta_y
    x3 = goal_x - self.t_delta_out_x
    y3 = goal_y - self.t_delta_y
    x4 = goal_x + self.t_delta_out_x
    y4 = goal_y - self.t_delta_y

    return [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

  def get_obj_bin(self, coords, angle, pivot):

    angle = math.radians(angle)  

    x1 = math.cos(angle)*(coords[0][0]-pivot[0]) - math.sin(angle)*(coords[0][1]-pivot[1]) + pivot[0]
    y1 = math.sin(angle)*(coords[0][0]-pivot[0]) + math.cos(angle)*(coords[0][1]-pivot[1]) + pivot[1]

    x2 = math.cos(angle)*(coords[1][0]-pivot[0]) - math.sin(angle)*(coords[1][1]-pivot[1]) + pivot[0] 
    y2 = math.sin(angle)*(coords[1][0]-pivot[0]) + math.cos(angle)*(coords[1][1]-pivot[1]) + pivot[1]

    x3 = math.cos(angle)*(coords[2][0]-pivot[0]) - math.sin(angle)*(coords[2][1]-pivot[1]) + pivot[0]
    y3 = math.sin(angle)*(coords[2][0]-pivot[0]) + math.cos(angle)*(coords[2][1]-pivot[1]) + pivot[1] 

    x4 = math.cos(angle)*(coords[3][0]-pivot[0]) - math.sin(angle)*(coords[3][1]-pivot[1]) + pivot[0]
    y4 = math.sin(angle)*(coords[3][0]-pivot[0]) + math.cos(angle)*(coords[3][1]-pivot[1]) + pivot[1]

    return [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]

  def grid_valid(self, goal_xyz, robot_theta, js, new_js):
    rect_coords = self.get_rect_coords(goal_xyz)
    pivot = (goal_xyz[0], goal_xyz[1])
    obj_bin = self.get_obj_bin(rect_coords, robot_theta, pivot)
    rect_line_segments = self.get_rect_line_segments(obj_bin)

    state_xyz_before = self.convert_js_xyz(js)
    state_xyz_after = self.convert_js_xyz(new_js)
    coord1 = (state_xyz_before[0], state_xyz_before[1])
    coord2 = (state_xyz_after[0], state_xyz_after[1])
    a = LineString([coord1, coord2])

    valid = True
    for i in rect_line_segments:
      valid = valid and not i.intersects(a)

    return valid


  def is_valid(self, js):
    """
    js: current [ext, yaw, lift, curr_x, curr_y, base_angle]
    delta: delta [ext, yaw, lift, curr_x, curr_y, base_angle]
    """
    valid_js = False
    # joint_states
    if js[0] <= self.s_high[0] and \
      js[0] >= self.s_low[0] and \
      js[1] <= self.s_high[1] and \
      js[1] >= self.s_low[1] and \
      js[2] <= self.s_high[2] and \
      js[2] >= self.s_low[2] and \
      js[3] <= self.base_high[0] and \
      js[3] >= self.base_low[0] and \
      js[4] <= self.base_high[1] and \
      js[4] >= self.base_low[1] and \
      js[5] <= self.base_high[2] and \
      js[5] >= self.base_low[2]:
      valid_js = True
    return valid_js

  def update_dynamics(self, js, delta):
    """
    js: [curr_ext, curr_yaw, curr_lift, curr_x, curr_y, base_rot]
    deltas: [delta_ext, delta_yaw, delta_lift, delta_base, base_rot]
    returns: updated state
    """
    js_valid = False
    grid_valid = False
    js_new = np.copy(js)
    js_final = np.copy(js)

    # update the joint states with delta
    js_new[0] = js[0] + delta[0]
    js_new[1] = js[1] + delta[1]
    js_new[2] = js[2] + delta[2]
    js_new[3] = js[3] - delta[3] * np.cos(js[5]).item() # - because positive base trans goes left
    js_new[4] = js[4] - delta[3] * np.sin(js[5]).item() # - because positive base trans goes left
    js_new[5] = (js[5] + delta[4] + np.pi) % (2 * np.pi) - np.pi # constrains angle to be within -pi ~ pi

    if self.is_valid(js_new):
        js_valid = True
        
        if self.grid_valid(self.goal_xyz, js_new[5], js, js_new):
          grid_valid = True
          js_final = js_new
    return js_final, js_valid, grid_valid

  def convert_js_xyz(self, joint_state):
      extension = joint_state[0]
      yaw = joint_state[1]
      lift = joint_state[2]
      base_x = joint_state[3]
      base_y = joint_state[4]
      base_angle = joint_state[5]

      gripper_len = 0.27
      base_gripper_yaw = -0.09

      # find cx, cy in base frame
      point = (0.03, 0.17)
      pivot = (0, 0)
      cx, cy = self.rotate_odom(point, base_angle, pivot)

      # cx, cy in origin frame
      cx += base_x
      cy += base_y

      extension_y_offset = extension * np.cos(base_angle)
      extension_x_offset = extension * -np.sin(base_angle)
      yaw_delta = yaw - base_gripper_yaw
      gripper_y_offset = gripper_len * np.cos(yaw_delta + base_angle)
      gripper_x_offset = gripper_len * -np.sin(yaw_delta + base_angle)

      x = cx + extension_x_offset + gripper_x_offset
      y = cy + extension_y_offset + gripper_y_offset
      z = lift

      return np.array([x.item(), y.item(), z])

  def rotate_odom(self, coord, angle, pivot):
      x1 = (
          math.cos(angle) * (coord[0] - pivot[0])
          - math.sin(angle) * (coord[1] - pivot[1])
          + pivot[0]
      )
      y1 = (
          math.sin(angle) * (coord[0] - pivot[0])
          + math.cos(angle) * (coord[1] - pivot[1])
          + pivot[1]
      )

      return (x1, y1)


  def convert_coordinates(self, point, angle):
      # Create the rotation matrix using the angle
      rotation_matrix = np.array(
          [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
      )

      # Convert the point to the new coordinate system
      new_point = np.dot(rotation_matrix, point)

      if np.sign(new_point[0]) >= 0:
          return True
      else:
          return False

  def is_point_in_half_circle(self, rotation_angle, center, radius, test_point, center_offset):
      rotated_angle = np.copy(rotation_angle)
      rotated_angle += np.pi / 2
      unit_vector = np.array([math.cos(rotated_angle), math.sin(rotated_angle)])
      center_copy = np.copy(center)
      center_copy += center_offset * unit_vector

      # Translate the test point coordinates relative to the center of the circle
      translated_point = [
          test_point[0] - center_copy[0],
          test_point[1] - center_copy[1],
      ]

      # Calculate the projection of the translated point onto a vector defined by the rotation angle
      projection = self.convert_coordinates(translated_point, rotated_angle)
    
      if projection and np.linalg.norm(translated_point) <= radius:
          return True
      else:
          return False
    
  def _terminal(self, robot_angle, s_xyz, goal_xyz):
    in_z = np.abs(goal_xyz[2] - s_xyz[2]) <= self.eps_z
    in_half = self.is_point_in_half_circle(robot_angle, goal_xyz[:2], self.eps, s_xyz[:2], self.center_offset)
    in_circle = np.linalg.norm(goal_xyz[:2] - s_xyz[:2]) <= self.eps_circle
    no_steps_left = self.steps >= self.max_steps
    if (in_half or in_circle) and in_z: return True, True
    if no_steps_left: return True, False
    return False, False


  def calculate_area(self, p1, p2, p3):
    return 0.5 * abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))

  def get_trapezoid_vertices(self, goal):
    goal_x = goal[0]
    goal_y = goal[1]
    goal_z = goal[2]

    x1 = goal_x - self.t_delta_in_x
    y1 = goal_y
    x2 = goal_x + self.t_delta_in_x
    y2 = goal_y
    x3 = goal_x - self.t_delta_out_x
    y3 = goal_y - self.t_delta_y_trap
    x4 = goal_x + self.t_delta_out_x
    y4 = goal_y - self.t_delta_y_trap

    return [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

  def in_trapezoid(self, vertices, point):
    
    p1, p2, p3, p4 = vertices
    area_trapezoid = self.calculate_area(p1, p2, p3) + self.calculate_area(p1, p3, p4)

    area_triangle1 = self.calculate_area(p1, p2, point)
    area_triangle2 = self.calculate_area(p2, p3, point)
    area_triangle3 = self.calculate_area(p3, p4, point)
    area_triangle4 = self.calculate_area(p1, p4, point)

    return abs(area_triangle1 + area_triangle2 - area_trapezoid) < 1e-6

  def in_trap_reward_area(self, test_point, goal, robot_theta):
    trap_vertices = self.get_trapezoid_vertices(goal)

    pivot = (goal[0], goal[1])
    trapezoid_vertices = self.get_obj_bin(trap_vertices, robot_theta, pivot)

    test_point_xy = [test_point[0], test_point[1]]
    if (
      self.in_trapezoid(trapezoid_vertices, test_point_xy)
    ):
      return True
    else:
      return False

  def get_bad_rects(self, goal_xyz, robot_theta):
    rect_coords = self.get_rect_coords(goal_xyz)
    
    rect1_coord_1 = (rect_coords[0][0] - self.bad_x_delta, rect_coords[0][1])
    rect1_coord_2 = rect_coords[0]
    rect1_coord_3 = (rect_coords[2][0] - self.bad_x_delta, rect_coords[2][0])
    rect1_coord_4 = rect_coords[2]
    rect1_coords = [rect1_coord_1, rect1_coord_2, rect1_coord_3, rect1_coord_4]

    rect2_coord_1 = rect_coords[1]
    rect2_coord_2 = (rect_coords[1][0] + self.bad_x_delta, rect_coords[1][1])
    rect2_coord_3 = rect_coords[3]
    rect2_coord_4 = (rect_coords[3][0] + self.bad_x_delta, rect_coords[3][1])
    rect2_coords = [rect2_coord_1, rect2_coord_2, rect2_coord_3, rect2_coord_4]

    pivot = (goal_xyz[0], goal_xyz[1])
    rotated_rect1_coords = self.get_obj_bin(rect1_coords, robot_theta, pivot)
    rotated_rect2_coords = self.get_obj_bin(rect2_coords, robot_theta, pivot)

    return rotated_rect1_coords, rotated_rect2_coords

  def in_bad_rects(self, test_point, goal_xyz, robot_theta):
    rect1_coords, rect2_coords = self.get_bad_rects(goal_xyz, robot_theta)
    if self.in_trapezoid(rect1_coords, test_point) or self.in_trapezoid(rect2_coords, test_point):
      return True
    else:
      return False
    
  def compute_reward_special(self, achieved_goal, goal, state_js, grid_valid, action):

    rotation_angle = state_js[5]
    center = goal[:2]
    radius = self.eps
    test_point = achieved_goal[:2]
    center_offset = self.center_offset
    reward = self.reward_sparse_default

    # trying to get through goal blinder
    grid_reward = 0
    if not grid_valid:
      grid_reward = self.grid_reward

    # z rewards
    in_z = np.abs(goal[2] - achieved_goal[2]) <= self.eps_z
    if in_z:
      reward = self.reward_sparse_z
      
    # bad rect rewards
    bad_rect_reward = 0
    if self.in_bad_rects(test_point, goal, state_js[5]):
      bad_rect_reward = self.reward_bad_rect

    # base rotation rewards
    rotation_reward = 0
    if np.abs(state_js[5]) > self.max_base_angle or np.abs(state_js[3]) > 0.3 or np.abs(state_js[4]) > 0.1:
      rotation_reward = self.reward_exceed_rotation
      
    # discourage base movements in sphere
    close_base_reward = 0
    if np.linalg.norm(achieved_goal - goal) <= self.base_threshold:
      if action == 6 or action == 7 or action == 8 or action == 9:
        base_reward = self.base_reward
    
    # rectangle rewards
    if self.in_trap_reward_area(achieved_goal, goal, rotation_angle) and in_z:
      reward = self.reward_sparse_rect

    # goal rewards
    in_hemisphere = self.is_point_in_half_circle(rotation_angle, center, radius, test_point, center_offset)
    in_circle = np.linalg.norm(center - test_point) <= self.eps_circle
    if (in_hemisphere or in_circle) and in_z:
      reward = self.reward_sparse_goal

    if self.sparse_reward:
      return reward + rotation_reward + bad_rect_reward + grid_reward
    else:
      dist_reward = 0
      if np.linalg.norm(achieved_goal - goal) >= self.dist_threshold:
        dist_reward = -np.linalg.norm(achieved_goal - goal) * self.reward_dist
      
      return dist_reward + reward + bad_rect_reward + rotation_reward

  def step(self,a):
    s = self.state_js

    u = np.copy(a)
    u = np.array(self.kp_delta_mapping[int(u)]) * (3/5)
    goal_xyz = np.copy(self.goal_xyz)
    for i in range(self.prop_steps):
      s, js_valid, grid_valid = self.update_dynamics(s,u)


    self.state_js = s
    self.state_xyz = self.convert_js_xyz(self.state_js)
    self.delta_state = goal_xyz - self.state_xyz

    terminal, success = self._terminal(self.state_js[5], self.state_xyz, goal_xyz)
    
    reward = self.compute_reward_special(self.state_xyz, goal_xyz, self.state_js, grid_valid, a)


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
    def __init__(self, eval_env, num_episodes=20, plot_interval=100, verbose=0):
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
      
        plt.savefig('/share/portal/nlc62/ppo_data/ppo_rewards/rewards_ws_57_big_tanh')
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
        plt.savefig('/share/portal/nlc62/ppo_data/ppo_success/success_ws_57_big_tanh')
        plt.close()


def plot_traj(state_list, action_list, goal, episode):
  kp_mapping = {
    0: "Arm out",
    1: "Arm in",
    2: "Gripper right",
    3: "Gripper left",
    4: "Arm up",
    5: "Arm down",
    6: "Base forward (left)", 
    7: "Base backward (right)",
    8: "Base Rotate Left", 
    9: "Base Rotate Right"

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
  ax.set_ylim(-0.68, 0.68)
  ax.set_zlim(-1, 1)

  ax.set_xlabel('Position X (m)')
  ax.set_ylabel('Position Y (m)')
  ax.set_zlabel('Position Z (m)')
  plt.title("HAL Controller Sim with PPO")

  plt.savefig("/home/strech/catkin_ws/src/stretch_ros/stretch_learning/nodes/plots/base_ppo/eval_ex_52_" + str(episode))

  plt.close()

def evaluate(model, env, start=None, goal=None, num_episodes=1):
    all_episode_successes = []
    for i in range(num_episodes):
      state_list, action_list = [], []
      episode_rewards = []
      done = False
      obs = env.reset(start=start, goal=goal)[0] # start=[0.2,-0.09,0.5,0, 0, 0],goal=[0.389,0.691,0.851, 0, 0, 0]
      # obs = obs[0]
      state_list.append(obs)
      # goal = obs['desired_goal']
      goal = env.goal_xyz
      while not done:
        obs = torch.tensor(obs)
        print(obs)
        pred = model(obs)
        action = torch.argmax(pred).item()
        action_list.append(action)
        obs, reward, done, _, info = env.step(action)
        state_list.append(obs)
        # @aravind: There has to be a way of getting the success directly from the environment.
      all_episode_successes.append(info['is_success'])
      plot_traj(state_list, action_list, [0, 0, 0], i)

    success_rate = 1.0 * sum(all_episode_successes)/num_episodes
    return success_rate, state_list, action_list, goal

from gymnasium.envs.registration import register
register(id="HalControllerEnv",entry_point=HalControllerEnv,max_episode_steps=500)

"""Training Policy"""

from stable_baselines3 import PPO

num_cpu = 16
env_id = "HalControllerEnv"

vec_env = make_vec_env(env_id, n_envs=num_cpu)
# env = gymnasium.make("HalControllerEnv")

# Create the PPO model

# sparse
# 3.7: {'gamma': 0.07014402469935861, 'max_grad_norm': 2.5777062645397173, 'exponent_n_steps': 7, 'learning_rate': 1.8117726147884592e-05, 'gae_lambda': 0.0013221280357480253, 'ent_coef': 0.045403657580233014, 'net_arch': 'small', 'act_func': 'tanh'}
# 2.4: {'gamma': 0.007564196707558446, 'max_grad_norm': 0.3749094204638814, 'exponent_n_steps': 3, 'learning_rate': 1.0392770462903161e-05, 'gae_lambda': 0.000820292414273172, 'ent_coef': 0.00911617857452049, 'net_arch': 'big', 'act_func': 'tanh'}
# 5: {'gamma': 0.08072210939801787, 'max_grad_norm': 1.8898858376696066, 'exponent_n_steps': 7, 'learning_rate': 2.052866608973938e-05, 'gae_lambda': 0.00016254510231109498, 'ent_coef': 0.009366128789784711, 'net_arch': 'small', 'act_func': 'tanh'}
# 4.7: {'gamma': 0.04504971073790163, 'max_grad_norm': 0.367608705708055, 'exponent_n_steps': 4, 'learning_rate': 0.00012578130543058395, 'gae_lambda': 0.008442598164832036, 'ent_coef': 0.025448815549483313, 'net_arch': 'small', 'act_func': 'relu'}

# 10000: {'gamma': 0.01422455636575307, 'max_grad_norm': 0.7031076887649994, 'exponent_n_steps': 4, 'learning_rate': 8.847792035890729e-05, 'gae_lambda': 0.003818155614091569, 'ent_coef': 0.12669444682406672, 'net_arch': 'big', 'act_func': 'tanh'}
# 11000: {'gamma': 0.09984138561308638, 'max_grad_norm': 3.499957998088178, 'exponent_n_steps': 5, 'learning_rate': 0.00014012474040704737, 'gae_lambda': 0.00011688963898406382, 'ent_coef': 0.03975368478353873, 'net_arch': 'small', 'act_func': 'relu'}
# 14000: {'gamma': 0.0967076182828145, 'max_grad_norm': 2.666165675885895, 'exponent_n_steps': 3, 'learning_rate': 5.831793736040651e-05, 'gae_lambda': 0.013913428735569389, 'ent_coef': 0.03150917098782848, 'net_arch': 'small', 'act_func': 'relu'}
# 6000: {'gamma': 0.08493256473246337, 'max_grad_norm': 0.8319399927557504, 'exponent_n_steps': 8, 'learning_rate': 0.0002290286274942187, 'gae_lambda': 0.00011356329839306686, 'ent_coef': 0.19882160844882035, 'net_arch': 'big', 'act_func': 'tanh'}
# 6000: {'gamma': 0.09578481880172345, 'max_grad_norm': 4.930536165039469, 'exponent_n_steps': 7, 'learning_rate': 0.00021169039391886468, 'gae_lambda': 0.0011270005336213862, 'ent_coef': 0.13930554139003642, 'net_arch': 'big', 'act_func': 'tanh'}
# 5700: {'gamma': 0.05754178181807216, 'max_grad_norm': 3.356820892178437, 'exponent_n_steps': 8, 'learning_rate': 0.0005329213151863372, 'gae_lambda': 0.004873909911799735, 'ent_coef': 0.14021034737641483, 'net_arch': 'big', 'act_func': 'tanh'}
# 5400: {'gamma': 0.07063805779991296, 'max_grad_norm': 1.0350677710637386, 'exponent_n_steps': 6, 'learning_rate': 0.00016141590733342257, 'gae_lambda': 0.0003207402234448613, 'ent_coef': 0.06956028814780876, 'net_arch': 'small', 'act_func': 'tanh'}
# 5000: {'gamma': 0.02707515445567827, 'max_grad_norm': 3.281715907665024, 'exponent_n_steps': 4, 'learning_rate': 5.417326653586055e-05, 'gae_lambda': 0.0003848930762401007, 'ent_coef': 0.1880018212352874, 'net_arch': 'big', 'act_func': 'tanh'}
# 5200: {'gamma': 0.09409853864014396, 'max_grad_norm': 3.3337056210870504, 'exponent_n_steps': 7, 'learning_rate': 0.00016981455758522608, 'gae_lambda': 0.06304877427203286, 'ent_coef': 0.14937352405443094, 'net_arch': 'small', 'act_func': 'relu'}

hyperparameters = {
  'gamma': (1 - 0.09409853864014396), 
  'max_grad_norm': 3.3337056210870504, 
  'n_steps':2**7, 
  'learning_rate': 0.00016981455758522608, 
  'gae_lambda': (1 - 0.06304877427203286), 
  'ent_coef': 0.14937352405443094, 
  "policy_kwargs": {
    "net_arch": {"pi": [256, 256, 256], "vf": [256, 256, 256], 
    "activation_fn":'relu'}
  },
}
# model = PPO("MlpPolicy", vec_env, verbose=1, device="cuda", **hyperparameters)

# Create the RewardCallback
env = gymnasium.make("HalControllerEnv")
reward_callback = RewardCallback(env)

# model = PPO.load("/share/portal/nlc62/hal-skill-repo/ppo_delta_3d/ppo_fixed_800_big_tanh.zip", env=vec_env)

# model.gamma = 1 - 0.05754178181807216
# model.max_grad_norm = 3.356820892178437
# model.n_steps = 2**6
# model.learning_rate = 0.0005329213151863372
# model.gae_lambda = 1 - 0.004873909911799735
# model.ent_coef = 0.14021034737641483
new_hyperparameters = {
  'gamma': (1 - 0.05754178181807216), 
  'max_grad_norm': 3.356820892178437, 
  'n_steps':2**6, 
  'learning_rate': 0.0005329213151863372, 
  'gae_lambda': (1 - 0.004873909911799735), 
  'ent_coef': 0.14021034737641483, 
  "policy_kwargs": {
    "net_arch": {"pi": [64, 64, 64], "vf": [64, 64, 64], 
    "activation_fn":'tanh'}
  },
}

# Create a new PPO instance with the desired hyperparameters
# new_model = PPO(policy=model.policy, env=env, **new_hyperparameters)

# sys.exit()

# Train the model with the callback
# model.learn(total_timesteps=7000000, callback=reward_callback)

# model.save("./ppo_ws_57_big_tanh")

# model = model.load("/share/portal/nlc62/hal-skill-repo/ppo_delta_3d/ppo_nobase_57_big_tanh.zip")

# env = gymnasium.make("HalControllerEnv")
# success_rate, state_list, action_list, goal = evaluate(model, env, 100)
# print("Success Rate: ",success_rate)
# success_rate, state_list, action_list, goal = evaluate(model, env, 1)
# plot_traj(state_list, action_list, goal)