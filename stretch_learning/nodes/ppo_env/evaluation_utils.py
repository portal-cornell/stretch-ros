
import numpy as np
import matplotlib.pyplot as plt
from agent import HAL_Agent

def plot_traj(state_list, action_list, goal, episode, plot_save_dir):
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
  sim_scatter = ax.scatter(-plot_x[:-1], -plot_y[:-1], -plot_z[:-1], s=5, c=action_list, cmap='viridis', alpha=1)
  handles, _ = sim_scatter.legend_elements()

  filtered_kp_mapping = [kp_mapping[i] for i in np.unique(action_list)]
  plt.legend(handles, filtered_kp_mapping, title="Key Presses")

  # Plot the start
  plt.plot(
    [-plot_x[0]],
    [-plot_y[0]],
    [-plot_z[0]],
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
    [-plot_x[-1]],
    [-plot_y[-1]],
    [-plot_z[-1]],
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

  if plot_save_dir[-1] != "/":
    plot_save_dir += "/"
  save_filename = "eval_25_"
  save_path = plot_save_dir + save_filename

  plt.savefig(save_path + str(episode))

  plt.close()

def evaluate(ppo_agent, env, num_episodes, plot_save_dir, reset_mode="standard"):
  agent = HAL_Agent(ppo_agent, env)
  
  all_episode_successes = []
  for i in range(num_episodes):
    state_list, action_list = [], []
    episode_rewards = []
    done = False
    obs = env.reset(mode=reset_mode)[0]
    state_list.append(obs[:3])
    goal = env.goal_xyz
    while not done:
      action = agent.select_action(obs)
      action_list.append(action)
      obs, reward, done, _, info = env.step(action)
      state_list.append(obs[:3])

    all_episode_successes.append(info['is_success'])
    plot_traj(state_list, action_list, [0, 0, 0], i, plot_save_dir)

  success_rate = 1.0 * sum(all_episode_successes)/num_episodes
  return success_rate, state_list, action_list, goal

