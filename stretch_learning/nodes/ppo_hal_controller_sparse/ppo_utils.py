import torch
import torch.nn as nn
import re
import numpy as np
import matplotlib.pyplot as plt
import ppo

config = {
    "input_size": 3,
    "output_size": 6,
    "hidden_dim": 64,
    "num_layers": 3,
    "activation": nn.Tanh,
}


def is_valid(js, delta):
    """
    js: current [ext, yaw, lift]
    delta: delta [ext, yaw, lift]
    """
    s_high = np.array([0.457, 1.5, 1.1])  # 4.586
    s_low = np.array([0.0025, -1.3837, 0.1538])
    if (
        js[0] + delta[0] <= s_high[0]
        and js[0] + delta[0] >= s_low[0]
        and js[1] + delta[1] <= s_high[1]
        and js[1] + delta[1] >= s_low[1]
        and js[2] + delta[2] <= s_high[2]
        and js[2] + delta[2] >= s_low[2]
    ):
        return True
    else:
        return False


def update_dynamics(js, delta):
    """
    js: [curr_ext, curr_yaw, curr_lift]
    deltas: [delta_ext, delta_yaw, delta_lift]
    returns: updated state
    """
    valid_action = False
    if is_valid(js, delta):
        valid_action = True
        js[0] = js[0] + delta[0]
        js[1] = js[1] + delta[1]
        js[2] = js[2] + delta[2]

    return js, valid_action


def convert_js_xyz(joint_state):
    extension = joint_state[0]
    yaw = joint_state[1]
    lift = joint_state[2]

    gripper_len = 0.22
    base_gripper_yaw = -0.09
    yaw_delta = -(yaw - base_gripper_yaw)  # going right is more negative
    y = gripper_len * torch.cos(torch.tensor([yaw_delta])) + extension
    x = gripper_len * torch.sin(torch.tensor([yaw_delta]))
    z = lift

    return [x, y, z]


def plot_traj(state_list, action_list, goal, episode):
    kp_mapping = {
        0: "Arm out",
        1: "Arm in",
        2: "Gripper right",
        3: "Gripper left",
        4: "Arm up",
        5: "Arm down",
    }
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.grid()

    s = np.array(state_list)
    print(s[0].shape)
    plot_x = s[:, 0]
    plot_y = s[:, 1]
    plot_z = s[:, 2]
    sim_scatter = ax.scatter(
        plot_x[:-1],
        plot_y[:-1],
        plot_z[:-1],
        s=5,
        c=action_list,
        cmap="viridis",
        alpha=1,
    )
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

    ax.set_xlabel("Position X (m)")
    ax.set_ylabel("Position Y (m)")
    ax.set_zlabel("Position Z (m)")
    plt.title("HAL Controller Sim with PPO")
    # plt.title(name)

    plt.savefig("PPO_eval_2400_small" + str(episode))

    plt.close()


def evaluate(model, starts, goals, num_episodes=1, max_steps=600):
    kp_delta_mapping = {
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
    eps = 0.03

    all_episode_successes = []
    for i in range(len(starts)):
        start = starts[i]
        goal = goals[i]
        print(f"start:{start},goal:{goal}")
        start_xyz = torch.tensor(convert_js_xyz(torch.tensor(start)))
        goal_xyz = torch.tensor(convert_js_xyz(torch.tensor(goal)))
        state_list, action_list = [], []
        episode_rewards = []
        done = False
        steps = 0
        state_list.append(list(goal_xyz - start_xyz))
        while (not done) and (steps < max_steps):
            inp = torch.FloatTensor(goal_xyz - start_xyz).unsqueeze(0)
            action = model(inp)
            keypressed_index = torch.argmax(action).item()
            deltas = kp_delta_mapping[keypressed_index]

            action_list.append(keypressed_index)
            start, _ = update_dynamics(start, deltas)
            start_xyz = torch.tensor(convert_js_xyz(start))

            state_list.append(list(goal_xyz - start_xyz))

            if torch.norm(goal_xyz - start_xyz) < eps:
                done = True
            steps += 1
        plot_traj(state_list, action_list, [0, 0, 0], i)
        print(f"Success: {done}")

    # success_rate = 1.0 * sum(all_episode_successes) / num_episodes
    return


def load_pth_file_to_model(model, path="policy.pth"):
    state_dict = torch.load(
        "policy.pth", map_location="cuda" if torch.cuda.is_available() else "cpu"
    )

    r = ".*policy_net.*"
    pattern = re.compile(r)
    result = {}
    for key in list(state_dict.keys()):
        val = state_dict[key]
        res = pattern.search(key)

        if res is not None:
            result[key] = val

    result["model.4.bias"] = state_dict["action_net.bias"]
    result["model.4.weight"] = state_dict["action_net.weight"]

    # model = MLP(**config)

    for key in list(result.keys()):
        if "model.4" in key:
            continue

        result[key.replace("mlp_extractor.policy_net", "model")] = result.pop(key)

    miss_keys, unexpected_keys = model.load_state_dict(result, strict=False)

    # print(miss_keys)
    # print(unexpected_keys)
    return model


def end_eff_to_xy(extension, yaw):
    # extension, yaw = deltas
    gripper_len = 0.22
    base_gripper_yaw = -0.09
    yaw_delta = -(yaw - base_gripper_yaw)  # going right is more negative
    x = gripper_len * np.sin(yaw_delta)
    y = gripper_len * np.cos(yaw_delta) + extension
    return x.item(), y.item()


model = ppo.MLP(**config)
model = load_pth_file_to_model(
    model,
    "/home/strech/catkin_ws/src/stretch_ros/stretch_learning/checkpoints/ppo_point_and_shoot/policy.pth",
)


def main():
    s_high = np.array([0.457, 1.5, 1.1])  # 4.586
    s_low = np.array([0.0025, -1.3837, 0.1538])
    starts = []
    goals = []
    for i in range(10):
        start = []
        goal = []
        for i in range(2):
            start.append(np.random.uniform(s_low[i], s_high[i]))
            goal.append(np.random.uniform(s_low[i], s_high[i]))
        start.append(0.6)
        goal.append(0.6)
        starts.append(start)
        goals.append(goal)

    evaluate(model, starts, goals, num_episodes=1)
    ext, yaw, lift = 0.27, 0.193, 0.92
    x, y = end_eff_to_xy(ext, yaw)
    z = lift
    print(f"current x y z: {x}, {y}, {z}")
    print(f"goal xyz: {x}, {y+0.1}, {z}")
    print(torch.softmax(model(torch.tensor([0, 0.1, 0])), dim=-1))


if __name__ == "__main__":
    main()
