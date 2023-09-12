from pathlib import Path
import time
import sys
import matplotlib.pyplot as plt
import matplotlib
import os
import torch

import numpy as np

from bc.model import BC
# from dataset import (gripper_len,base_gripper_yaw)
gripper_len = 0.22
base_gripper_yaw = -0.09

kp_delta_mapping = {
    # arm out
    0: [0.04, 0],
    # arm in
    1: [-0.04, 0],
    # gripper right
    2: [0, 0.010472],
    # gripper left
    3: [0, -0.010472],
}


def load_bc_model(ckpt_path, device):
    ckpt_path = Path(ckpt_path)
    print(f"Loading {ckpt_path.stem}")

    model = BC(is_2d=False,use_delta=True)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    return model


def convert_js_xy(extension,yaw):
    yaw_delta = -(yaw - base_gripper_yaw)  # going right is more negative
    y = gripper_len * np.cos(yaw_delta) + extension
    x = gripper_len * np.sin(yaw_delta)
    return x,y

def simulate(inp, model, iterations):
    """
    inp: [wrist extension, joint wrist yaw, delta wrist extension, delta joint wrist yaw] torch tensor
    model: takes inp and outputs [delta wrist extension, delta wrist yaw]
    wrist extension: lower (0.0025) upper (0.457)
    wrist yaw: (-1.3837785024051723, 4.585963397116449)
    """
    device = "cpu"
    inp_x = []
    inp_y = []
    goal_ext = inp[0,2]+inp[0,0]
    goal_yaw = inp[0,3]+inp[0,1]
    curr_x,curr_y = convert_js_xy(inp[0,0], inp[0,1])
    goal_x,goal_y = convert_js_xy(goal_ext, goal_yaw)

    inp_x.append(goal_x - curr_x)
    inp_y.append(goal_y - curr_y)
    temp_inp = inp.clone()
    delta_list = []
    onpolicy_kp = []
    for i in range(iterations):
        start = time.time()

        temp_inp = torch.tensor([[curr_x,curr_y,(goal_x-curr_x),(goal_y-curr_y)]])
        prediction = model(temp_inp)
        prediction = prediction.flatten()

        predicted_kp = torch.argmax(prediction).item()
        onpolicy_kp.append(predicted_kp)
        deltas = torch.Tensor(kp_delta_mapping[predicted_kp]).to(device)
        # joint limit constraints 
        if inp[0, 0] + deltas[0] < 0.457 and \
        inp[0, 0] + deltas[0] > 0.0025 and \
        inp[0, 1] + deltas[1] < 4.5859 and \
        inp[0, 1] + deltas[1] > -1.3837:
            inp[0, :2] += deltas
            inp[0, 2:] -= deltas

        curr_x,curr_y = convert_js_xy(inp[0,0], inp[0,1]) 

        delta_list.append(torch.norm(inp[0, 2:]).item())
        inp_x.append(goal_x - curr_x)
        inp_y.append(goal_y - curr_y)

    print(np.mean(delta_list))
    return inp_x, inp_y, min(delta_list), onpolicy_kp


def run_simulate(start_ext, start_yaw, goal_ext, goal_yaw, ckpt_path, iterations): 
    matplotlib.use("Agg")
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device="cpu"
    # base_folder = "./ckpts/20230904-005802_use_delta"
    print(device)
    # for ckpt_path in os.listdir(base_folder):
    
    epoch = ckpt_path.split("_")[0]
    # ckpt_path = r"C:\Users\adipk\Documents\HAL\hal-skill-repo\point_and_shoot_debug\ckpts\20230901-065332_use_delta\epoch=100_mean_deltas=0.111.pt"
    model = load_bc_model(Path(ckpt_path), device)

    # ext_min = 0.0025
    # ext_max = 0.457
    # yaw_min = -1.383
    # yaw_max = 4.586

    # ext_1 = torch.rand(1)
    # yaw_1 = torch.rand(1)

    # start_ext = (ext_min - ext_max) * ext_1 + ext_max
    # start_yaw = (yaw_min - yaw_max) * yaw_1 + yaw_max

    start_ext,start_yaw = torch.tensor([start_ext]) ,torch.tensor([start_yaw])
    start_tensor = torch.cat((start_ext, start_yaw))

    # ext_2 = torch.rand(1)
    # yaw_2 = torch.rand(1)

    # goal_ext = (ext_min - ext_max) * ext_2 + ext_max
    # goal_yaw = (yaw_min - yaw_max) * yaw_2 + yaw_max
    goal_ext,goal_yaw  = torch.tensor([goal_ext]), torch.tensor([goal_yaw])
    goal_tensor = torch.cat((goal_ext - start_ext, goal_yaw - start_yaw))

    start_tensor = start_tensor.to(device)
    goal_tensor = goal_tensor.to(device)
    inp = torch.cat((start_tensor, goal_tensor)).unsqueeze(0)

    # iterations = 100

    print(start_tensor)
    print([goal_ext, goal_yaw])

    inp_x, inp_y, delta_min,onpolicy_kp = simulate(inp, model, iterations)

    kp_mapping = ["Arm out", "Arm in", "Gripper right", "Gripper left"]
    onpolicy_kp = np.array(onpolicy_kp,dtype = np.int32)

    # scatter = plt.scatter(np.array(inp_x[1:]), -np.array(inp_y[1:]), c=onpolicy_kp, cmap='viridis', s=5, alpha=1)
    # handles, _ = scatter.legend_elements()
    # filtered_kp_mapping = [kp_mapping[i] for i in np.unique(onpolicy_kp)]

    
    # plt.legend(handles, filtered_kp_mapping, title="Key Presses")

    # plt.plot(inp_x, inp_y, marker=".", markersize=1, color="blue", label="trajectory")
    start_x,start_y = convert_js_xy(start_ext.item(),start_yaw.item())
    goal_x,goal_y = convert_js_xy(goal_ext.item(),goal_yaw.item())
    rel_x = (goal_x-start_x)
    rel_y = (goal_y-start_y)
    # plt.plot(
    #     [rel_x],
    #     [-rel_y],
    #     marker=".",
    #     markersize=15,
    #     color="green",
    #     label="start",
    # )
    # plt.plot(
    #     [0],
    #     [0],
    #     marker="*",
    #     markersize=15,
    #     color="red",
    #     label="goal",
    # )

    # plt.xlim(-0.6,0.8)
    # plt.ylim(-0.4,0.8)

    # plt.xlabel("Reative X")
    # plt.ylabel("Relative Y")

    # plt.title(f"Point and Shoot Simulator {epoch}")
    # plt.savefig(
    #     f'{plot_save_path}/'
    #     + str(delta_min)
    #     + ".png"
    # )
    # plt.show()
    # plt.close()

    return np.array(inp_x[1:]), np.array(inp_y[1:]), onpolicy_kp


if __name__ == '__main__': 
    matplotlib.use("Agg")
    # device = "cpu"
    start_ext = 0.35 
    start_yaw = 3.52 
    goal_ext = 0.02
    goal_yaw = -0.74 
    ckpt_path =  '/home/strech/catkin_ws/src/stretch_ros/stretch_learning/checkpoints/point_shoot/20230905-180606_use_delta/epoch=700_mean_deltas=0.021.pt'
    save_fig_path = '/home/strech/catkin_ws/src/stretch_ros/stretch_learning/nodes/plots/sep_11_graphs'
    run_simulate(start_ext, start_yaw, goal_ext, goal_yaw, ckpt_path, save_fig_path)