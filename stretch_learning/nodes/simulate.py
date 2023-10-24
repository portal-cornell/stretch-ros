from pathlib import Path
import time
import sys
import matplotlib.pyplot as plt
import os
import torch

import numpy as np

from ablate_bc import BC
# from dataset import (gripper_len,base_gripper_yaw)
# from dataset import get_train_valid_split

kp_mapping = ["Arm out", "Arm in", "Gripper right", "Gripper left"]
kp_delta_mapping = {
    # arm out
    0: [0.04, 0],
    # arm in
    1: [-0.04, 0],
    # gripper right
    2: [0, -0.010472],
    # gripper left
    3: [0, 0.010472],
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
    gripper_len = 0.22
    base_gripper_yaw = -0.09
    yaw_delta = -(yaw - base_gripper_yaw)  # going right is more negative
    y = gripper_len * np.cos(yaw_delta) + extension
    x = gripper_len * np.sin(yaw_delta)
    return x,y

def run_simulate(inp, goal, model, iterations, is_2d, use_delta, is_xy):
    """
    inp: [wrist extension, joint wrist yaw, delta wrist extension, delta joint wrist yaw] torch tensor
    goal: [goal_ext, goal_yaw]
    model: takes inp and outputs [delta wrist extension, delta wrist yaw]
    wrist extension: lower (0.0025) upper (0.457)
    wrist yaw: (-1.3837785024051723, 4.585963397116449)
    """
    inp_x = []
    inp_y = []
    delta_list = []
    # goal_ext = inp[0,2]+inp[0,0]
    # goal_yaw = inp[0,3]+inp[0,1]
    if not is_xy:
        # curr_x,curr_y = convert_js_xy(inp[0,0], inp[0,1])
        # goal_x,goal_y = convert_js_xy(goal[0], goal[1])
        if is_2d:
            c_ext, c_yaw = -inp[0, 0] + goal[0], -inp[0, 1] + goal[1]
            c_x, c_y = convert_js_xy(c_ext, c_yaw)
            g_x, g_y = convert_js_xy(goal[0], goal[1])
            d_x, d_y = g_x - c_x, g_y - c_y

            # d_x, d_y = convert_js_xy(inp[0, 0], inp[0, 1])
        else:
            c_x, c_y = convert_js_xy(inp[0, 0], inp[0, 1])
            g_x, g_y = convert_js_xy(goal[0], goal[1])
            d_x, d_y = g_x - c_x, g_y - c_y
            
        delta_list.append(torch.norm(torch.tensor([d_x,d_y])).item())
        inp_x.append(d_x)
        inp_y.append(d_y)
    else:
        curr_x, curr_y = inp[0, 0], inp[0, 1]
        goal_x, goal_y = goal[0], goal[1]
        inp_x.append(goal_x - curr_x)
        inp_y.append(goal_y - curr_y)

    temp_inp = inp.clone()
    
    onpolicy_kp = []
    
    for i in range(iterations):
        start = time.time()

        if is_2d: 
            # deltas
            # curr_x, curr_y = convert_js_xy(inp[0, 0], inp[0, 1])
            # temp_inp = torch.tensor(
            #     [[curr_x, curr_y]]
            # )
            c_ext, c_yaw = -inp[0, 0] + goal[0], -inp[0, 1] + goal[1]
            c_x, c_y = convert_js_xy(c_ext, c_yaw)
            g_x, g_y = convert_js_xy(goal[0], goal[1])
            d_x, d_y = g_x - c_x, g_y - c_y
            temp_inp = torch.tensor(
                [[d_x, d_y]]
            )
            
        else:
            if use_delta:
                c_x, c_y = convert_js_xy(inp[0, 0], inp[0, 1])
                g_x, g_y = convert_js_xy(goal[0], goal[1])
                d_x, d_y = g_x - c_x, g_y - c_y
                temp_inp = torch.tensor([[c_x,c_y,d_x,d_y]])
            else:
                c_x, c_y = convert_js_xy(inp[0, 0], inp[0, 1])
                g_x, g_y = convert_js_xy(goal[0], goal[1])
                temp_inp = torch.tensor([[c_x,c_y,g_x,g_y]])
       
        prediction = model(temp_inp)
        prediction = prediction.flatten()

        predicted_kp = torch.argmax(prediction).item()
        
        print(f'inp: {temp_inp}, pred: {kp_mapping[predicted_kp]}')
        onpolicy_kp.append(predicted_kp)

        deltas = torch.Tensor(kp_delta_mapping[predicted_kp]).to(device)

        if is_2d:
            if (-inp[0, 0] + goal[0])+ deltas[0] < 0.457 and \
            (-inp[0, 0] + goal[0]) + deltas[0] > 0.0025 and \
            (-inp[0, 1] + goal[1]) + deltas[1] < 4.5859 and \
            (-inp[0, 1] + goal[1]) + deltas[1] > -1.3837:
                inp[0, :2] -= deltas
            # else:
            #     print("Joint constraints violated")
            #     print(inp)
            #     print(goal)
            #     print(deltas)
                
        else:
            # joint limit constraints 
            if inp[0, 0] + deltas[0] < 0.457 and \
            inp[0, 0] + deltas[0] > 0.0025 and \
            inp[0, 1] + deltas[1] < 4.5859 and \
            inp[0, 1] + deltas[1] > -1.3837:
                inp[0, :2] += deltas
                if use_delta:
                    inp[0, 2:] -= deltas
            # else:
                # print("Joint limit constraints exceeded")

        if is_2d:
            c_ext, c_yaw = -inp[0, 0] + goal[0], -inp[0, 1] + goal[1]
            c_x, c_y = convert_js_xy(c_ext, c_yaw)
            g_x, g_y = convert_js_xy(goal[0], goal[1])
            d_x, d_y = g_x - c_x, g_y - c_y
        else:
            if use_delta:
                c_x, c_y = convert_js_xy(inp[0, 0], inp[0, 1])
                g_x, g_y = convert_js_xy(goal[0], goal[1])
                d_x, d_y = g_x - c_x, g_y - c_y
            else:
                c_x, c_y = convert_js_xy(inp[0, 0], inp[0, 1])
                g_x, g_y = convert_js_xy(goal[0], goal[1])
                d_x, d_y = g_x - c_x, g_y - c_y

        delta_list.append(torch.norm(torch.tensor([d_x,d_y])).item())
        inp_x.append(d_x)
        inp_y.append(d_y)

    # print(np.mean(delta_list))
    return inp_x, inp_y, min(delta_list), np.argmin(np.array(delta_list)), onpolicy_kp


if __name__ == "__main__":
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device="cpu"
    base_folder = "/home/strech/catkin_ws/src/stretch_ros/stretch_learning/checkpoints/point_shoot/20230918-145332_use_delta"
    _, new_dir = os.path.split(base_folder)
    if not os.path.exists("./sim_figures/" + new_dir):
        os.mkdir("./sim_figures/" + new_dir)
    # os.mkdir("./sim_figures/" + new_dir)
    print(device) 

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


    # train_dataloader, val_dataloader, val_endpoints = get_train_valid_split(
    #     is_2d=True, use_delta=True
    # )
    
    # start_points = []
    # end_points = []

    # for batch in train_dataloader:
    #     inp, key_pressed = batch.values()
    #     start_points.append([inp[0, 0], inp[0, 1]])
    #     end_points.append([inp[0, 0] + inp[0, 2], inp[0, 1] + inp[0, 3]])
        
    # print()
    # print(val_endpoints)
    # print(torch.from_numpy(np.array(val_endpoints)).to(device))
    # sys.exit()

    ckpt_path = 'epoch=400_success=0.250.pt'
    # for ckpt_path in os.listdir(base_folder):
    epoch = ckpt_path.split("_")[0]
    # ckpt_path = r"C:\Users\adipk\Documents\HAL\hal-skills-repo\point_and_shoot_debug\ckpts\20230901-065332_use_delta\epoch=100_mean_deltas=0.111.pt"
    model = load_bc_model(Path(base_folder, ckpt_path), device)
    model.eval()
    # pred = model(torch.tensor([-0.25, -0.15]))
    # print(pred)
    # print(torch.argmax(pred))
    # sys.exit()
    success = 0
    iteration_list = []
    is_2d = False
    use_delta = True
    is_xy = False
    for i in range(len(begin_points)):
        ext_min = 0.0025
        ext_max = 0.457
        yaw_min = -1.383
        yaw_max = 4.586
    
        if use_delta:
            inp = torch.tensor([[begin_points[i][0], begin_points[i][1], goal_points[i][0] - begin_points[i][0], goal_points[i][1] - begin_points[i][1]]]).to(device)
        else:
            inp = torch.tensor([[begin_points[i][0], begin_points[i][1], goal_points[i][0], goal_points[i][1]]]).to(device)

            # start_ext,start_yaw = torch.tensor([begin_points[i][0]]) ,torch.tensor([begin_points[i][1]])
            # start_tensor = torch.cat((start_ext, start_yaw))

            # goal_ext,goal_yaw  = torch.tensor([goal_points[i][0]]), torch.tensor([goal_points[i][1]])
            # goal_tensor = torch.cat((goal_ext - start_ext, goal_yaw - start_yaw))

            # start_tensor = start_tensor.to(device)
            # goal_tensor = goal_tensor.to(device)
            # inp = torch.cat((start_tensor, goal_tensor)).unsqueeze(0)

        max_iterations = 350

        goal_point = [goal_points[i][0], goal_points[i][1]]
        
        inp_x, inp_y, delta_min, iterations, onpolicy_kp = run_simulate(inp, goal_point, model, max_iterations, is_2d, use_delta, is_xy)

