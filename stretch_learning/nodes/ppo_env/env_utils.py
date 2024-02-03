import random
import numpy as np
import torch
import math
from shapely.geometry import LineString

def rotate_odom(coord, angle, pivot):
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

def convert_js_xyz(joint_state):
    """
    calculates the xyz point in the griper with respect to the Lidar on Stretch
    """
    extension = joint_state[0]
    yaw = joint_state[1]
    lift = joint_state[2]
    base_x = joint_state[3]
    base_y = joint_state[4]
    base_angle = joint_state[5]

    gripper_len = 0.27 
    base_gripper_yaw = -0.09 # correction for straightened gripper
    
    # cx, cy is the base of the gripper wrt the lidar in meters
    # lidar point does not rotate with the base
    # find cx, cy in base frame
    point = (0.03, 0.17)
    pivot = (0, 0)
    cx, cy = rotate_odom(point, base_angle, pivot)

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

def recovery_reset(s_low, s_high, g_low, g_high, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    xyz_dims = 3
    s_dims = len(s_low)
    g_dims = len(g_low)
    state_js = np.zeros((s_dims,), dtype=np.float32)
    state_xyz = np.zeros((xyz_dims,), dtype=np.float32)
    goal = np.zeros((g_dims,), dtype=np.float32)
    goal_xyz = np.zeros((xyz_dims,), dtype=np.float32)
    delta_state = goal_xyz - state_xyz

    goal = np.random.uniform(g_low, g_high)
    
    # set the maximum height to be more than a keypress below the shelf
    below_shelf = 0.08
    s_high = np.array([0.4, goal[1], goal[2] - below_shelf, 0, 0, 0]) 

    # set minimum extension to be more than a keypress past the goal
    minimum_retract = 0.025
    s_low  = np.array([goal[0]+minimum_retract, goal[1],  0.1538, 0, 0, 0])

    state_js = np.random.uniform(s_low, s_high)
    
    state_xyz = convert_js_xyz(np.copy(state_js))
    goal_xyz = convert_js_xyz(np.copy(goal))

    state_js = np.float32(state_js)
    goal = np.float32(goal)
    state_xyz = np.float32(state_xyz)
    goal_xyz = np.float32(goal_xyz)
    delta_state = goal_xyz - state_xyz
    # xyz_dims = 3
    # s_dims = len(s_low)
    # g_dims = len(g_low)
    # state_js = np.zeros((s_dims,), dtype=np.float32)
    # state_xyz = np.zeros((xyz_dims,), dtype=np.float32)
    # goal = np.zeros((g_dims,), dtype=np.float32)
    # goal_xyz = np.zeros((xyz_dims,), dtype=np.float32)
    # delta_state = goal_xyz - state_xyz

    # goal = np.random.uniform(g_low, g_high)

    # # modifications to ensure recovery is necessary
    # below_shelf = 0.08 # meters below object height
    # s_high[1] = goal[1] + 0.001
    # s_high[2] = goal[2] - below_shelf

    # minimum_retract = 0.025 # (meters) ensures at least one retract keypress
    # s_low[0] = goal[0] + minimum_retract 
    # s_low[1] = goal[1] - 0.001

    # state_js = np.random.uniform(s_low, s_high)

    # state_xyz = convert_js_xyz(np.copy(state_js))
    # goal_xyz = convert_js_xyz(np.copy(goal))

    # state_js = np.float32(state_js)
    # state_xyz = np.float32(state_xyz)
    # goal_xyz = np.float32(goal_xyz)
    # delta_state = goal_xyz - state_xyz
    
    return state_js, state_xyz, goal_xyz, delta_state

def standard_reset(s_low, s_high, g_low, g_high, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    xyz_dims = 3
    s_dims = len(s_low)
    g_dims = len(g_low)
    state_js = np.zeros((s_dims,), dtype=np.float32)
    state_xyz = np.zeros((xyz_dims,), dtype=np.float32)
    goal = np.zeros((g_dims,), dtype=np.float32)
    goal_xyz = np.zeros((xyz_dims,), dtype=np.float32)
    delta_state = goal_xyz - state_xyz

    goal = np.random.uniform(g_low, g_high)
    state_js = np.random.uniform(s_low, s_high)
    
    state_xyz = convert_js_xyz(np.copy(state_js))
    goal_xyz = convert_js_xyz(np.copy(goal))

    state_js = np.float32(state_js)
    state_xyz = np.float32(state_xyz)
    goal_xyz = np.float32(goal_xyz)
    delta_state = goal_xyz - state_xyz
    
    return state_js, state_xyz, goal_xyz, delta_state

def fixed_reset(start, goal):
    """
    joint states: [ext, yaw, lift, base_x, base_y, base_rot]
    start (List[float]): start joint states
    goal (List[float]): goal joint states
    """
    state_js = np.array(start, dtype=np.float32)
    state_xyz = convert_js_xyz(state_js)
    goal = np.array(goal,dtype = np.float32)
    goal_xyz = convert_js_xyz(goal)
    delta_state = goal_xyz - state_xyz

    state_js = np.float32(state_js)
    state_xyz = np.float32(state_xyz)
    goal_xyz = np.float32(goal_xyz)
    delta_state = goal_xyz - state_xyz
    
    return state_js, state_xyz, goal_xyz, delta_state

def get_rect_line_segments(coords):
    a = LineString([coords[0], coords[1]])
    b = LineString([coords[0], coords[2]])
    c = LineString([coords[1], coords[3]])
    return [a, b, c]

def get_rect_coords(goal, blinder_x, blinder_y):
    goal_x = goal[0]
    goal_y = goal[1]

    x1 = goal_x - blinder_x
    y1 = goal_y + blinder_y
    x2 = goal_x + blinder_x
    y2 = goal_y + blinder_y
    x3 = goal_x - blinder_x
    y3 = goal_y - blinder_y
    x4 = goal_x + blinder_x
    y4 = goal_y - blinder_y

    return [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

def get_obj_bin(coords, angle, pivot):
    """
    angles the blinders with respect to the current base rotation
    coords [(x1, y1), ..., (x4, y4)]: coordinates of rectangle for blinders
    angle (rad): base rotation
    pivot (goal x, goal y): goal point to rotate about
    """
    x1 = math.cos(angle)*(coords[0][0]-pivot[0]) - math.sin(angle)*(coords[0][1]-pivot[1]) + pivot[0]
    y1 = math.sin(angle)*(coords[0][0]-pivot[0]) + math.cos(angle)*(coords[0][1]-pivot[1]) + pivot[1]

    x2 = math.cos(angle)*(coords[1][0]-pivot[0]) - math.sin(angle)*(coords[1][1]-pivot[1]) + pivot[0] 
    y2 = math.sin(angle)*(coords[1][0]-pivot[0]) + math.cos(angle)*(coords[1][1]-pivot[1]) + pivot[1]

    x3 = math.cos(angle)*(coords[2][0]-pivot[0]) - math.sin(angle)*(coords[2][1]-pivot[1]) + pivot[0]
    y3 = math.sin(angle)*(coords[2][0]-pivot[0]) + math.cos(angle)*(coords[2][1]-pivot[1]) + pivot[1] 

    x4 = math.cos(angle)*(coords[3][0]-pivot[0]) - math.sin(angle)*(coords[3][1]-pivot[1]) + pivot[0]
    y4 = math.sin(angle)*(coords[3][0]-pivot[0]) + math.cos(angle)*(coords[3][1]-pivot[1]) + pivot[1]

    return [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]

def blinder_valid(goal_xyz, robot_theta, js, new_js, blinder_x, blinder_y):
    rect_coords = get_rect_coords(goal_xyz, blinder_x, blinder_y)
    pivot = (goal_xyz[0], goal_xyz[1])
    obj_bin = get_obj_bin(rect_coords, robot_theta, pivot)
    rect_line_segments = get_rect_line_segments(obj_bin)

    state_xyz_before = convert_js_xyz(js)
    state_xyz_after = convert_js_xyz(new_js)
    coord1 = (state_xyz_before[0], state_xyz_before[1])
    coord2 = (state_xyz_after[0], state_xyz_after[1])
    a = LineString([coord1, coord2])

    valid = True
    for i in rect_line_segments:
      valid = valid and not i.intersects(a)

    return valid

def is_valid(js, s_low, s_high):
    """
    js: current [ext, yaw, lift, curr_x, curr_y, base_angle]
    delta: delta [ext, yaw, lift, curr_x, curr_y, base_angle]
    """
    valid_js = False
    # joint_states
    if js[0] <= s_high[0] and \
        js[0] >= s_low[0] and \
        js[1] <= s_high[1] and \
        js[1] >= s_low[1] and \
        js[2] <= s_high[2] and \
        js[2] >= s_low[2]:

        valid_js = True
    
    return valid_js

def shelf_valid(goal_xyz, js, new_js, shelf_x_offset, shelf_y_offset, shelf_z_offset):
			
    state_xyz_before = convert_js_xyz(js)
    state_xyz_after = convert_js_xyz(new_js)
    
    goal_x_high = goal_xyz[0] + shelf_x_offset
    goal_x_low = goal_xyz[0] - shelf_x_offset

    goal_y_high = goal_xyz[1] + shelf_y_offset
    goal_y_low = goal_xyz[1] - shelf_y_offset

    shelf_z = goal_xyz[2] - shelf_z_offset

    in_x = state_xyz_after[0] <= goal_x_high and state_xyz_after[0] >= goal_x_low
    in_y = state_xyz_after[1] <= goal_y_high and state_xyz_after[1] >= goal_y_low
    cross_under_above = (state_xyz_before[2] <= shelf_z and state_xyz_after[2] >= shelf_z)
    cross_above_under = (state_xyz_before[2] >= shelf_z and state_xyz_after[2] <= shelf_z)
    cross_z = cross_under_above or cross_above_under

    return not (in_x and in_y and cross_z)



