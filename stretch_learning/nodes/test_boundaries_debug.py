import torch
import numpy as np
from dataset import get_train_valid_split, save_h5
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
# from bc_new import BC
from ablate_bc import BC
from pathlib import Path
import sys
kp_mapping = ["Arm out predict", "Arm in predict", "Gripper right predict", "Gripper left predict"]
policy_mapping = ["Arm out train", "Arm in train", "Gripper right train", "Gripper left train"]
device = "cuda" if torch.cuda.is_available() else "cpu"
from tqdm import tqdm
import pandas as pd
import h5py

def get_mean_std(csvs):
    values_list = {
        "wrist_extension": [],
        "wrist_yaw": [],
    }
    debug_count = 0
    for csv_path in tqdm(csvs, total=len(csvs)):
        df = pd.read_csv(str(csv_path), header=0)
        debug_count += len(df.wrist_extension_pos)
        values_list["wrist_extension"].extend(df.wrist_extension_pos.tolist())
        values_list["wrist_yaw"].extend(df.joint_wrist_yaw_pos.tolist())
    return {
        "ext_mean": np.mean(values_list["wrist_extension"]),
        "ext_std": np.std(values_list["wrist_yaw"]),
        "yaw_mean": np.mean(values_list["wrist_extension"]),
        "yaw_std": np.std(values_list["wrist_yaw"]),
    }

@torch.no_grad()
def main():
    # ps_csvs = Path("/share/portal/jlr429/hal-skill-repo/data/point_shoot_csvs")
    # csvs = []
    # for csv in ps_csvs.glob("**/*.csv"):
    #     csvs.append(csv)
    # # assert len(csvs) == 4 * 10
    # mean_std_dict = get_mean_std(csvs)

    # train_set, validation_set = [], []
    # for csv in tqdm(csvs, total=len(csvs), desc="all csvs"):
    #     if "9" in csv.stem or "10" in csv.stem:
    #         validation_set.append(csv)
    #     else:
    #         train_set.append(csv)
    
    # save_dir = Path("/share/portal/nlc62/hal-skill-repo/point_and_shoot_debug/data/point_shoot_csvs")
    # train_split_path = Path(save_dir, "train_split_4d_deltas.h5")
    # val_split_path = Path(save_dir, "val_split_4d_deltas.h5")

    # is_2d = True
    # use_delta = True

    # save_h5(train_set, train_split_path, is_2d, use_delta, mean_std_dict)
    # save_h5(validation_set, val_split_path, is_2d, use_delta, mean_std_dict)
    # sys.exit()
    is_2d = False
    # test_trajs = [traj for traj in reg_trajs.glob("*.csv")]
    extensions = [0.45, 0.05]  # [x extension, y extension]
    paddings = [0.05, 0.05]  # [x padding, y padding]

    # Plot training data (ground truth)
    expert_kps = np.empty(0)
    expert_pos = np.empty((0, 2))

    train_dataloader, val_dataloader, _ = get_train_valid_split(is_2d=False, use_delta=True)
    for test_data in train_dataloader:
        inp, kp = test_data.values()
        expert_kps = np.append(expert_kps, kp, axis=0)
        expert_pos = np.append(expert_pos, inp[:, 2:], axis=0)

    # expert_pos = np.flip(expert_pos, axis=1)  # ext to y-axis, yaw to x-axis
    # expert_pos[:, 1] = -expert_pos[:, 1]
    # expert_pos[:, 0] = -expert_pos[:, 0]
    
    x_min, x_max = expert_pos[:, 0].min(), expert_pos[:, 0].max()
    y_min, y_max = expert_pos[:, 1].min(), expert_pos[:, 1].max()
    mins = np.array([x_min, y_min])
    maxs = np.array([x_max, y_max])
    res = np.array([0.005, 0.005])
    goal_x, goal_y = 0.0, 0.0

    expert_kps = np.int64(expert_kps)
    # decision_boundary(
    #     expert_pos,
    #     expert_kps,
    #     (goal_x, goal_y),
    #     title="training boundaries with xy",
    #     file="data",
    #     save_dir="raw_x_y",
    #     x_padding=extensions[0] + paddings[0],
    #     y_padding=extensions[1] + paddings[1],
    # )

    
    # exit()

    x_min, x_max = min(x_min, x_max), max(
        x_min, x_max
    )  # since the horizontal axes will be flipped later
    mins[0], maxs[0] = x_min, x_max
    mins[0], mins[1] = mins[0] - extensions[0], mins[1] - extensions[1]
    maxs[0], maxs[1] = maxs[0] + extensions[0], maxs[1] + extensions[1]
    # mins = [-0.8, -0.8]
    # maxs = [0.8, 0.8]
    # print(mins)
    # print(maxs)
    # sys.exit()
    # ranges = [np.arange(mins[i], maxs[i] + res[i], res[i]) for i in range(len(mins))]
    
    # grids = np.meshgrid(*ranges, indexing="ij")
    print(x_min,x_max,y_min,y_max)
    x = np.linspace(-0.5, 0.5, num=300)
    y = np.linspace(-0.45, 0.8, num=300)

    grids = np.meshgrid(x, y)
    pts = np.stack(grids, axis=-1).reshape(-1, len(mins))
    print(f"Evaluating {len(pts)} data points")
    model = BC(is_2d=False, use_delta=True)
    
    for ckpt_path in Path("/share/portal/nlc62/hal-skill-repo/point_and_shoot_debug/ckpts/20230913-205137_use_delta").glob("*.pt"):
        # model.train()
        print(f"Evaluating {ckpt_path.stem}")
        state_dict = torch.load(ckpt_path, map_location=torch.device(device))
        model.load_state_dict(state_dict)
        model.to(device)
        # model = model.float()
        # torch.compile(model)
        model.eval()

        predicted_kps = []
        pts = np.stack(grids, axis=-1).reshape(-1, len(mins))
        delta_pts = []
        for p in pts:

            dx, dy = goal_x - p[0], goal_y - p[1]
            delta_pts.append([dx, dy])
            inp = np.append(p, [dx, dy])
            inp = torch.from_numpy(inp).float().unsqueeze(0).to(device)
            
            if is_2d:
                inp = inp[0, 2:]
            
            
            predicted_action = model(inp)
            predicted_kp = torch.argmax(predicted_action).item()
            # print(f'inp: {inp} pred: {predicted_kp}')
            predicted_kps.append([predicted_kp])
        predicted_kps = np.array(predicted_kps).flatten()
        # pts = np.flip(pts, axis=0)
        # pts = -pts
        # pts[:, 0] = -pts[:, 0]
        # decision_boundary(
        #     pts,
        #     predicted_kps,
        #     (goal_x, goal_y),
        #     title=f"{ckpt_path.stem}",
        #     file="pred",
        #     save_dir=Path("raw_x_y", ckpt_path.parent.stem),
        #     x_padding=paddings[0],
        #     y_padding=paddings[1],
        # )
        path = f"hello_{ckpt_path.stem}"
        overlay_decision_boundary(
            expert_pos,
            expert_kps,
            np.array(delta_pts),
            predicted_kps,
            (goal_x, goal_y),
            path,
            title="training boundary " + str(path),
            file="data",
            save_dir="raw_x_y/20230913-205137_use_delta",
            x_padding=extensions[0] + paddings[0],
            y_padding=extensions[1] + paddings[1]
        )
        # sys.exit()
    # print("Drawing boundary")
    # print(pts.shape)
    # pts = np.flip(pts, axis=1)
    # pts = -pts
    # decision_boundary(
    #     pts,
    #     predicted_kps,
    #     (goal_x, goal_y),
    #     title="Trained Model Decision Boundary",
    #     file="pred",
    # )


def decision_boundary(
    pts, labels, goal, path, title, file=None, save_dir="temp", x_padding=0.5, y_padding=0.1
):
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
    ax = plt.gca()
    ax.set_xlim([x_min - x_padding, x_max + x_padding])
    ax.set_ylim([y_min - y_padding, y_max + y_padding])

    scatter = plt.scatter(pts[:, 0], pts[:, 1], c=labels, cmap="viridis", s=5, alpha=1)
    plt.plot(goal[0], goal[1], marker="*", markersize=15, color="red")
    handles, _ = scatter.legend_elements()
    filtered_kp_mapping = [kp_mapping[i] for i in np.unique(labels)]
    plt.legend(handles, filtered_kp_mapping, title="Classes")
    plt.xlabel("Relative x")
    plt.ylabel("Relative y")
    plt.title(title)
    print(f"Saving as decision_boundary{'_' + file if file else ''}.png")

    save_path = Path(save_dir, f"{ckpt_path}")
    save_path.parent.mkdir(exist_ok=True)
    plt.savefig(
        # f'decision_boundary{"_" + file if file else ""}.png',
        save_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    # plt.show()

def overlay_decision_boundary(
    expert_pos, expert_kps, pts, labels, goal, path, title, file=None, save_dir="temp", x_padding=0.5, y_padding=0.1
):

    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
    ax = plt.gca()
    ax.set_xlim([x_min - x_padding, x_max + x_padding])
    ax.set_ylim([y_min - y_padding, y_max + y_padding])

    scatter = plt.scatter(pts[:, 0], pts[:, 1], c=labels, cmap="viridis", s=5, alpha=1)
    plt.plot(goal[0], goal[1], marker="*", markersize=15, color="red")
    handles, _ = scatter.legend_elements()
    filtered_kp_mapping = [kp_mapping[i] for i in np.unique(labels)]
    plt.legend(handles, filtered_kp_mapping, title="Classes")
    plt.xlabel("Relative x")
    plt.ylabel("Relative y")
    plt.title(title)

    scatter = plt.scatter(expert_pos[:, 0], expert_pos[:, 1], c=expert_kps, cmap="viridis", edgecolors='black', linewidth=0.3, s=5, alpha=1)
    handles2, _ = scatter.legend_elements()
    expert_kp_mapping = [policy_mapping[i] for i in np.unique(expert_kps)]
    plt.legend(handles+handles2, filtered_kp_mapping+expert_kp_mapping, title="Classes")

    save_path = Path(save_dir, f"{path}.png")
    save_path.parent.mkdir(exist_ok=True)
    print(f"Saving as {save_path}")
    plt.savefig(
        # f'decision_boundary{"_" + file if file else ""}.png',
        save_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

def convert_js_xy(extension,yaw):
    gripper_len = 0.22
    base_gripper_yaw = -0.09
    yaw_delta = -(yaw - base_gripper_yaw)  # going right is more negative
    y = gripper_len * np.cos(yaw_delta) + extension
    x = gripper_len * np.sin(yaw_delta)
    return x,y
    
if __name__ == "__main__":
    main()
