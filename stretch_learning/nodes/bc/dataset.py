import re
import h5py
import torch

from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


class PSDataset(Dataset):
    def __init__(self, data_path):
        super(PSDataset, self).__init__()
        assert data_path.exists()
        self.data = h5py.File(data_path, "r", libver="latest")

    def __len__(self):
        return len(self.data[list(self.data.keys())[0]])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return {
            "input": self.data["input"][idx],
            "key_pressed": self.data["key_pressed"][idx],
        }


def get_train_valid_split():
    # get csvs
    point_shoot_dir = Path("/share/portal/jlr429/Hal-Skill-Repo/data/point_shoot")
    ps_csvs = Path(point_shoot_dir, "csvs")
    csvs = []
    for csv in ps_csvs.glob("**/*.csv"):
        csvs.append(csv)

    # split into train and validation
    demo_type_to_csv_paths = defaultdict(list)
    for csv_path in csvs:
        name = re.sub("[0-9]", "", csv_path.stem)
        demo_type_to_csv_paths[name].append(csv_path)

    train_set, validation_set = [], []
    train_set_names, validation_set_names = [], []
    for demo_type, csv_paths in demo_type_to_csv_paths.items():
        target_num = int(len(csv_paths) * 0.8)
        print(f"{demo_type=}: {target_num=}")
        for i, csv_path in enumerate(csv_paths):
            if i < target_num:
                train_set.append(csv_path)
                train_set_names.append(csv_path.name)
            else:
                validation_set.append(csv_path)
                validation_set_names.append(csv_path.name)

    save_dir = Path("/share/portal/jlr429/Hal-Skill-Repo/data/point_shoot")
    train_split_path = Path(save_dir, "train_split.h5")
    val_split_path = Path(save_dir, "val_split.h5")
    save_h5(train_set, train_split_path)
    save_h5(validation_set, val_split_path)

    train_dataset = PSDataset(train_split_path)
    val_dataset = PSDataset(val_split_path)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=64,
        num_workers=8,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=64,
        num_workers=8,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )

    return train_dataloader, val_dataloader


def save_h5(csvs, output_file):
    if output_file.exists():
        # print(f"{output_file.name} exists!")
        output_file.unlink()
    h5_data = {
        "csv_end": [],
        "input": [],
        "key_pressed": [],
    }
    # (x, y, z) = (joint_lift_pos, wrist_extension_pos, joint_gripper_finger_left_pos)
    debug_count = 0
    for csv_path in tqdm(csvs, total=len(csvs)):
        df = pd.read_csv(str(csv_path), header=0)
        inputs, kps, csv_ends = [], [], []
        final_row = df.iloc[-1]
        final_coord = _calculate_final_coord(final_row)
        debug_count += len(df) - 1
        for _, row in df[:-1].iterrows():
            single_js = _end_eff_only(row)
            single_js += final_coord
            inputs.append(single_js)
            kps.append(row.key_pressed)
            csv_ends.append(debug_count)
        h5_data["input"].extend(inputs)
        h5_data["key_pressed"].extend(kps)
        h5_data["csv_end"].extend(csv_ends)

    with h5py.File(output_file, "w") as f:
        for key in h5_data.keys():
            f.create_dataset(key, data=np.array(h5_data[key]), dtype=np.float32)


def _calculate_final_coord(final_row):
    lift, pitch = final_row.joint_lift_pos, final_row.joint_wrist_pitch_pos
    gripper_len = 0.21
    gripper_delta_y = gripper_len * np.sin(pitch)
    end_eff_y = lift + gripper_delta_y
    end_eff_x = final_row.wrist_extension_pos
    gripper_left = final_row.joint_gripper_finger_left_pos
    return [end_eff_y, end_eff_x, gripper_left]


def _end_eff_only(row):
    lift, pitch, ext, gripper_left = None, None, None, None
    for key in row.keys():
        if key == "joint_lift_pos":
            lift = row[key]
        elif key == "joint_wrist_pitch_pos":
            pitch = row[key]
        elif key == "wrist_extension_pos":
            ext = row[key]
        elif key == "joint_gripper_finger_left_pos":
            gripper_left = row[key]
    gripper_len = 0.21
    gripper_delta_y = gripper_len * np.sin(pitch)
    end_eff_y = lift + gripper_delta_y
    end_eff_x = ext
    return [end_eff_y, end_eff_x, gripper_left]