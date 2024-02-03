from rosbags.typesys import get_types_from_msg, register_types
from rosbags.serde import deserialize_cdr, ros1_to_cdr
from rosbags.rosbag1 import Reader
import numpy as np
from icecream import ic

from rosbags.highlevel import AnyReader
import csv
import cv2
import warnings
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from tqdm import tqdm
from scipy import ndimage

kp_reduced_mapping = {
    # arm out
    "w": 0,
    # arm in
    "x": 1,
    # gripper right
    "a": 2,
    # gripper left
    "d": 3,
    # base forward
    "4": 4,
    # base back
    "6": 5,
    # base rotate left
    "7": 6,
    # base rotate right
    "9": 7,
    # arm up
    "8": 8,
    # arm down
    "2": 9,
}

shared_stretch_data_dir = Path(
    "/home/strech/catkin_ws/src/stretch_ros/stretch_learning/nodes"
)


def parse_bags(bagfile_dir_name, override):
    bagfile_dir = Path(shared_stretch_data_dir, bagfile_dir_name)
    # print(bagfile_dir)
    # assert bagfile_dir.exists()
    csv_save_dir = Path(
        shared_stretch_data_dir, bagfile_dir_name.replace("_bags", "_csvs")
    )
    img_save_dir = Path(
        shared_stretch_data_dir, bagfile_dir_name.replace("_bags", "_imgs")
    )
    csv_save_dir.mkdir(exist_ok=True)
    img_save_dir.mkdir(exist_ok=True)
    bagfiles = [bagfile for bagfile in bagfile_dir.glob("*.bag")]
    print(f"{len(bagfiles)=}")
    for i, bagfile in tqdm(enumerate(bagfiles), maxinterval=len(bagfiles), ncols=0):
        csv_path = Path(csv_save_dir, f"{bagfile.stem}.csv")
        if csv_path.exists() and not override:
            print(f"skipping {bagfile} because {csv_path} exists")
            continue
        else:
            print(f"Parsing {bagfile.stem} to {csv_path}")
        img_path = Path(img_save_dir, f"{bagfile.stem}")
        img_path.mkdir(exist_ok=True)
        convert_bagfile_to_csv(str(bagfile), str(csv_path), str(img_path))


def convert_bagfile_to_csv(bagfile_path, csv_path, img_path):
    key_pressed_topic = "/key_pressed"
    wrist_image_topic = "/wrist_camera/color/image_raw"
    head_image_topic = "/head_camera/color/image_raw"
    joint_states_topic = "/stretch/joint_states"
    transforms_topic = "/tf"
    transforms_static_topic = "/tf_static"
    topics_list = [
        key_pressed_topic,
        wrist_image_topic,
        head_image_topic,
        joint_states_topic,
        transforms_topic,
        transforms_static_topic,
    ]

    # Iterate through all topics in bag and write their data to csv
    # create head image path
    head_image_paths = []
    wrist_image_paths = []
    joint_state_data = []
    key_pressed_data = []

    with Reader(bagfile_path) as reader:
        typs = {}
        for conn in reader.connections:
            typs.update(get_types_from_msg(conn.msgdef, conn.msgtype))

            register_types(typs)
            i = 0

        for connection, t, rawdata in tqdm(reader.messages(), ncols=0, desc="Parsing"):
            topic = connection.topic
            if topic not in topics_list:
                i += 1
                continue
            msg = deserialize_cdr(
                ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype
            )

            if topic == wrist_image_topic:
                img = cv2.cvtColor(
                    msg.data.reshape((msg.height, msg.width, 3)), cv2.COLOR_RGB2BGR
                )
                cv2.imwrite(img_path + f"/wrist_{Path(bagfile_path).stem}_{i}.png", img)
                wrist_image_paths.append(
                    (
                        t,
                        img_path + f"/wrist_{Path(bagfile_path).stem}_{i}.png",
                    )
                )

            if topic == head_image_topic:
                img = cv2.cvtColor(
                    msg.data.reshape((msg.height, msg.width, 3)), cv2.COLOR_RGB2BGR
                )
                # img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                img = ndimage.rotate(img, -96)
                cv2.imwrite(img_path + f"/head_{Path(bagfile_path).stem}_{i}.png", img)
                head_image_paths.append(
                    (
                        t,
                        img_path + f"/head_{Path(bagfile_path).stem}_{i}.png",
                    )
                )

            if topic == joint_states_topic:
                js_data = [(x, y) for (x, y) in zip(msg.name, msg.position)]
                js_data.sort(key=lambda x: x[0])
                js_data = [x[1] for x in js_data]

                joint_state_data.append((t, js_data))

            if topic == key_pressed_topic:
                if msg.keypressed not in kp_reduced_mapping:
                    print(f"keypressed {msg.keypressed} not in mapping")
                    continue
                key_pressed_data.append((t, kp_reduced_mapping[msg.keypressed]))

            if topic == transforms_topic:
                print(msg)
                sys.exit()

        head_image_paths.sort(key=lambda x: x[0])
        wrist_image_paths.sort(key=lambda x: x[0])
        joint_state_data.sort(key=lambda x: x[0])
        key_pressed_data.sort(key=lambda x: x[0])

        # get list with most logs
        head_img_len = len(head_image_paths)
        wrist_img_len = len(wrist_image_paths)
        js_len = len(joint_state_data)
        kp_len = len(key_pressed_data)

        if (
            head_img_len >= wrist_img_len
            and head_img_len >= js_len
            and head_img_len >= kp_len
        ):
            longest_list = head_image_paths
        elif (
            wrist_img_len >= head_img_len
            and wrist_img_len >= js_len
            and wrist_img_len >= kp_len
        ):
            longest_list = wrist_image_paths
        elif js_len >= head_img_len and js_len >= wrist_img_len and js_len >= kp_len:
            longest_list = joint_state_data
        elif kp_len >= head_img_len and kp_len >= wrist_img_len and kp_len >= js_len:
            longest_list = key_pressed_data

        # longest_list = (
        #     wrist_image_paths if wrist_img_len >= head_img_len else head_image_paths
        # )
        # if js_len > len(longest_list):
        #     warnings.warn("joint state data is longer than image data")
        # if kp_len > len(longest_list):
        #     warnings.warn("key pressed data is longer than image data")

        # try:
        head_image_paths = expand_list(longest_list, head_image_paths)
        # wrist_image_paths = expand_list(longest_list, wrist_image_paths)
        joint_state_data = expand_list(longest_list, joint_state_data)

        if key_pressed_data == []:
            key_pressed_data = expand_list(longest_list, [(0, 0)])
        else:
            key_pressed_data = expand_list(longest_list, key_pressed_data)
        # except:
        #     breakpoint()

    head_image_paths = [x[1] for x in head_image_paths]
    # wrist_image_paths = [x[1] for x in wrist_image_paths]
    joint_state_data = [x[1] for x in joint_state_data]
    key_pressed_data = [x[1] for x in key_pressed_data]

    data = {
        "head_image_path": head_image_paths,
        # "wrist_image_path": wrist_image_paths,
        "joint_state_data": joint_state_data,
        "key_pressed_data": key_pressed_data,
    }

    # create a pandas DataFrame from the dictionary
    df = pd.DataFrame(data)

    # calculate goal_pos, delta_pos
    goal_pos_data, delta_pos_data = [], []
    final_row = df.iloc[-1]

    # js_data = final_row.joint_state_data
    # js_total = [
    #     float(js_data[-1]),
    #     float(js_data[-2]),
    #     float(js_data[-5]),
    # ]
    # goal_pos = end_eff_to_xyz(*js_total)

    transform = final_row.transform

    for _, row in df[:-1].iterrows():
        js_data = row.joint_state_data
        js_total = [
            float(js_data[-1]),
            float(js_data[-2]),
            float(js_data[-5]),
        ]
        curr_pos = end_eff_to_xyz(*js_total)
        goal_pos_data.append(
            np.array2string(goal_pos, precision=5, separator=",", suppress_small=True)
        )
        delta_pos = np.array(goal_pos) - np.array(curr_pos)
        delta_pos_data.append(
            np.array2string(delta_pos, precision=5, separator=",", suppress_small=True)
        )
    # add last row
    goal_pos_data.append(
        np.array2string(goal_pos, precision=5, separator=",", suppress_small=True)
    )
    delta_pos_data.append(
        np.array2string(
            np.array([0, 0, 0]), precision=5, separator=",", suppress_small=True
        )
    )
    df["goal_pos"] = goal_pos_data
    df["delta_pos"] = delta_pos_data

    # save the DataFrame to a CSV file
    df.to_csv(csv_path, index=False)


def end_eff_to_xyz(extension, yaw, lift):
    gripper_len = 0.27
    base_gripper_yaw = -0.09
    yaw_delta = -(yaw - base_gripper_yaw)
    y = gripper_len * np.cos(yaw_delta) + extension
    x = gripper_len * np.sin(yaw_delta)
    z = lift
    return np.array([x, y, z])


def expand_list(reference_list, shorter_list):
    """
    Helper function that fills the shorter_list to length of longer_list.
    If timestamp of current element is less than timestamp of the next element in shorter_list,
    add a copy of the shorter_list element. Otherwise, increment the pointer in shorter_list.

    If shorter_list has no next element, add a copy of the last element in shorter_list.
    If longest_list has no next element, add the rest of shorter_list.

    Returns:
        expanded_list: list with length of longer_list but contents of shorter_list copied
    """
    if len(reference_list) == len(shorter_list):
        return shorter_list

    expanded_list = []
    shorter_list_pointer = 0

    for i, (timestamp, _) in enumerate(reference_list):
        if shorter_list_pointer == len(shorter_list):
            # there are more images after the last element in shorter_list
            # in this case, we just copy the last element until both lists are same length
            while len(expanded_list) < len(reference_list):
                expanded_list.append(shorter_list[-1])
            break

        expanded_list.append(shorter_list[shorter_list_pointer])
        if shorter_list[shorter_list_pointer][0] < timestamp:
            shorter_list_pointer += 1

    assert len(expanded_list) == len(reference_list)
    return expanded_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--bagfile_dir", type=str, required=True)
    parser.add_argument("--override", action="store_true", default=False)
    args = parser.parse_args()
    parse_bags(args.bagfile_dir, args.override)
