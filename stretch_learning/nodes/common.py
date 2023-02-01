from pathlib import Path
import csv
import traceback
import requests
import math

root_dir = ""
user = str(Path.home())
if user == "/home/jlr429":
    #g2
    root_dir = Path("/share/cuvl/jlr429/bc")
elif user == "/home/portal":
    root_dir = Path("/home/portal/juntao")
elif user == "/home/strech":
    rood_dir = Path("~/")
else:
    print("No valid home directory")
    exit(0)
DATA_DIR = Path(root_dir, "data")

# csvs
CSV_DIR = Path(DATA_DIR, "csv")
REG_CSVS_DIR = Path(CSV_DIR, "reg_rollouts")
OVERSHOOT_ROLLOUTS_DIR = Path(CSV_DIR, "overshoot_rollouts")
RECOVERY_ROLLOUTS_DIR = Path(CSV_DIR, "recovery_rollouts")
LEARNER_ROLLOUTS_DIR = Path(CSV_DIR, "learner_rollouts")

# test trajectories
TEST_TRAJ_DIR = Path(DATA_DIR, "test_traj")
NO_CORRECTIONS_DIR = Path(TEST_TRAJ_DIR, "bc_no_corrections")
TWO_CORRECTIONS_DIR = Path(TEST_TRAJ_DIR, "bc_2_corrections")
IQL_SUBDOM_DIR  = Path(TEST_TRAJ_DIR, "iql_subdom")

# ckpts
ALL_CKPTS_DIR = Path(root_dir, "stretch_behavioral_cloning", "src", "all_ckpts")

# plots
PLOTS_DIR = Path(root_dir, "stretch_behavioral_cloning", "plots")
SUBCOSTS_HEADER = ["Expert", "Unsuccessful", "Dist from Transition (meters)", "Total Subcost"]

two_correction_episode_names = ["far_open_drawer_2022-11-16-20-49-30", 
                                "far_open_drawer_2022-11-16-20-46-18",
                                "high_open_drawer_2022-11-16-20-30-34",
                                "high_open_drawer_2022-11-16-20-29-26"]
                            
five_correction_episode_names = two_correction_episode_names + [
    "far_open_drawer_2022-11-16-20-44-33", "far_open_drawer_2022-11-16-20-42-57",
    "far_open_drawer_2022-11-16-20-40-33", "high_open_drawer_2022-11-16-20-21-40",
    "high_open_drawer_2022-11-16-20-27-12", "high_open_drawer_2022-11-16-20-26-21"
]


# helper functions   
def calculate_y_pos(row):
    lift, pitch = row["joint_lift_pos"].item(), row["joint_wrist_pitch_pos"].item()
    gripper_len = 0.21
    gripper_delta_y = gripper_len * math.sin(pitch)
    y_pos = lift + gripper_delta_y
    return y_pos

def save_lines_to_csv(header, rows, save_path, verbose=True):
    if not save_path.parent.exists():
        save_path.parent.mkdir()
        
    with open(save_path, "w", encoding="UTF8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    if verbose:
        print(f"Saved {len(rows)} lines to {save_path}")

def progress_alerts(original_function=None, *, func_name="clip-metric"):
    def _progress_alerts(func):
        def wrapper(*args, **kwargs):
            try:
                func(*args, **kwargs)
            except Exception as e:
                if post_slack_message(f"Error in training {func_name}"):
                    stack = traceback.format_exc()
                    post_slack_message(stack)
                    post_slack_message("-" * 80)
                else: 
                    traceback.print_exc()
                exit()
            post_slack_message(f"{func_name} finishing running!")
        return wrapper 

    if original_function:
        return _progress_alerts(original_function)
    
    return _progress_alerts

def post_slack_message(msg):
    if "jlr429" in str(root_dir) or "juntao" in str(root_dir):
        msg = msg.replace("\'", "-")
        msg = msg.replace("\"", "-")
        payload = '{"text":"%s"}' % msg
        requests.post("https://hooks.slack.com/services/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", payload)
        return True
    else:
        return False