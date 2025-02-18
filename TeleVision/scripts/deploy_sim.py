import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    from replay_demo import Player

from pathlib import Path
import h5py
from tqdm import tqdm
import time
import yaml
import pickle
import torch
import cv2
from collections import deque
import argparse
import sys

sys.path.append("../")
from act.utils import parse_id

# from act.imitate_episodes import RECORD_DIR, DATA_DIR, LOG_DIR

from pathlib import Path

current_dir = Path(__file__).parent.resolve()
DATA_DIR = (current_dir.parent / "data/").resolve()
RECORD_DIR = (DATA_DIR / "recordings/").resolve()
LOG_DIR = (DATA_DIR / "logs/").resolve()
# print(f"\nDATA dir: {DATA_DIR}")


def get_norm_stats(data_path):
    # norm_stats = {
    #     "action_mean": np.array([]), "action_std": np.array([]),
    #     "qpos_mean": np.array([]), "qpos_std": np.array([]),
    # }
    with open(data_path, "rb") as f:
        norm_stats = pickle.load(f)
    return norm_stats


def load_policy(policy_path, device):
    policy = torch.jit.load(policy_path, map_location=device)
    return policy


def normalize_input(state, left_img, right_img, norm_stats, last_action_data=None, 
                    device="cuda:0", obs_dim=24, iphone="none"):
    # import ipdb; ipdb.set_trace()
    # left_img = cv2.resize(left_img, (308, 224))
    # right_img = cv2.resize(right_img, (308, 224))
    if iphone != "none":
        left_img = left_img.astype(dtype=np.float32) / 255.
        right_img = (torch.from_numpy(right_img) - norm_stats["depth_mean"]) / norm_stats["depth_std"]
        right_img = right_img[None,:]
        print("shapes", left_img.shape, right_img.shape)
        image_data = torch.from_numpy(np.concatenate([left_img, right_img], axis=0)[None,:])
        image_data = image_data.view((1, 1, 4, 480, 640)).to(device=device)

    else:
        image_data = torch.from_numpy(np.stack([left_img, right_img], axis=0)) / 255.0
        image_data = image_data.view((1, 2, 3, 480, 640)).to(device=device)


    qpos_data = (torch.from_numpy(state) - norm_stats["qpos_mean"]) / norm_stats[
        "qpos_std"
    ]
    qpos_data = qpos_data.view((-1, obs_dim)).to(device=device)
    # print("qpos dtype", state.dtype, norm_stats["qpos_mean"].dtype, norm_stats["qpos_std"].dtype)
    # print("qpos value", state, norm_stats["qpos_mean"], norm_stats["qpos_std"])
    # print("after qdata", qpos_data)
    # exit()
    

    if last_action_data is not None:
        last_action_data = (
            torch.from_numpy(last_action_data)
            .to(device=device)
            .view((1, -1))
            .to(torch.float)
        )
        qpos_data = torch.cat((qpos_data, last_action_data), dim=1)
    return (qpos_data, image_data)


def merge_act(actions_for_curr_step, k=0.01):
    actions_populated = np.all(actions_for_curr_step != 0, axis=1)
    actions_for_curr_step = actions_for_curr_step[actions_populated]

    exp_weights = np.exp(-k * np.arange(actions_for_curr_step.shape[0]))
    exp_weights = (exp_weights / exp_weights.sum()).reshape((-1, 1))
    raw_action = (actions_for_curr_step * exp_weights).sum(axis=0)

    return raw_action


if __name__ == "__main__":
    # parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    # parser.add_argument(
    #     "--taskid", action="store", type=str, help="task id", required=True
    # )
    # parser.add_argument(
    #     "--exptid", action="store", type=str, help="experiment id", required=True
    # )
    # parser.add_argument(
    #     "--resume_ckpt",
    #     action="store",
    #     type=str,
    #     help="resume checkpoint",
    #     required=True,
    # )
    # args = vars(parser.parse_args())

    taskid = "00"
    exptid = "01"
    resume_ckpt = "11000"

    episode_name = "processed_episode_0.hdf5"
    task_dir, task_name = parse_id(RECORD_DIR, taskid)
    episode_path = (Path(task_dir) / "processed" / episode_name).resolve()
    exp_path, _ = parse_id((Path(LOG_DIR) / task_name).resolve(), exptid)

    norm_stat_path = Path(exp_path) / "dataset_stats.pkl"
    policy_path = Path(exp_path) / f"traced_jit_{resume_ckpt}.pt"

    temporal_agg = False
    action_dim = 24

    chunk_size = 60
    device = "cuda:1"
    device = "cuda:1"

    data = h5py.File(str(episode_path), "r")
    actions = np.array(data["qpos_action"])
    left_imgs = np.array(data["observation.image.left"])
    right_imgs = np.array(data["observation.image.right"])
    states = np.array(data["observation.state"])
    # init_action = np.array(data.attrs["init_action"])
    data.close()
    timestamps = states.shape[0]

    norm_stats = get_norm_stats(norm_stat_path)
    policy = load_policy(policy_path, device)
    policy.to(device)
    policy.to(device)
    policy.eval()

    history_stack = 0
    if history_stack > 0:
        last_action_queue = deque(maxlen=history_stack)
        for i in range(history_stack):
            last_action_queue.append(actions[0])
    else:
        last_action_queue = None
        last_action_data = None
    player = Player(dt=1 / 30)

    if temporal_agg:
        all_time_actions = np.zeros([timestamps, timestamps + chunk_size, action_dim])
    else:
        num_actions_exe = chunk_size

    try:
        output = None
        act_index = 0
        for t in tqdm(range(timestamps)):
            if history_stack > 0:
                last_action_data = np.array(last_action_queue)

            data = normalize_input(
                states[t], left_imgs[t], right_imgs[t], norm_stats, last_action_data, device
                )
            print("model input data: ", data[0].shape, data[1].shape)
            if temporal_agg:
                output = (
                    policy(*data)[0].detach().cpu().numpy()
                )  # (1,chuck_size,action_dim)
                all_time_actions[[t], t : t + chunk_size] = output
                print("model output: ", output.shape)
                act = merge_act(all_time_actions[:, t])
            else:
                if output is None or act_index == num_actions_exe - 1:
                    print("Inference...")
                    output = policy(*data)[0].detach().cpu().numpy()
                    act_index = 0
                act = output[act_index]
                act_index += 1
            # import ipdb; ipdb.set_trace()
            if history_stack > 0:
                last_action_queue.append(act)
            act = act * norm_stats["action_std"] + norm_stats["action_mean"]
            player.step(act, left_imgs[t], right_imgs[t])
    except KeyboardInterrupt:
        player.end()
        exit()
