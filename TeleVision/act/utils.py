import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader
import time
import IPython

e = IPython.embed
from pathlib import Path

QPOS_DIM = 24

STATE_NAME = ["head", "arm", "wrist", "hand", "force"]
STATE_DIM = [2, 4*2, 2, 6*2, 6*2]

def get_state_dim(keep_state, action_dim, history_stack=0, left_right_mask=None):
    if left_right_mask is None:
        left_right_mask = [1, 1]
    state_dim = 0
    for i, sname in enumerate(STATE_NAME):
        if sname in keep_state:
            sdim = STATE_DIM[i]
            if sname == "force":
                sdim = 2
            if sname != "head" and sum(left_right_mask) == 1:
                state_dim += sdim // 2
            else:
                state_dim += sdim
    print("state_dim, action_dim, history_stack: ", state_dim, action_dim, (history_stack))
    return state_dim, state_dim + action_dim * (history_stack)

def simplify_force(force, threshold=150):
    force = (np.mean(force, axis=-1, keepdims=True) > threshold).astype(force.dtype)
    return force

def preprocess_data(state, keep_state, 
                    left_right_mask=None, 
                    raw_control_state=False,
                    stats_mode=None,
                    process_callback=None):
    """ Preprocess the state data to keep only the necessary states.
    Args:
        state (np.ndarray): The state data to preprocess.
        keep_state (list[str]): The list of states to keep.
        left_right_mask (list[int]): The mask to separate left and right arm, [1, 0] for only keep left.
        raw_control_state (bool): For raw control state, switch left and right of arm.
    """
    if left_right_mask is None:
        left_right_mask = [1, 1]
        
    left_right_mask = [i for i, m in enumerate(left_right_mask) if m == 1]
    state_dim_num = len(state.shape)
        
    if state_dim_num == 1:
        state = state.reshape([1, -1])
        
    state_name = STATE_NAME
    state_dim = STATE_DIM
    idx = 0
    state_list = []
    for i, sname in enumerate(state_name):
        if sname in keep_state:
            s = state[:, idx:idx+state_dim[i]]
            if sname != "head":
                s = s.reshape([state.shape[0], 2, -1])
                if sname == "force":
                    if stats_mode == "std":
                        s = np.ones((s.shape[0], 2))
                    elif stats_mode == "mean":
                        s = np.zeros((s.shape[0], 2))
                    else:
                        s = simplify_force(s)

                if process_callback is not None:
                    s = process_callback(sname, s)
                
                s = s[:, left_right_mask].reshape([state.shape[0], -1])
                
            state_list.append(s)
        idx += state_dim[i]
    
    # print("state_list: ", [s.shape for s in state_list])
    state = np.concatenate(state_list, axis=1)
    if state_dim_num == 1:
        state = state[0]
    return state

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        episode_ids,
        dataset_dir,
        camera_names,
        norm_stats,
        episode_len,
        history_stack=0,
        # ignore_force=True,
        # ignore_hand=False,
        state_name=None,
        left_right_mask=None,
        chop_data=None,
        progress_bar=False,
        iphone_image="none",
        iphone_depth_type="depth",
        onehot_task_id=None,
        cancel_unsafe=False,
    ):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats.copy()
        self.iphone_image = iphone_image
        self.iphone_depth_type = iphone_depth_type
        if state_name is None:
            state_name = ["head", "arm", "wrist", "hand", "force"]
        if left_right_mask is None:
            left_right_mask = [1, 1]
        self.state_name = state_name
        self.left_right_mask = left_right_mask
        # print("state_name: ", self.norm_stats["qpos_mean"].shape, self.norm_stats["qpos_mean"].dtype)
        self.norm_stats["qpos_std"] = preprocess_data(
            self.norm_stats["qpos_std"], state_name, left_right_mask, stats_mode="std"
        ).astype(np.float32)
        self.norm_stats["qpos_mean"] = preprocess_data(
            self.norm_stats["qpos_mean"], state_name, left_right_mask, stats_mode="mean"
        ).astype(np.float32)
        if progress_bar:
            self.norm_stats["action_mean"] = np.concatenate([self.norm_stats["action_mean"], [0.5]], dtype=np.float32)
            self.norm_stats["action_std"] = np.concatenate([self.norm_stats["action_std"], [0.29]], dtype=np.float32)
        if cancel_unsafe:
            self.norm_stats["action_mean"] = np.concatenate([self.norm_stats["action_mean"], [0.5]], dtype=np.float32)
            self.norm_stats["action_std"] = np.concatenate([self.norm_stats["action_std"], [0.29]], dtype=np.float32)
        if onehot_task_id is not None:
            assert hasattr(onehot_task_id, "shape")
            self.norm_stats["qpos_mean"] = np.concatenate([
                self.norm_stats["qpos_mean"], 
                np.zeros((onehot_task_id.shape[1]), dtype=np.float32)
            ], axis=0)
            self.norm_stats["qpos_std"] = np.concatenate([
                self.norm_stats["qpos_std"], 
                np.ones((onehot_task_id.shape[1]), dtype=np.float32)
            ], axis=0)
        
        # print("qpos_mean: ", self.norm_stats["qpos_mean"].shape, self.norm_stats["qpos_mean"].dtype)
        
        self.is_sim = None
        self.max_pad_len = 200
        # self.chop_len = [20, 90]
        # self.chop_len = [0, 10]
        self.chop_len = chop_data
        action_str = "qpos_action"

        self.history_stack = history_stack

        self.dataset_paths = []
        self.roots = []
        self.is_sims = []
        self.original_action_shapes = []

        self.states = []
        self.image_dict = dict()
        for cam_name in self.camera_names:
            self.image_dict[cam_name] = []
        self.actions = []


        for i, episode_id in enumerate(self.episode_ids):
            episode_path = os.path.join(self.dataset_dir, f"processed_episode_{episode_id}.hdf5")
            root = h5py.File(episode_path, "r")
            action = root[action_str]
            if chop_data is not None:
                if self.chop_len[1] > 0:
                    action = action[self.chop_len[0]:-self.chop_len[1]]
                else:
                    action = action[self.chop_len[0]:]

            self.dataset_paths.append(episode_path)
            self.roots.append(root)
            self.is_sims.append(root.attrs["sim"])

            
            # append progress bar to action
            if progress_bar == -1:
                action = np.concatenate([action, np.linspace(0, 1, len(action))[:, None]], axis=1)
            elif progress_bar == 0:
                action = action
            elif isinstance(progress_bar,int):
                # print("len(action) is", len(action))
                ones_len = min(len(action), progress_bar)
                zeros_len = len(action) - ones_len
                action = np.concatenate([action, 
                                         np.concatenate([np.zeros((zeros_len, 1)), np.ones((ones_len, 1))], axis=0)],
                                           axis=1)

            
            if cancel_unsafe:   
                # append done label to action
                cancel_unsafe_str = "cancel_unsafe"
                cancel_unsafe_labels = root[cancel_unsafe_str]
                cancel_unsafe_labels = np.expand_dims(cancel_unsafe_labels, 1).astype(np.float32)
                action = np.concatenate([action, cancel_unsafe_labels], axis=1)
            
            self.original_action_shapes.append(action.shape)
            self.actions.append(np.array(action))
            
            state = np.array(root["observation.state"])
            if chop_data is not None:
                if self.chop_len[1] > 0:
                    state = state[self.chop_len[0]:-self.chop_len[1]]
                else:
                    state = state[self.chop_len[0]:]
            
            state = preprocess_data(state, state_name, left_right_mask=left_right_mask)
            
            if onehot_task_id is not None:
                _onehot = np.expand_dims(onehot_task_id[i], 0).repeat(state.shape[0], axis=0)
                state = np.concatenate([state, _onehot], axis=1)
            
            self.states.append(state)

            if self.iphone_image != "none":
                iphone_file = h5py.File(os.path.join(self.dataset_dir, f"iphone_episode_{episode_id}.hdf5"), "r")
                for cam_name in self.camera_names:
                    # cam_name in ["image", "depth"]
                    if self.iphone_depth_type == "depth_any" and cam_name == "depth":
                        # cam_name = "depth_any"
                        d = iphone_file["depth_any"]
                        # if cam_name not in self.image_dict:
                        #     self.image_dict[cam_name] = []
                    else:
                        d = iphone_file[cam_name]
                    if chop_data is not None:
                        self.image_dict[cam_name].append(d[self.chop_len[0]:-self.chop_len[1]])
                    else:
                        self.image_dict[cam_name].append(d)
            else:
                for cam_name in self.camera_names:
                    if chop_data is not None:
                        self.image_dict[cam_name].append(root[f"observation.image.{cam_name}"][self.chop_len[0]:-self.chop_len[1]])
                    else:
                        self.image_dict[cam_name].append(root[f"observation.image.{cam_name}"])

        print(f"Loaded {len(self.episode_ids)} episodes")

        self.is_sim = self.is_sims[0]

        self.episode_len = episode_len
        self.cumulative_len = np.cumsum(self.episode_len)

        # self.__getitem__(0) # initialize self.is_sim

    # def __len__(self):
    #     return len(self.episode_ids)

    def _locate_transition(self, index):
        assert index < self.cumulative_len[-1]
        episode_index = np.argmax(
            self.cumulative_len > index
        )  # argmax returns first True index
        start_ts = index - (
            self.cumulative_len[episode_index] - self.episode_len[episode_index]
        )
        return episode_index, start_ts

    def __getitem__(self, ts_index):
        sample_full_episode = False  # hardcode
        startt = time.time()
        index, start_ts = self._locate_transition(ts_index)

        original_action_shape = self.original_action_shapes[index]
        episode_len = original_action_shape[0]

        if sample_full_episode:
            start_ts = 0
        else:
            start_ts = np.random.choice(episode_len)

        # get observation at start_ts only
        qpos = self.states[index][start_ts]
        # qvel = root['/observations/qvel'][start_ts]

        if self.history_stack > 0:
            last_indices = np.maximum(
                0, np.arange(start_ts - self.history_stack, start_ts)
            ).astype(int)
            last_action = self.actions[index][last_indices, :]

        image_dict = dict()
        #print("time-1", time.time()-startt)
        for cam_name in self.camera_names:
            image_dict[cam_name] = self.image_dict[cam_name][index][start_ts]
        # get all actions after and including start_ts
        all_time_action = self.actions[index][:]

        #print("time0", time.time()-startt)
        all_time_action_padded = np.zeros(
            (self.max_pad_len + original_action_shape[0], original_action_shape[1]),
            dtype=np.float32,
        )
        all_time_action_padded[:episode_len] = all_time_action
        all_time_action_padded[episode_len:] = all_time_action[-1]

        padded_action = all_time_action_padded[start_ts : start_ts + self.max_pad_len]
        real_len = episode_len - start_ts

        is_pad = np.zeros(self.max_pad_len)
        is_pad[real_len:] = 1

        #print("time1", time.time()-startt)
        # new axis for different cameras
        all_cam_images = []
        if self.iphone_image == "none":
            for cam_name in self.camera_names:
                all_cam_images.append(image_dict[cam_name])
            all_cam_images = np.stack(all_cam_images, axis=0)
            image_data = torch.from_numpy(all_cam_images)
            image_data = image_data / 255.0
        else:
            img = image_dict["image"] / 255.0
            if self.iphone_image in ["rgbd", "rgb_lowd"]:
                if self.iphone_image == "rgbd":
                    depth = image_dict["depth"]
                    depth = (depth - self.norm_stats["depth_mean"].numpy()) / self.norm_stats["depth_std"].numpy()
                else:
                    depth = image_dict["lowres_depth"]
                    depth = (depth - self.norm_stats["ldepth_mean"].numpy()) / self.norm_stats["ldepth_std"].numpy()
                
                # (1, 4, *, *)
                all_cam_images = np.concatenate([img, depth], axis=0)[None,:]
            else:
                all_cam_images = img[None,:]
            image_data = torch.from_numpy(all_cam_images.astype(np.float32))

        # construct observations
        
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()
        if self.history_stack > 0:
            last_action_data = torch.from_numpy(last_action).float()

        # normalize image and change dtype to float
        
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats[
            "action_std"
        ]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats[
            "qpos_std"
        ]
        if self.history_stack > 0:
            last_action_data = (
                last_action_data - self.norm_stats["action_mean"]
            ) / self.norm_stats["action_std"]
            qpos_data = torch.cat((qpos_data, last_action_data.flatten()))
        # print(f"qpos_data: {qpos_data.shape}, action_data: {action_data.shape}, image_data: {image_data.shape}, is_pad: {is_pad.shape}")
        #print("data shape", image_data.shape, qpos_data.shape, action_data.shape)
        #print("time", time.time()-startt)
        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes, ignore_force=True, 
                   iphone=False, depth_type="depth"):
    action_str = "qpos_action"
    all_qpos_data = []
    all_action_data = []
    all_episode_len = []
    all_depth = []
    all_ldepth = []
    all_depth_any = []
    depth_num = 3
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(
            dataset_dir, f"processed_episode_{episode_idx}.hdf5"
        )
        # print("loading", dataset_path)
        with h5py.File(dataset_path, "r") as root:
            if ignore_force:
                # print("episode_idx:", episode_idx, root.keys())
                qpos = root["observation.state"][:, :QPOS_DIM]
            else:
                qpos = root["observation.state"][()]
            action = root[action_str][()]
        if iphone != "none" and len(all_depth) <= depth_num:
            dataset_path = os.path.join(
                dataset_dir, f"iphone_episode_{episode_idx}.hdf5"
            )
            with h5py.File(dataset_path, "r") as root:
                all_depth.append(torch.from_numpy(root["depth"][()]))
                if iphone == "rgb_lowd":
                    all_ldepth.append(torch.from_numpy(root["lowres_depth"][()]))
                if depth_type == "depth_any":
                    all_depth_any.append(torch.from_numpy(root["depth_any"][()]))

        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
        all_episode_len.append(len(qpos))
    all_qpos_data = torch.cat(all_qpos_data)
    all_action_data = torch.cat(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(
        dim=0, keepdim=True
    )  # (episode, timstep, action_dim)
    action_std = all_action_data.std(dim=0, keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=0, keepdim=True)
    qpos_std = all_qpos_data.std(dim=0, keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # clipping

    stats = {
        "action_mean": action_mean.numpy().squeeze(),
        "action_std": action_std.numpy().squeeze(),
        "qpos_mean": qpos_mean.numpy().squeeze(),
        "qpos_std": qpos_std.numpy().squeeze(),
        "example_qpos": qpos,
    }
    if iphone != "none":
        all_depth = torch.cat(all_depth)
        stats["depth_mean"] = all_depth.mean()
        stats["depth_std"] = torch.clip(all_depth.std(), 1e-2, np.inf)
        if iphone == "rgb_lowd":
            all_ldepth = torch.cat(all_ldepth)
            stats["ldepth_mean"] = all_ldepth.mean()
            stats["ldepth_std"] = torch.clip(all_ldepth.std(), 1e-2, np.inf)
            
        if depth_type == "depth_any":
            all_depth_any = torch.cat(all_depth_any)
            stats["depth_mean"] = all_depth_any.mean()
            stats["depth_std"] = torch.clip(all_depth_any.std(), 1e-2, np.inf) 

    return stats, all_episode_len


def find_all_processed_episodes(path):
    # episodes = [f for f in os.listdir(path)]
    # print("Found episodes: ", episodes[0])
    import glob
    episodes = glob.glob(os.path.join(path, "processed_episode_*.hdf5"))
    episodes = [os.path.basename(e) for e in episodes]
    # print("Found episodes: ", episodes[0])
    return episodes


def BatchSampler(batch_size, episode_len_l, sample_weights=None):
    sample_probs = (
        np.array(sample_weights) / np.sum(sample_weights)
        if sample_weights is not None
        else None
    )
    sum_dataset_len_l = np.cumsum(
        [0] + [np.sum(episode_len) for episode_len in episode_len_l]
    )
    while True:
        batch = []
        for _ in range(batch_size):
            episode_idx = np.random.choice(len(episode_len_l), p=sample_probs)
            step_idx = np.random.randint(
                sum_dataset_len_l[episode_idx], sum_dataset_len_l[episode_idx + 1]
            )
            batch.append(step_idx)
        yield batch


def load_data(dataset_dir, camera_names, batch_size_train, batch_size_val, 
            #   ignore_force, ignore_hand, 
              state_name, left_right_mask,
              chop_data, progress_bar,
              iphone="none",
              iphone_depth_type="depth",
              history_stack=0,
              onehot_task_id=None,
              cancel_unsafe=False,
              ):
    print(f"\nData from: {dataset_dir}\n")
    ignore_force = "force" not in state_name

    all_eps = find_all_processed_episodes(dataset_dir)
    num_episodes = len(all_eps)

    if onehot_task_id is not None:
        assert onehot_task_id.shape[0] == num_episodes
    
    # obtain train test split
    train_ratio = 0.9
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[: int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes) :]
    print(f"Train episodes: {len(train_indices)}, Val episodes: {len(val_indices)}")
    print(f"Val indices: {val_indices}")
    # obtain normalization stats for qpos and action
    norm_stats, all_episode_len = get_norm_stats(dataset_dir, num_episodes, ignore_force, iphone, iphone_depth_type)

    train_episode_len_l = [all_episode_len[i] for i in train_indices]
    val_episode_len_l = [all_episode_len[i] for i in val_indices]

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(
        train_indices, 
        dataset_dir, 
        camera_names, 
        norm_stats, 
        train_episode_len_l, 
        # ignore_force=ignore_force, 
        # ignore_hand=ignore_hand,
        state_name=state_name,
        left_right_mask=left_right_mask,
        chop_data=chop_data,
        progress_bar=progress_bar,
        iphone_image=iphone,
        iphone_depth_type = iphone_depth_type,
        history_stack=history_stack,
        onehot_task_id=onehot_task_id,
        cancel_unsafe=cancel_unsafe,
    )
    val_dataset = EpisodicDataset(
        val_indices, 
        dataset_dir, 
        camera_names, 
        norm_stats, 
        val_episode_len_l, 
        # ignore_force=ignore_force, 
        # ignore_hand=ignore_hand,
        state_name=state_name,
        left_right_mask=left_right_mask,
        chop_data=chop_data,
        progress_bar=progress_bar,
        iphone_image=iphone,
        iphone_depth_type = iphone_depth_type,
        history_stack=history_stack,
        onehot_task_id=onehot_task_id,
        cancel_unsafe=cancel_unsafe,
    )
    train_indices = train_dataset.episode_ids
    val_indices = val_dataset.episode_ids

    train_episode_len_l = [all_episode_len[i] for i in train_indices]
    val_episode_len_l = [all_episode_len[i] for i in val_indices]
    batch_sampler_train = BatchSampler(batch_size_train, train_episode_len_l)
    batch_sampler_val = BatchSampler(batch_size_val, val_episode_len_l, None)

    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler_train,
        pin_memory=True,
        num_workers=1 if (chop_data is not None) else 16,
        prefetch_factor=2,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_sampler=batch_sampler_val,
        pin_memory=True,
    )

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose


### helper functions


def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def parse_id(base_dir, prefix):
    base_path = Path(base_dir)
    # Ensure the base path exists and is a directory
    if not base_path.exists() or not base_path.is_dir():
        raise ValueError(
            f"The provided base directory does not exist or is not a directory: \n{base_path}"
        )

    # Loop through all subdirectories of the base path
    for subfolder in base_path.iterdir():
        # if subfolder.is_dir() and subfolder.name == prefix:
        if subfolder.is_dir() and subfolder.name.startswith(prefix):
            print(f"Found subfolder: {subfolder}")
            return str(subfolder), subfolder.name

    # If no matching subfolder is found
    return None, None


def find_all_ckpt(base_dir, prefix="policy_epoch_"):
    base_path = Path(base_dir)
    # Ensure the base path exists and is a directory
    if not base_path.exists() or not base_path.is_dir():
        raise ValueError(
            "The provided base directory does not exist or is not a directory."
        )

    ckpt_files = []
    for file in base_path.iterdir():
        if file.is_file() and file.name.startswith(prefix):
            ckpt_files.append(file.name)
    # find latest ckpt
    ckpt_files = sorted(
        ckpt_files, key=lambda x: int(x.split(prefix)[-1].split("_")[0]), reverse=True
    )
    epoch = int(ckpt_files[0].split(prefix)[-1].split("_")[0])
    return ckpt_files[0], epoch
