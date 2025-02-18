import os
import shutil
import random
from collections import defaultdict
import argparse
import yaml

import numpy as np


def load_train_config_from_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    
    default_config = config["DefaultCfg"]
    skill_map = {}
    subdir_process_map = {}
    
    for skill in config['SkillCfg']:
        skill_config = default_config.copy()
        skill_config.update(skill)
        
        skill_map[skill_config["name"]] = [(
            skill_config["data"], skill_config["label"]
        )]    

        if "subdirs_to_block" in skill.keys():
            subdir_process_map[skill_config["name"]] = (
                skill_config["subdirs_to_block"],
                skill_config["total_episodes_num"],
                skill_config["adopt_ratio"]
            )
        
    return skill_map, subdir_process_map



def create_link(fromf, tof, idx, overwrite=False):
    dst_fname = tof
    # print("is dst", os.path.islink(dst_fname))
    if os.path.islink(dst_fname) and overwrite:
        os.remove(dst_fname)
    os.symlink(fromf, dst_fname)
    print("ls -n ", os.readlink(dst_fname), dst_fname)
    idx += 1
    return idx

def get_absolute_path(base_dir, path):
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(base_dir, path))

    
if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--src_dir", type=str, default="../dataset/")
    ap.add_argument("--dst_dir", type=str, default="data/recordings/")
    ap.add_argument("--config", type=str, default="../dataset/train_config.yaml")
    ap.add_argument("--overwrite", action="store_true")
    
    args = ap.parse_args()
    
    working_dir = os.getcwd()
    src_dir = get_absolute_path(working_dir, args.src_dir)
    dst_dir = get_absolute_path(working_dir, args.dst_dir)
    
    skill_map, subdir_process_map = load_train_config_from_yaml(args.config)
    overwrite = args.overwrite
    
    # find all path under src_dir
    for k, v in skill_map.items():
        dst_task_dir = os.path.join(dst_dir, k, "processed")
        if os.path.exists(dst_task_dir):
            shutil.rmtree(dst_task_dir)
        os.makedirs(dst_task_dir, exist_ok=True)
        idx = 0
        for data_path in v:
            label_id = None
            if isinstance(data_path, tuple):
                data_path, label_id = data_path
            full_data_path = os.path.join(src_dir, data_path)

            if not os.path.exists(full_data_path):
                print(f"Data path {full_data_path} does not exist, do process first, it will be skipped.")
                continue
            
            subdirs = os.listdir(full_data_path)
            subdirs.sort()
            subdirs_to_block, total_episodes_num, adopt_ratio = subdir_process_map.get(k, ([], -1, 0))
            
            subdirs_to_adopt = [fpath for fpath in subdirs if fpath not in subdirs_to_block]
            episodes_to_adopt = []
            episodes_to_adopt_extra = []
            
            if label_id is not None:
                
                # adopt all episode under subdirs_to_adopt
                for fpath in subdirs_to_adopt:
                    labeled_fpath = os.path.join(full_data_path, fpath, f"processed_{label_id}")
                    if os.path.isdir(labeled_fpath):
                        for label_file in os.listdir(labeled_fpath):
                            episodes_to_adopt.append(os.path.join(labeled_fpath, label_file))
                            
                # block (1-ratio) episodes under subdirs_to_block, adopt the rest (ratio)
                for fpath in subdirs_to_block:
                    labeled_fpath = os.path.join(full_data_path, fpath, f"processed_{label_id}")
                    if os.path.isdir(labeled_fpath):
                        for label_file in os.listdir(labeled_fpath):
                            episodes_to_adopt_extra.append(os.path.join(labeled_fpath, label_file))
                

                if adopt_ratio > 0:
                    adopt_num_in_extra = int(np.ceil(len(episodes_to_adopt)*(adopt_ratio/(1-adopt_ratio))))
                    adopt_num_in_adopt = total_episodes_num - adopt_num_in_extra
                    
                    random.shuffle(episodes_to_adopt)
                    random.shuffle(episodes_to_adopt_extra)
                    # episodes_to_adopt.extend(episodes_to_adopt_extra[:adopt_num_in_extra])
                    episodes_chosen = episodes_to_adopt[:adopt_num_in_adopt] + episodes_to_adopt_extra[:adopt_num_in_extra]
                    
                    print(f"=== Adopt {adopt_num_in_extra} extra episodes ===")
                    for f in episodes_to_adopt_extra[:adopt_num_in_extra]:
                        print(f)
                    print("=== That is all ===\n")
                elif total_episodes_num > 0:
                    adopt_num_in_adopt = total_episodes_num
                    random.shuffle(episodes_to_adopt)
                    episodes_chosen = episodes_to_adopt[:adopt_num_in_adopt]
                else:
                    episodes_chosen = episodes_to_adopt
                
            else:
                for fpath in subdirs_to_adopt:
                    episodes_to_adopt.append(os.path.join(full_data_path, fpath, "processed.hdf5"))

                    
                episodes_chosen = episodes_to_adopt
                
            
            if label_id is not None:
                for fromf in episodes_chosen:
                    dst_fname= os.path.join(dst_task_dir, f"processed_episode_{idx}.hdf5")
                    idx = create_link(fromf, dst_fname, idx, overwrite)
            else:
                for fromf in episodes_chosen:
                    dst_fname= os.path.join(dst_task_dir, f"processed_episode_{idx}.hdf5")
                    idx = create_link(fromf, dst_fname, idx, overwrite)
