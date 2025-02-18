import os
import sys
import argparse
import yaml


def load_train_config_from_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    
    default_config = config["DefaultCfg"]
    train_map = {}
    
    for skill in config['SkillCfg']:
        skill_config = default_config.copy()
        skill_config.update(skill)
        
        train_map[skill_config["name"]] = skill_config
    
    return train_map

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="../dataset/train_config.yaml")
    ap.add_argument("--exptid", type=str)
    args = ap.parse_args()
    
    train_map = load_train_config_from_yaml(args.config)
    skill_id = args.exptid.split('-')[0]
    skill_train_map = None
    for k,v in train_map.items():
        if k.startswith(skill_id):
            skill_train_map = v
            break
    if skill_train_map is not None:
        print("skill", skill_train_map["name"])
    else:
        raise ValueError(f"Skill {skill_id} not found")
    
    os.chdir(os.path.join(os.getcwd(), "act"))
    
    cmd = f"""
python imitate_episodes.py --policy_class ACT --kl_weight 10 \
--chunk_size 30 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 50001 \
--lr 5e-5 --seed 0 \
--taskid {skill_id} \
--exptid {args.exptid} \
--backbone dino_v2 \
--state_name "head" "arm" "wrist" "hand" \
--left_right_mask {skill_train_map["left_right_mask"]} \
--progress {skill_train_map["progress"]} \
--no_wandb \
"""
    if sum(skill_train_map["chop_data"]) != 0:
        cmd += f"--chop_data {skill_train_map['chop_data'][0]} {skill_train_map['chop_data'][1]} "
        
    os.system(cmd)