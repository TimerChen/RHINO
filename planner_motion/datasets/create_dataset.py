import pandas as pd
import os
import shutil
import random
from collections import defaultdict

data_dir = "mydata/motions_processed"
skip_link_creation = True

# 读取csv文件
csv_path = f"{data_dir}/data_index.txt"
df = pd.read_csv(csv_path, header=None, names=['id', 'class', 'path'])

# 创建软连接目录
os.makedirs('soft_links', exist_ok=True)

# 初始化一个字典来存储每个类的ID
class_dict = defaultdict(list)

os.makedirs(f"{data_dir}", exist_ok=True)
os.makedirs(f"{data_dir}/human", exist_ok=True)
os.makedirs(f"{data_dir}/humanoid", exist_ok=True)
os.makedirs(f"{data_dir}/h1", exist_ok=True)
os.makedirs(f"{data_dir}/annots", exist_ok=True)
# 创建软连接并填充class_dict
for index, row in df.iterrows():
    j = row['id']
    src_path = os.path.join("../../", row['path'].strip(), )
    # dest_path = os.path.join('soft_links', f"{row['id']}.npy")
    
    # 创建软连接
    # "${dir_path}/human_data.npy" "motions_processed/human/${j}.npy"
    if skip_link_creation:
        # print(f"{data_dir}/human/{j}.npy", os.path.exists(f"{data_dir}/human/{j}.npy"))
        # if os.path.exists(f"{data_dir}/human/{j}.npy"):
        os.remove(f"{data_dir}/human/{j}.npy")
        # if os.path.exists(f"{data_dir}/humanoid/{j}.npy"):
        os.remove(f"{data_dir}/humanoid/{j}.npy")
        # if os.path.exists(f"{data_dir}/h1/{j}.npy"):
        os.remove(f"{data_dir}/h1/{j}.npy")
        os.symlink(os.path.join(src_path, "human_data.npy"), f"{data_dir}/human/{j}.npy")
        os.symlink(os.path.join(src_path, "humanoid_data.npy"), f"{data_dir}/humanoid/{j}.npy")
        os.symlink(os.path.join(src_path, "humanoid_h1_data.npy"), f"{data_dir}/h1/{j}.npy")
        with open(os.path.join(data_dir, "annots", f"{j}.txt"), "w") as f:
            f.write(row['class'].strip())
    
    
    # 将id加入到对应类的列表中
    class_dict[row['class']].append(row['id'])

# 打开train.txt和val.txt文件
with open(f"{data_dir}/train.txt", 'w') as train_file, open(f"{data_dir}/val.txt", 'w') as val_file:
    train_ids = []
    val_ids = []
    
    # 遍历每个类
    for cls, ids in class_dict.items():
        # 打乱id顺序
        random.shuffle(ids)
        # 每类选择20条放入train_ids，其余放入val_ids
        train_ids.extend(ids[:35])
        val_ids.extend(ids[35:])
    
    # 排序
    train_ids.sort()
    val_ids.sort()
    
    # 写入文件
    for id in train_ids:
        train_file.write(f"{id}\n")
    for id in val_ids:
        val_file.write(f"{id}\n")
print("任务完成！")
