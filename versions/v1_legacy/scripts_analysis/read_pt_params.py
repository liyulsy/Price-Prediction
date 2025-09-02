import torch
import os

folder = "cache/hpo_timexer"

# 只读取以trial_开头、_best_model.pt结尾的文件
pt_files = [f for f in os.listdir(folder) if f.startswith("trial_") and f.endswith("_best_model.pt")]
pt_files.sort(key=lambda x: int(x.split('_')[1]))  # 按trial编号排序

for pt_file in pt_files:
    pt_path = os.path.join(folder, pt_file)
    print(f"\n参数文件: {pt_path}")
    params = torch.load(pt_path, map_location='cpu')
    print(f"参数总数: {len(params)}")
    for k, v in params.items():
        print(f"  {k}: {tuple(v.shape)}")
    print("-"*40) 