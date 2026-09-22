import os
import json
import argparse
from tqdm import tqdm
import re
def count_image_tags(text):
    # 定义正则表达式模式，用于匹配 <image> 标签
    pattern = r'<image>'
    
    # 使用 re.findall() 查找所有匹配的标签
    matches = re.findall(pattern, text)
    
    # 返回匹配的数量
    return len(matches)
import pdb
        
arg = argparse.ArgumentParser()
arg.add_argument('--data_root', type=str, default='./data')
arg.add_argument('--split', type=str, default='train', choices=['train', 'val'])
args = arg.parse_args()
data_root = args.data_root
split = args.split
root = os.path.join(data_root, split)
for dataset in sorted(os.listdir(root)):
    print(f"Processing {dataset}...")
    dataset_path = os.path.join(root, dataset)
    for file in sorted(os.listdir(dataset_path)):
        if '.json' not in file:
            continue
        print(f"    Processing {file}...")
        with open(os.path.join(dataset_path, file), 'r') as f:
            data = json.load(f)
        changed = False
        for item in data:
            images = item['images']
            question, answer = item['messages'][0]["content"], item['messages'][1]["content"]

            assert len(images) == count_image_tags(question)
            if 'q2' in file:
                if not (answer.startswith("<DYNAMIC OBJECTS>") and answer.endswith("</DYNAMIC OBJECTS>")):
                    assert "Please predict their future driving behaviors, which can be divided into SPEED decisions and PATH decisions." in question
                    item['messages'][1]["content"] = f'<DYNAMIC OBJECTS>{answer}</DYNAMIC OBJECTS>'
                    changed = True
            elif 'q6' in file:
                if not (answer.startswith("<SPEED PATH PLAN>") and answer.endswith("</SPEED PATH PLAN>")):
                    assert "what is your plan for the next three seconds?" in question
                    item['messages'][1]["content"] = f'<SPEED PATH PLAN>{answer}</SPEED PATH PLAN>'
                    changed = True
            elif 'q7' in file:
                if not (answer.startswith("<PLANNING>") and answer.endswith("</PLANNING>")):
                    assert "Your task is to do your best to predict future waypoints for the vehicle over the next 10 timesteps" in question
                    item['messages'][1]["content"] = f'<PLANNING>{answer}</PLANNING>'
                    changed = True
        if changed:
            print(f"    Changed {file}...")
            with open(os.path.join(dataset_path, f"{file}"), 'w') as f:
                json.dump(data, f, indent=4)
            