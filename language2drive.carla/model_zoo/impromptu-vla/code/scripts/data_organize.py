import os
import json
import argparse
from tqdm import tqdm
arg = argparse.ArgumentParser()
arg.add_argument('--data_root', type=str, default='./ImpromptuData')
arg.add_argument('--split', type=str, default='train', choices=['train', 'val'])
args = arg.parse_args()
data_root = args.data_root
split = args.split

for dataset in sorted(os.listdir(data_root)):
    print(f"Processing {dataset}...")
    q1, q2, q3, q4, q5, q6, q7 = [], [], [], [], [], [], []
    qdict = {
        "q1": q1,
        "q2": q2,
        "q3": q3,
        "q4": q4,
        "q5": q5,
        "q6": q6,
        "q7": q7,
    }
    dataset_path = os.path.join(data_root, dataset)
    for pdata in sorted(os.listdir(dataset_path)):
        if split not in pdata:
            continue
        print(f"    Processing {pdata}...")
        cdata = {}
        pdata_path = os.path.join(dataset_path, pdata)
        for file in sorted(os.listdir(pdata_path)):
            idx = file.split('.')[0]
            if idx not in cdata:
                cdata[idx] = {}
        for idx in tqdm(cdata):
            for q in [f'q{i}' for i in range(1, 8)]:
                image_file = os.path.join(pdata_path, f'{idx}.{q}_images.json')
                with open(image_file, 'r') as f:
                    image_list = json.load(f)
                images = [os.path.join(pdata_path, f'{idx}.{view}.png') for view in image_list]
                if images:
                    question_path = os.path.join(pdata_path, f'{idx}.{q}_question.txt')
                    with open(question_path, 'r') as f:
                        question = f.read().strip()
                    answer_path = os.path.join(pdata_path, f'{idx}.{q}_answer.txt')
                    with open(answer_path, 'r') as f:
                        answer = f.read().strip()
                    qdict[q].append({
                        "id": idx,
                        "images": images,
                        "messages": [
                            {
                                "role": "user",
                                "content": question
                            },
                            {
                                "role": "assistant",
                                "content": answer
                            }
                        ]
                    })
                    
    for q, res in qdict.items():
        os.makedirs(f'./data/{split}/{dataset}', exist_ok=True)
        with open(os.path.join(f'./data/{split}/{dataset}', f'{q}.json'), 'w') as f:
            json.dump(res, f, indent=4)
