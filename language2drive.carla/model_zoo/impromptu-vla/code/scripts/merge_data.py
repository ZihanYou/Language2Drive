import os
import json
import argparse
arg = argparse.ArgumentParser()
arg.add_argument('--data_root', type=str, default='./data')
arg.add_argument('--split', type=str, default='train', choices=['train', 'val'])
arg.add_argument('--output_dir', type=str, default='jsons')
args = arg.parse_args()
data_root = args.data_root
split = args.split
root = os.path.join(data_root, split)
output = os.path.join(data_root, args.output_dir, split)
os.makedirs(output, exist_ok = True)
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
for dataset in sorted(os.listdir(root)):
    for q, res in qdict.items():
        file = os.path.join(root, dataset, f"{q}.json")
        with open(file, 'r') as f:
            res += json.load(f)
for q, res in qdict.items():
    output_file = os.path.join(output, f"{q}.json")
    with open(output_file, 'w') as f:
        json.dump(res, f, indent=4)
with open(os.path.join(output, f"{split}.json"), 'w') as f:
    json.dump(q1 + q2 + q3 + q4 + q5 + q6 + q7, f, indent=4)