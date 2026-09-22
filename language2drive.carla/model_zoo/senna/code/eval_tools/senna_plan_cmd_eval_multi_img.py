import copy
import json

from tqdm import tqdm
from llava.model.builder import load_pretrained_model, load_senna_pretrained_model
from llava.mm_utils import get_model_name_from_path

from data_tools.senna_qa_utils import eval_multi_img_model_wo_init


eval_data_path = '/path/to/your/eval/data/eval_plan_qa.json'
model_path = "/path/to/the/model"
save_path = "/path/to/save/eval/result/eval_result.json"
save_path = '/job_data/pred_result.json'

# load vlm model and generate
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_senna_pretrained_model(
    model_path, None, model_name='llava', device_map=0)


with open(eval_data_path, 'r') as file:
    eval_data = json.load(file)

tot_num, correct_num = 0, 0

SPEED_PLAN = ['KEEP', 'ACCELERATE', 'DECELERATE', 'STOP']
PATH_PLAN = ['RIGHT_TURN', 'RIGHT_CHANGE', 'LEFT_TURN', 'LEFT_CHANGE', 'STRAIGHT']

metric_tot_cnt = {speed + '_' + path: 0 for speed in SPEED_PLAN for path in PATH_PLAN}
metric_correct_cnt = copy.deepcopy(metric_tot_cnt)

eval_record = {}

speed_tp = {'KEEP': 0,
            'ACCELERATE': 0,
            'DECELERATE': 0,
            'STOP': 0}

speed_fp, speed_fn = copy.deepcopy(speed_tp), copy.deepcopy(speed_tp)

path_tp = {'RIGHT_TURN': 0,
           'LEFT_TURN': 0,
           'STRAIGHT': 0}

path_fp, path_fn = copy.deepcopy(path_tp), copy.deepcopy(path_tp)

f1_score = {'KEEP': 0,
            'ACCELERATE': 0,
            'DECELERATE': 0,
            'STOP': 0,
            'RIGHT_TURN': 0,
            'LEFT_TURN': 0,
            'STRAIGHT': 0}

for sample in tqdm(eval_data):
    img_path = sample['image']
    question = sample['conversations'][0]['value']

    if 'SPEED plan' in question:
        args = type('Args', (), {
            "model_path": model_path,
            "model_base": None,
            "query": question,
            "conv_mode": 'llava_v1',
            "image_file": sample['images'],
            "sep": ",",
            "temperature": 0,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": 512
        })()

        tot_num = tot_num + 1
        gt_answer = sample['conversations'][1]['value']
        answer = eval_multi_img_model_wo_init(args, tokenizer, model, image_processor)

        speed_plan, path_plan = gt_answer.split(', ')
        path_plan = path_plan.split('\n')[0]

        for key in speed_tp.keys():
            if key in answer:  # P
                if speed_plan in answer:
                    speed_tp[key] += 1  # TP
                else:
                    speed_fp[key] += 1  # FP
            else:  # N
                if key in speed_plan:
                    speed_fn[key] += 1  # FN

        for key in path_tp.keys():
            if key in answer:  # P
                if path_plan in answer:
                    path_tp[key] += 1  # TP
                else:
                    path_fp[key] += 1  # FP
            else:  # N
                if key in path_plan:
                    path_fn[key] += 1  # FN


        metric_tot_cnt[speed_plan+'_'+path_plan] += 1
        if speed_plan in answer and path_plan in answer:
            correct_num = correct_num + 1
            metric_correct_cnt[speed_plan+'_'+path_plan] += 1
        else:
            fail_case = {
                'gt': gt_answer,
                'pred': answer,
            }
            eval_record[sample['token']] = fail_case


for key in f1_score.keys():
    if key in speed_tp.keys():
        if speed_tp[key] + speed_fp[key] != 0:
            precision = speed_tp[key] / (speed_tp[key] + speed_fp[key])
        else:
            precision = 0
        if speed_tp[key] + speed_fn[key] != 0:
            recall = speed_tp[key] / (speed_tp[key] + speed_fn[key])
        else:
            recall = 0
        if precision + recall != 0:
            f1_score[key] = 2.0 * precision * recall / (precision + recall)
        else:
            f1_score[key] = 0
    if key in path_tp.keys():
        if path_tp[key] + path_tp[key] != 0:
            precision = path_tp[key] / (path_tp[key] + path_fp[key])
        else:
            precision = 0
        if path_tp[key] + path_fn[key] != 0:
            recall = path_tp[key] / (path_tp[key] + path_fn[key])
        else:
            recall = 0
        if precision + recall != 0:
            f1_score[key] = 2.0 * precision * recall / (precision + recall)
        else:
            f1_score[key] = 0

print("\n\n=========== F1 Score ===========\n\n")
for k, v in f1_score.items():
    print(f"{k}: {v}")
print("\n\n================================\n\n")

print(f'\nTotal Number: {tot_num}\n')
print(f'\nCorrect Number: {correct_num}\n')

print('\n------------------------------\n\n')
print(f"Planning Accuracy: {correct_num/tot_num * 100:.2f}%")
print('\n\n------------------------------\n')

for key in metric_tot_cnt.keys():
    if metric_tot_cnt[key] > 0:
        print(f"{key}: num: {metric_tot_cnt[key]}, correct num: {metric_correct_cnt[key]}, {100*metric_correct_cnt[key]/metric_tot_cnt[key]:.2f}%")

eval_record['summary'] = f'Total Number: {tot_num}'
eval_record['summary'] = eval_record['summary'] + '\n' + f'Correct Number: {correct_num}'
eval_record['summary'] = eval_record['summary'] + '\n' + f"Planning Accuracy: {correct_num/tot_num * 100:.2f}%"

for key in metric_tot_cnt.keys():
    if metric_tot_cnt[key] > 0:
        eval_record['summary'] = eval_record['summary'] + '\n' + \
            f"{key}: num: {metric_tot_cnt[key]}, correct num: {metric_correct_cnt[key]}, {100*metric_correct_cnt[key]/metric_tot_cnt[key]:.2f}%"

eval_record['f1_score'] = {}
for k, v in f1_score.items():
    eval_record['f1_score'][k] = v

with open(save_path, "w") as f:
    json.dump(eval_record, f)
    print(f'\nEval results saved to {save_path}\n')
