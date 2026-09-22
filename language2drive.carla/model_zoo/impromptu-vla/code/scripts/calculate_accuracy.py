import json
import re
import os
import argparse
from collections import Counter
import re
from typing import List, Dict, Optional

parser = argparse.ArgumentParser(description='')
parser.add_argument('--gt_folder', type=str, required=True,
                    help='/path/to/gt/folder')
parser.add_argument('--pred_folder', type=str, required=True,
                    help='/path/to/pred/folder')
parser.add_argument('--save_path', type=str, required=True,
                    help='/path/to/save/results')

args = parser.parse_args()


def extract_multiple_fields(
    raw_text: str,
    keys: List[str],
    value_filters: Optional[Dict[str, str]] = None
) -> List[Dict[str, str]]:
    field_matches = {key: None for key in keys}

    for key in keys:
        pattern = rf'"{re.escape(key)}"\s*:\s*"([^"]+)"'
        all_matches = re.findall(pattern, raw_text)

        if value_filters and key in value_filters:
            all_matches = [v for v in all_matches if value_filters[key] in v]
        if all_matches:
            field_matches[key] = all_matches[0]
        else:
            field_matches[key] = None

    return field_matches


def count_matching_elements(l1, l2):
    c1 = Counter(l1)
    c2 = Counter(l2)
    return sum(min(c1[x], c2[x]) for x in c1)


class Accuracy_task:
    def __init__(self, type, pred, gt, formatted=True):
        self.type = type
        self.pred = pred
        self.formatted = formatted
        if self.formatted:
            self.pred = jsonalize(self.pred)
        self.gt = gt

    def execute(self):
        if self.type == 'q1':
            def extract_objects_generic(text):
                if "No" in text:
                    return []
                pattern = re.compile(
                    r"(?:a|an)\s+([a-zA-Z\s]+)\s+located\s+(\d+)\s+meters\s+ahead\s+of\s+me\s+and\s+(\d+)\s+meters\s+to\s+the\s+(left|right|front|back)"
                )

                matches = pattern.finditer(text)
                objects = []
                for match in matches:
                    obj_type = match.group(1).strip()  # .strip() 移除前后空格
                    distance_ahead = int(match.group(2))
                    offset_distance = int(match.group(3))
                    direction = match.group(4)

                    objects.append({
                        "type": obj_type,
                        "distance_ahead_m": distance_ahead,
                        "offset_distance_m": offset_distance,
                        "offset_direction": direction
                    })
                return objects

            gt_objects = extract_objects_generic(self.gt)
            pred_objects = extract_objects_generic(self.pred)
            if len(gt_objects) == 0 and len(pred_objects) == 0:
                return 1
            elif len(gt_objects) == 0 or len(pred_objects) == 0:
                return 0

            temp_pred_objects = list(pred_objects)
            temp_gt_objects = list(gt_objects)

            true_positives = 0
            false_positives = 0
            false_negatives = 0
            matched_gt_indices = set()

            for pred_obj in temp_pred_objects:
                found_match = False
                for i, gt_obj in enumerate(temp_gt_objects):
                    if i in matched_gt_indices:
                        continue
                    if (pred_obj["type"] == gt_obj["type"]):
                        true_positives += 1
                        found_match = True
                        matched_gt_indices.add(i)
                        break
                if not found_match:
                    false_positives += 1

            false_negatives = len(temp_gt_objects) - true_positives

            precision = true_positives / \
                (true_positives + false_positives) if (true_positives +
                                                       false_positives) > 0 else 0
            recall = true_positives / \
                (true_positives + false_negatives) if (true_positives +
                                                       false_negatives) > 0 else 0

            f1_score = (2 * precision * recall) / (precision +
                                                   recall) if (precision + recall) > 0 else 0

            return f1_score

        elif self.type == 'q2':
            pattern = r"Object \d+:\s*(\w+),\s*(\w+)"
            results = re.findall(pattern, self.gt)
            gt_num = len(results)
            if isinstance(self.pred, str) and self.formatted == True:
                # print(self.pred)
                pattern = r'"speed_decision"\s*:\s*"([^"]+)"\s*,\s*"path_decision"\s*:\s*"([^"]+)"'
                pred_results = re.findall(pattern, self.pred)
                # print(pred_results)
                # calculate the speed ratio
                set1_pr = [pred_results[i][0]
                           for i in range(len(pred_results))]
                set1_gt = [results[i][0] for i in range(len(results))]
                a1_num = count_matching_elements(set1_pr, set1_gt)
                # calculate the path ratio
                set2_pr = [pred_results[i][1]
                           for i in range(len(pred_results))]
                set2_gt = [results[i][1] for i in range(len(results))]
                a2_num = count_matching_elements(set2_pr, set2_gt)
                # calculate both
                set3_pr = [(pred_results[i][0], pred_results[i][1])
                           for i in range(len(pred_results))]
                set3_gt = [(results[i][0], results[i][1])
                           for i in range(len(results))]
                a3_num = count_matching_elements(set3_pr, set3_gt)
                # calculate results
                if gt_num == 0:
                    return (int(a1_num == 0), int(a2_num == 0), int(a3_num == 0))
                else:
                    return (a1_num/gt_num, a2_num/gt_num, a3_num/gt_num)

            elif self.formatted == True and not isinstance(self.pred, str):
                # calculate the speed ratio
                # print(self.pred)
                set1_pr = [self.pred[i]['prediction']['speed_decision']
                           for i in range(len(self.pred))]
                set1_gt = [results[i][0] for i in range(len(results))]
                a1_num = count_matching_elements(set1_pr, set1_gt)
                # calculate the path ratio
                set2_pr = [self.pred[i]['prediction']['path_decision']
                           for i in range(len(self.pred))]
                set2_gt = [results[i][1] for i in range(len(results))]
                a2_num = count_matching_elements(set2_pr, set2_gt)
                # calculate both
                set3_pr = [(self.pred[i]['prediction']['speed_decision'], self.pred[i]
                            ['prediction']['path_decision']) for i in range(len(self.pred))]
                set3_gt = [(results[i][0], results[i][1])
                           for i in range(len(results))]
                a3_num = count_matching_elements(set3_pr, set3_gt)

                # calculate results
                if gt_num == 0:
                    return (int(a1_num == 0), int(a2_num == 0), int(a3_num == 0))
                else:
                    return (a1_num/gt_num, a2_num/gt_num, a3_num/gt_num)
            else:
                print(self.pred)
                pred_results = re.findall(pattern, self.pred)
                # calculate the speed ratio
                set1_pr = [pred_results[i][0]
                           for i in range(len(pred_results))]
                set1_gt = [results[i][0] for i in range(len(results))]
                a1_num = count_matching_elements(set1_pr, set1_gt)
                # calculate the path ratio
                set2_pr = [pred_results[i][1]
                           for i in range(len(pred_results))]
                set2_gt = [results[i][1] for i in range(len(results))]
                a2_num = count_matching_elements(set2_pr, set2_gt)
                # calculate both
                set3_pr = [(pred_results[i][0], pred_results[i][1])
                           for i in range(len(pred_results))]
                set3_gt = [(results[i][0], results[i][1])
                           for i in range(len(results))]
                a3_num = count_matching_elements(set3_pr, set3_gt)
                # calculate results
                if gt_num == 0:
                    return (int(a1_num == 0), int(a2_num == 0), int(a3_num == 0))
                else:
                    return (a1_num/gt_num, a2_num/gt_num, a3_num/gt_num)

        elif self.type == 'q3':
            pass
        elif self.type == 'q4':
            if isinstance(self.pred, str):
                if 'None' in self.pred:
                    self.pred = 'None'
                if 'Red' in self.pred:
                    self.pred = 'Red'
                if 'Green' in self.pred:
                    self.pred = 'Green'
                if 'Yellow' in self.pred:
                    self.pred = 'Yellow'
            else:
                self.pred = self.pred['traffic_light_status']
            if str(self.gt) == self.pred:
                return 1
            else:
                # print(self.pred,self.type)
                return 0
        elif self.type == 'q5':
            pass
        elif self.type == 'q6':
            pattern = r"<SPEED PATH PLAN>\s*([A-Z+_]+)\s*,\s*([A-Z+_]+)\s*</SPEED PATH PLAN>"
            if isinstance(self.pred, str):
                search_result = re.findall(pattern, self.pred)
                if len(search_result):
                    self.pred = {
                        'Velocity Plan': search_result[0][0], 'Path Plan': search_result[0][1]}
                    # print(self.pred)
                else:
                    self.pred = extract_multiple_fields(
                        self.pred, ['Velocity Plan', 'Path Plan'])
            results = re.findall(pattern, self.gt)[0]
            a1 = 0
            a2 = 0
            a3 = 0
            if self.pred.get('Velocity Plan', "") == results[0]:
                a1 = 1
            if self.pred.get('Path Plan', "") == results[1]:
                a2 = 1
            if a1 == 1 and a2 == 1:
                a3 = 1
            return (a1, a2, a3)
        elif self.type == 'q7':

            pattern = r'\[(-?\d+\.\d+),\s*(-?\d+\.\d+)\]'
            matches = re.findall(pattern, self.gt)

            gt_coordinates = [(float(x), float(y)) for x, y in matches]
            if self.pred == {}:
                return None
            # print(self.pred)
            if isinstance(self.pred, str):
                pattern = r'\[\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*\]'
                matches = re.findall(pattern, self.pred)
                self.pred = [(float(x), float(y)) for x, y in matches]
                try:
                    self.pred = {
                        'predicted_waypoints': {
                            't+0.5s': self.pred[0],
                            't+1.0s': self.pred[1],
                            't+1.5s': self.pred[2],
                            't+2.0s': self.pred[3],
                            't+2.5s': self.pred[4],
                            't+3.0s': self.pred[5],
                            't+3.5s': self.pred[6],
                            't+4.0s': self.pred[7],
                            't+4.5s': self.pred[8],
                            't+5.0s': self.pred[9] if len(self.pred) >= 10 else self.pred[8],

                        }
                    }
                except:
                    self.pred = {
                        'predicted_waypoints': {
                            't+0.5s': [0, 0],
                            't+1.0s': [0, 0],
                            't+1.5s': [0, 0],
                            't+2.0s': [0, 0],
                            't+2.5s': [0, 0],
                            't+3.0s': [0, 0],
                            't+3.5s': [0, 0],
                            't+4.0s': [0, 0],
                            't+4.5s': [0, 0],
                            't+5.0s': [0, 0]
                        }
                    }
            pred_coordinates = [
                self.pred['predicted_waypoints']['t+0.5s'],
                self.pred['predicted_waypoints']['t+1.0s'],
                self.pred['predicted_waypoints']['t+1.5s'],
                self.pred['predicted_waypoints']['t+2.0s'],
                self.pred['predicted_waypoints']['t+2.5s'],
                self.pred['predicted_waypoints']['t+3.0s'],
                self.pred['predicted_waypoints']['t+3.5s'],
                self.pred['predicted_waypoints']['t+4.0s'],
                self.pred['predicted_waypoints']['t+4.5s'],
                self.pred['predicted_waypoints']['t+5.0s'],
            ]
            l2_loss = 0
            loss_batch = []
            # print(gt_coordinates,pred_coordinates)
            for i in range(len(gt_coordinates)):
                l2_1 = ((gt_coordinates[i][0]-pred_coordinates[i][0]) **
                        2+(gt_coordinates[i][1]-pred_coordinates[i][1])**2)**0.5
                l2_2 = ((gt_coordinates[i][0]+pred_coordinates[i][0]) **
                        2+(gt_coordinates[i][1]-pred_coordinates[i][1])**2)**0.5
                l2_3 = ((gt_coordinates[i][0]-pred_coordinates[i][0]) **
                        2+(gt_coordinates[i][1]+pred_coordinates[i][1])**2)**0.5
                l2_4 = ((gt_coordinates[i][0]+pred_coordinates[i][0]) **
                        2+(gt_coordinates[i][1]+pred_coordinates[i][1])**2)**0.5
                l2_loss += min(l2_1, l2_2, l2_3, l2_4)
                if i == 1:
                    loss_batch.append(l2_loss/2)
                if i == 3:
                    loss_batch.append(l2_loss/4)
                if i == 5:
                    loss_batch.append(l2_loss/6)
                if i == 7:
                    loss_batch.append(l2_loss/8)
                if i == 9:
                    loss_batch.append(l2_loss/10)
            return loss_batch


def get_sorted_paths(dirname):
    filelist = os.listdir(dirname)
    filelist.sort()
    filelist = [os.path.join(dirname, file)
                for file in filelist if '.json' not in file]
    return filelist


gt_file = get_sorted_paths(args.gt_folder)

filelist = get_sorted_paths(args.pred_folder)

tasks_name = [file.split('/')[-1][:2] for file in gt_file]

all_pred_gt_pairs = [(task, file1, file2) for task, file1,
                     file2 in zip(tasks_name, gt_file, filelist)]

print(all_pred_gt_pairs)


def jsonalize(text):
    try:
        text = json.loads(text)
        return text
    except:
        pass
    try:
        text = text.split("```json\n")[1].split("\n```")[0]
        text = json.loads(text)
        return text
    except:
        return text


results_all = []
mode = 'sum'  # sum or detail
for q, gt_path, pred_path in all_pred_gt_pairs:
    print(f"Processing {q} {pred_path} {gt_path}...")

    with open(pred_path, 'r') as f:
        pred_data = [json.loads(line) for line in f]
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
    print(len(pred_data), len(gt_data))
    is_format = False
    tasks = [Accuracy_task(q, pred_data[i]['predict'], gt_data[i]['messages']
                           [1]['content'], is_format) for i in range(len(pred_data))]
    if q == 'q1':
        if mode == 'sum':
            correct = sum([task.execute() for task in tasks])
            all = len(tasks)
            accuracy = correct/all
            results_all.append(
                {'q': q, 'accuray': accuracy, 'filepath': pred_path, 'cnt': all})
        else:
            results = [task.execute() for task in tasks]
            results_all.append(
                {'q': q, 'filepath': pred_path, 'cnt': len(tasks), 'results': results})
    if q == 'q2':
        if mode == 'sum':
            correct1 = 0
            correct2 = 0
            correct3 = 0
            all = len(tasks)
            for task in tasks:
                a1, a2, a3 = task.execute()
                correct1 += a1
                correct2 += a2
                correct3 += a3
            accuracy1 = correct1/all
            accuracy2 = correct2/all
            accuracy3 = correct3/all
            results_all.append({'q': q, 'speed': accuracy1, 'path': accuracy2,
                               'both': accuracy3, 'filepath': pred_path, 'cnt': all})
        else:
            results = [list(task.execute()) for task in tasks]

            results_all.append(
                {'q': q, 'filepath': pred_path, 'cnt': len(tasks), 'results': results})
    if q == 'q4':
        if mode == 'sum':
            correct = sum([task.execute() for task in tasks])
            all = len(tasks)
            accuracy = correct/all
            results_all.append(
                {'q': q, 'accuray': accuracy, 'filepath': pred_path, 'cnt': all})
        else:
            results = [task.execute() for task in tasks]
            results_all.append(
                {'q': q, 'filepath': pred_path, 'cnt': len(tasks), 'results': results})
    if q == 'q6':
        if mode == 'sum':
            correct1 = 0
            correct2 = 0
            correct3 = 0
            all = len(tasks)
            for task in tasks:
                a1, a2, a3 = task.execute()
                correct1 += a1
                correct2 += a2
                correct3 += a3
            accuracy1 = correct1/all
            accuracy2 = correct2/all
            accuracy3 = correct3/all
            results_all.append({'q': q, 'speed': accuracy1, 'path': accuracy2,
                               'both': accuracy3, 'filepath': pred_path, 'cnt': all})
        else:
            results = [list(task.execute()) for task in tasks]

            results_all.append(
                {'q': q, 'filepath': pred_path, 'cnt': len(tasks), 'results': results})
    if q == 'q7':
        if mode == 'sum':
            all_loss_1 = 0
            all_loss_2 = 0
            all_loss_3 = 0
            all_loss_4 = 0
            all_loss_5 = 0
            all_cnt = 0
            for task in tasks:
                loss = task.execute()
                if loss is not None:
                    all_loss_1 += loss[0]
                    all_loss_2 += loss[1]
                    all_loss_3 += loss[2]
                    all_loss_4 += loss[3]
                    all_loss_5 += loss[4]

                    all_cnt += 1
            results_all.append({'q': q, 'loss_1': all_loss_1/all_cnt, 'loss_2': all_loss_2/all_cnt, 'loss_3': all_loss_3 /
                               all_cnt, 'loss_4': all_loss_4/all_cnt, 'loss_5': all_loss_5/all_cnt, 'pred_path': pred_path, 'cnt': all_cnt})
        else:
            results = [task.execute() for task in tasks]
            results_all.append(
                {'q': q, 'filepath': pred_path, 'cnt': len(tasks), 'results': results})

with open(args.save_path, 'w') as f:
    json.dump(results_all, f, indent=4)
