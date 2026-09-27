#!/usr/bin/env python3
"""Recompute the nuScenes rows of paper Table 9 from frozen predictions."""
import argparse
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from paths import PROJECT_ROOT, RELEASE_ROOT
from evaluate_predictions import evaluate_release
from model_eval.paper_metrics import MAIN_METRICS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions-dir',type=Path,default=PROJECT_ROOT/'assets/predictions')
    p.add_argument('--release-dir',type=Path,default=RELEASE_ROOT)
    p.add_argument('--output-dir',type=Path,default=PROJECT_ROOT/'outputs/paper_replay')
    p.add_argument('--check-paper',action='store_true')
    args=p.parse_args()
    reference=json.loads((PROJECT_ROOT/'config/paper_table9.json').read_text())
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        p.error('Use an empty output directory')
    rows=[]
    for model in reference['models']:
        report=evaluate_release(args.release_dir,args.predictions_dir/model['filename'],
                                args.output_dir/model['id'])
        metrics=report['headline']['all']['metrics']
        row={'model':model['label'],**{k:metrics[k]['mean'] for k in MAIN_METRICS}}
        if args.check_paper:
            for name in MAIN_METRICS:
                if format(row[name],'.2f')!=format(model['paper'][name],'.2f'):
                    raise ValueError('Paper value mismatch: '+model['label']+' '+name)
        rows.append(row)
        print(json.dumps(row),flush=True)
    with (args.output_dir/'table9.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,lineterminator='\n',fieldnames=['model']+list(MAIN_METRICS))
        writer.writeheader();writer.writerows(rows)
    lines=['| Model | SemSISR | SemCFPS | SCS | PCS |','|---|---:|---:|---:|---:|']
    lines += ['| '+r['model']+' | '+' | '.join(format(r[k],'.2f') for k in MAIN_METRICS)+' |' for r in rows]
    (args.output_dir/'table9.md').write_text('\n'.join(lines)+'\n')
    print('Paper replay complete; metrics use the 0-100 scale.')


if __name__=='__main__':main()
