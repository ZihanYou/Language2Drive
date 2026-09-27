#!/usr/bin/env python3
"""Evaluate complete predictions with the nuScenes paper 1.0 metric profile."""
import argparse
import csv
import json
from pathlib import Path

from paths import RELEASE_ROOT, DATA_ROOT
from release_io import load_release, sha256
from build_lane_pcus import MapGeometry
from model_eval.unified_semantic_metrics import make_corridors
from model_eval.paper_metrics import (PROFILE, CONDITIONS, WORDINGS, MAIN_METRICS,
    score_pair, summarize, language_controls)


def select_predictions(path, pcus, conditions, seed):
    by_id={p['pcu_id']:p for p in pcus}
    expected={(pid,a,c,seed) for pid in by_id for a in ('arm_0','arm_1') for c in conditions}
    selected={}
    with Path(path).open() as source:
        for number,line in enumerate(source,1):
            if not line.strip(): continue
            row=json.loads(line)
            if not isinstance(row,dict):
                raise ValueError('Prediction record must be an object at line '+str(number))
            key=tuple(row.get(k) for k in ('pcu_id','arm','condition','seed'))
            if not all(isinstance(k,str) for k in key[:3]) or type(key[3]) is not int:
                raise ValueError('Invalid prediction identity at line '+str(number))
            if key not in expected: continue
            if key in selected: raise ValueError('Duplicate prediction key: '+str(key))
            if row.get('public_input_hash')!=by_id[key[0]]['public_input']['public_input_hash']:
                raise ValueError('Prediction public input hash mismatch')
            selected[key]=row
    missing=expected-set(selected)
    if missing: raise ValueError('Incomplete requested run: {} missing outputs'.format(len(missing)))
    return selected


def evaluate_release(release_dir, predictions, output_dir, split='heldout', conditions=CONDITIONS,
                     seed=0, wordings=('canonical',)):
    conditions=tuple(conditions);wordings=tuple(wordings)
    if not conditions or len(set(conditions))!=len(conditions) or not set(conditions)<=set(CONDITIONS):
        raise ValueError('Invalid or duplicate conditions')
    if not wordings or len(set(wordings))!=len(wordings) or not set(wordings)<=set(WORDINGS)&set(conditions):
        raise ValueError('Headline wordings must be unique requested canonical/paraphrase conditions')
    output_dir=Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError('Use an empty output directory to preserve prior results')
    manifest,pcus=load_release(Path(release_dir),include_private=True,split=split)
    selected=select_predictions(predictions,pcus,conditions,seed)
    maps,corridors=make_corridors(pcus)
    for p in pcus:
        location=p['public_input']['location']
        if location not in maps: maps[location]=MapGeometry(location)
    rows=[]
    for p in pcus:
        for c in conditions:
            trajectories=[selected[(p['pcu_id'],a,c,seed)].get('trajectory') for a in ('arm_0','arm_1')]
            rows.append(dict(condition=c,seed=seed,**score_pair(p,trajectories,maps,corridors)))
    headline=summarize(rows,wordings)
    report=dict(profile=PROFILE,release_version=manifest['release_version'],split=split,seed=seed,
                headline_wordings=list(wordings),scale='0-100; SRG -100..100; paired_ADE_m in meters',
                aggregation='pair arms, mean wordings within PCU, equal PCU mean',
                headline=headline,per_condition={c:summarize(rows,(c,)) for c in conditions},
                language_controls=language_controls(rows))
    if set(WORDINGS)<=set(conditions):
        report['four_wording_supplement']=summarize(rows,WORDINGS)
    output_dir.mkdir(parents=True,exist_ok=True)
    (output_dir/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (output_dir/'per_pcu.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
    with (output_dir/'metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,lineterminator='\n',fieldnames=['condition','group','metric','mean','ci95_low','ci95_high','n','unit'])
        writer.writeheader()
        for condition,groups in [('headline',headline)]+list(report['per_condition'].items()):
            for group,entry in groups.items():
                for metric,stat in entry['metrics'].items():
                    writer.writerow(dict(condition=condition,group=group,metric=metric,mean=stat['mean'],
                        ci95_low=stat['ci95'][0],ci95_high=stat['ci95'][1],n=stat['n'],unit=stat['unit']))
    code_root=Path(__file__).resolve().parent
    files=['evaluate_predictions.py','score_lane_pcus.py','build_lane_pcus.py','release_io.py',
           'model_eval/paper_metrics.py','model_eval/semantic_metrics.py','model_eval/unified_semantic_metrics.py']
    provenance=dict(profile=PROFILE,release_manifest_sha256=sha256(Path(release_dir)/'release_manifest.json'),
                    public_sha256=manifest['public_sha256'],private_sha256=manifest['private_sha256'],
                    predictions_sha256=sha256(predictions),selected_outputs=len(selected),split=split,
                    conditions=list(conditions),wordings=list(wordings),seed=seed,
                    code_sha256={f:sha256(code_root/f) for f in files},
                    map_sha256={loc:sha256(DATA_ROOT/'maps/expansion'/(loc+'.json')) for loc in sorted(maps)})
    (output_dir/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-dir',type=Path,default=RELEASE_ROOT)
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--split',choices=('dev','heldout','all'),default='heldout')
    parser.add_argument('--conditions',default=','.join(CONDITIONS))
    parser.add_argument('--wordings',default='canonical',help='Headline wording set; controls are never wordings')
    parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args()
    if not (args.release_dir/'private_sidecars.jsonl').is_file():
        parser.error('Evaluation contracts missing; run python scripts/download_assets.py')
    try:
        report=evaluate_release(args.release_dir,args.predictions,args.output_dir,args.split,
            [c.strip() for c in args.conditions.split(',')],args.seed,[w.strip() for w in args.wordings.split(',')])
    except (ValueError,FileNotFoundError) as error:
        parser.error(str(error))
    print(json.dumps({k:report['headline']['all']['metrics'][k]['mean'] for k in MAIN_METRICS},indent=2))


if __name__=='__main__':main()
