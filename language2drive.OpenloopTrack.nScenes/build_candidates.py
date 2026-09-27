#!/usr/bin/env python3
"""Run the candidate construction and evidence pipeline, without auto-approval."""
import subprocess
import sys
from paths import PROJECT_ROOT,ARTIFACT_ROOT


def main():
    for folder in ('audit','candidates','scene_catalog','public_inputs','private_sidecars'):
        (ARTIFACT_ROOT/folder).mkdir(parents=True,exist_ok=True)
    scripts=['build_catalog.py','build_lane_pcus.py','build_speed_pcus.py',
             'combine_review_queues.py','audit_footprint.py','audit_footprint_swept.py',
             'audit_lidar.py','filter_candidates.py','export_review_queue.py','make_review_sheet.py']
    for script in scripts:
        subprocess.run([sys.executable,str(PROJECT_ROOT/script)],check=True,cwd=str(PROJECT_ROOT))
    subprocess.run([sys.executable,str(PROJECT_ROOT/'make_manifest.py'),'--split-only'],check=True,cwd=str(PROJECT_ROOT))
    subprocess.run([sys.executable,str(PROJECT_ROOT/'validate_artifacts.py')],check=True,cwd=str(PROJECT_ROOT))
    subprocess.run([sys.executable,str(PROJECT_ROOT/'run_model.py'),'--adapter-file',str(PROJECT_ROOT/'constant_velocity_adapter.py'),
                    '--output',str(ARTIFACT_ROOT/'model_outputs/constant_velocity.jsonl')],check=True,cwd=str(PROJECT_ROOT))
    subprocess.run([sys.executable,str(PROJECT_ROOT/'score_lane_pcus.py'),'--outputs',str(ARTIFACT_ROOT/'model_outputs/constant_velocity.jsonl'),
                    '--output-dir',str(ARTIFACT_ROOT/'scores/constant_velocity')],check=True,cwd=str(PROJECT_ROOT))
    subprocess.run([sys.executable,str(PROJECT_ROOT/'make_manifest.py')],check=True,cwd=str(PROJECT_ROOT))
    subprocess.run([sys.executable,str(PROJECT_ROOT/'render_review.py')],check=True,cwd=str(PROJECT_ROOT))
    print('Candidates are ready for per-case review. No approval was generated.')


if __name__=='__main__':main()
