import glob
import json
import gzip
import math
import os
from pathlib import Path
import subprocess
import tqdm

"""
Delete or rename runs in the dataset where the simulation crashed.
After renaming, check dataset size and number of frames with
du -hs data --exclude=FAILED_*
find data/*/[0-9]*/rgb | wc -l
or check scenario distribution with
echo "Number of succesful runs per scenario type"; for dir in data/*/; do echo $(ls "$dir" | grep -v FAILED_* | wc -l) - $dir; done | sort -rn -t " " -k 1
"""

RENAME = False  # Set to true to rename folders from failed runs with prefix
PREFIX = 'FAILED_'
DELETE = False
UNDO_RENAMING = False  # Use this to undo renaming

if __name__ == "__main__":

    data_save_root = 'database/simlingo/data'

    runs = glob.glob(f'{data_save_root}/**/*Town*', recursive=True)
    num_failed = 0

    for run in tqdm.tqdm(runs):
        if UNDO_RENAMING:
            p = Path(run)
            if p.stem.startswith(PREFIX):
                p.rename(Path(p.parent, p.stem[len(PREFIX):]))
        else:
            failed_flag = False
            results_file = run + '/results.json.gz'
            if not os.path.exists(results_file):
                failed_flag=True
            else:
                with gzip.open(results_file, 'r') as fin:        # 4. gzip
                    json_bytes = fin.read()                      # 3. bytes (i.e. UTF-8)
                json_str = json_bytes.decode('utf-8')            # 2. string (i.e. JSON)
                try:
                    data = json.loads(json_str)                      # 1. data

                    if data['status'] == 'Failed - Agent couldn\'t be set up':
                        #print('3')
                        failed_flag = True
                    if data['status'] == 'Failed':
                        #print('4')
                        failed_flag = True
                    if data['status'] == 'Failed - Simulation crashed':
                        #print('5')
                        failed_flag = True
                    if data['status'] == 'Failed - Agent crashed':
                        #print('6')
                        failed_flag = True
                    if data['scores']['score_composed'] < 100.0:  # we also count imperfect runs as failed (except minspeedinfractions)
                        cond1 = math.isclose(data['scores']['score_route'], 100)
                        cond2 = data['num_infractions'] == len(data['infractions']['min_speed_infractions'])
                        if not (cond1 and cond2):  # if the only problem is minspeedinfractions, keep it
                            failed_flag = True
                except:
                    print(f'Error while loading json in file {results_file}')
                    failed_flag = True
                

            if failed_flag:
                num_failed = num_failed + 1
                p = Path(run)
                if DELETE:
                    # use the special delete command for the ML cloud
                    print(f'Deleting {run}')
                    subprocess.call(['bash', '-c', f'find "{run}" -type f ! -size 0c | parallel -X --progress truncate -s0'])
                    subprocess.call(['bash', '-c', f'rm -rf "{run}"'])
                elif RENAME and not p.stem.startswith(PREFIX):
                    p.rename(Path(p.parent, PREFIX+p.stem))

    if not UNDO_RENAMING:
        print(f'Failed runs: {num_failed} (out of {len(runs)} in total)')
        print(f'Successful runs: {len(runs) - num_failed}')