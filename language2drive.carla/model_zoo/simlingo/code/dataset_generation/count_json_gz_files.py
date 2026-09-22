import json
import gzip
import glob

paths = [
    'database/simlingo/commentary',
    'database/simlingo/dreamer',
    'database/simlingo/drivelm',
]


for path in paths:
    # Get all .json.gz files in all subdirectories
    files = glob.glob(f'{path}/**/*.json.gz', recursive=True)
    # Count the number of files
    print(f'Found {len(files)} files in {path}')
    
