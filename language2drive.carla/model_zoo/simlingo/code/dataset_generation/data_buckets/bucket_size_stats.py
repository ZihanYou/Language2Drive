import json

file = 'database/bucketsv2_simlingo/buckets_stats.json'

with open(file, 'r') as f:
    data = json.load(f)

# get % of each bucket compared to 'total' bucket
bucket_relatives = {}
total = data['total']
for key in data.keys():
    if key == 'total':
        continue
    bucket_relatives[key] = data[key] / total

# pretty print
for key in bucket_relatives.keys():
    print(f'{key}: {bucket_relatives[key]*100:.2f}%')