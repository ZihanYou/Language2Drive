import pickle
import ujson

bucket_path = 'database/buckets_expertv4_0/buckets_paths.pkl'
save_path = 'database/buckets_expertv4_0'
buckets_stats = {}

with open(bucket_path, 'rb') as f:
    buckets = pickle.load(f)

for key, value in buckets.items():
    buckets_stats[key] = len(value)

# find unique bucket values
unique_bucket_values = set()
for key, value in buckets.items():
    unique_bucket_values.update(value)

buckets_stats['total'] = len(unique_bucket_values)

# save buckets stats as json
with open(f'{save_path}/buckets_stats.json', 'w') as f:
    ujson.dump(buckets_stats, f, indent=4)