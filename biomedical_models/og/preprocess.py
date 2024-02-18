import json
from tqdm import tqdm
import os

os.makedirs('data/pubmed_data', exist_ok=True)


# path = 'data/pubmed_jsonl_data/%s.jsonl'
path = 'data/filter/%s.jsonl'

new_path = 'data/og/%s.json'

for fname in ['train', 'valid', 'test']:
# for fname in ['test']:
    all = []
    with open(path % fname, 'r') as f:
        for line in tqdm(f):
            data = json.loads(line)
            data.pop("date")
            all.append(data)
    with open(new_path % fname, 'w') as f:
        json.dump(all, f, indent=4)