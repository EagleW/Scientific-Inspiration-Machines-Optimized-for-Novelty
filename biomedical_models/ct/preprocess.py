import json
from tqdm import tqdm
import os

os.makedirs('data/cite', exist_ok=True)


path = 'data/ct/%s.jsonl'

new_path = 'data/cite/%s.json'

for fname in ['train', 'valid', 'test']:
    all = []
    with open(path % fname, 'r') as f:
        for line in tqdm(f):
            data = json.loads(line)
            data.pop("date")
            all.append(data)
    with open(new_path % fname, 'w') as f:
        json.dump(all, f, indent=4)