import json
from tqdm import tqdm
import os

os.makedirs('data/kg_', exist_ok=True)


path = 'data/kg/%s.jsonl'

new_path = 'data/kg_/%s.json'

for fname in ['train', 'valid', 'test']:
# for fname in ['train']:
    all = []
    with open(path % fname, 'r') as f:
        all_abs = json.load(f)
        for data in tqdm(all_abs):
            data.pop("date")
            all.append(data)
    with open(new_path % fname, 'w') as f:
        json.dump(all, f, indent=4)