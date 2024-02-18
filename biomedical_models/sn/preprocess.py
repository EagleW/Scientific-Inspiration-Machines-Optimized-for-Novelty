import json
from tqdm import tqdm
import os

os.makedirs('data/sim_', exist_ok=True)


path = 'data/sim/%s.jsonl'

new_path = 'data/sim_/%s.json'

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