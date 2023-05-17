from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import json
import torch
import random
from torch.utils.data.distributed import DistributedSampler



class EntityDict(Dataset):
    def __init__(self, entity_dict_dir, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        path = entity_dict_dir + '/entities.json'
        self.id2entity = []
        idx = 0
        self.entity2idx = {}
        self.data = []
        self.entity_tokenize = []
        for obj in tqdm(json.load(open(path, 'r', encoding='utf-8'))):
            entity = obj['entity']
            if len(entity) > 0:
                self.id2entity.append(entity)
                self.entity2idx[entity] = idx

                tokenized_e = self.tokenizer(entity, truncation=True, max_length=128)
                entity_ids = tokenized_e.input_ids
                entity_mask = tokenized_e.attention_mask   

                out_dict = {
                    'entity_ids': torch.LongTensor(entity_ids),
                    'entity_mask': torch.LongTensor(entity_mask),
                    'entity_length': len(entity_ids),
                    'idx': idx
                }
                self.entity_tokenize.append(tokenizer.encode(entity))
                self.data.append(out_dict)
                idx += 1

    def entity_to_idx(self, entity):
        return self.entity2idx[entity]

    def idx_to_entity(self, idx):
        return self.id2entity[idx]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datum = self.data[idx]
        return datum

    def collate_fn(self, batch):
        batch_entry = {}
        B = len(batch)

        S_L = max(entry['entity_length'] for entry in batch)
        hr_token_ids = torch.ones(B, S_L, dtype=torch.long) * self.tokenizer.pad_token_id
        hr_mask = torch.zeros(B, S_L, dtype=torch.long)
        idxs = []

        for i, entry in enumerate(batch):
            hr_token_ids[i, :entry['entity_length']] = entry['entity_ids']

            hr_mask[i, :entry['entity_length']] = entry['entity_mask']

            idxs.append(entry['idx'])
        
        batch_entry['hr_token_ids'] = hr_token_ids

        batch_entry['hr_mask'] = hr_mask

        batch_entry['idxs'] = idxs

        return batch_entry

class T5Dataset(Dataset):
    def __init__(self, split='train', topk=-1, args=None, tokenizer=None,entity_dataset=None):
        super().__init__()
        fname = args.dataset_dir + '/%s.json' % split
        self.topk = topk
        self.tokenizer = tokenizer
        self.neg_num = args.neg_num
        self.entity_dataset = entity_dataset
        self.data = self.loadData(fname)

    def __len__(self):
        return len(self.data)

    def loadData(self, filename):
        data = []
        with open(filename, 'r') as f:
            for line in tqdm(f):
                cur_data = json.loads(line)
                input_ = cur_data["input"]
                output = cur_data["output"]
                ids = cur_data["id"]
                year = cur_data["year"]
                rel_sent = cur_data["rel_sent"]
                neg_sample = cur_data["neg_sample"]

                l_neg = len(neg_sample)
                if l_neg < self.neg_num:
                    random_select = random.sample(self.entity_dataset.id2entity, k=self.neg_num + 1 - l_neg)
                    for e in random_select:
                        if e != output:
                            neg_sample.append(e)

                source_id = self.tokenizer(input_, truncation=True, max_length=512).input_ids

                target_id = self.tokenizer(output, truncation=True, max_length=128).input_ids
                out_dict = {
                    'source': {'input':input_, 'forward':cur_data['forward'], 'id': ids, 'year': year, 'rel_sent':rel_sent },
                    'target': [output],
                    'entity': cur_data['entity'],
                    'neg_sample': neg_sample,
                    'input_length': len(source_id),
                    'input_ids': torch.LongTensor(source_id),
                    'target_ids': torch.LongTensor(target_id),
                    'target_length': len(target_id),
                }
                data.append(out_dict)
                if len(data) > self.topk and self.topk != -1:
                    return data
        return data

    def __getitem__(self, idx):
        datum = self.data[idx]

        negs = [datum['entity']]
        tmp = random.sample(datum['neg_sample'], k= min(self.neg_num-1,len(datum['neg_sample'])))
        negs.extend(tmp)

        neg_length = []
        neg_ids = []

        for neg in negs:
            neg_id = self.tokenizer(neg, truncation=True, max_length=128).input_ids
            neg_length.append(len(neg_id))
            neg_ids.append(torch.LongTensor(neg_id))
        
        datum['neg_length'] = neg_length
        datum['neg_ids'] = neg_ids

        return datum

    def collate_fn(self, batch):
        batch_entry = {}

        B = len(batch)

        S_L = max(entry['input_length'] for entry in batch)
        input_ids = torch.ones(B, S_L, dtype=torch.long) * self.tokenizer.pad_token_id
        attention_masks = torch.zeros(B, S_L, dtype=torch.long)

        T_L = max(entry['target_length'] for entry in batch)
        target_ids = torch.ones(B, T_L, dtype=torch.long) * self.tokenizer.pad_token_id

        N_L = max(neg_length for entry in batch for neg_length in entry['neg_length'])
        neg_ids = torch.ones(B * self.neg_num, N_L, dtype=torch.long) * self.tokenizer.pad_token_id

        targets = []
        sources= []

        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']
            target_ids[i, :entry['target_length']] = entry['target_ids']
            attention_masks[i, :entry['input_length']] = 1

            sources.append(entry['source'])
            targets.append(entry['target'])
            for j in range(self.neg_num):
                index = i  * self.neg_num + j
                neg_ids[index, :entry['neg_length'][j]] = entry['neg_ids'][j]

        batch_entry['input_ids'] = input_ids
        word_mask = target_ids != self.tokenizer.pad_token_id
        target_ids[~word_mask] = -100
        batch_entry['target_ids'] = target_ids
        batch_entry['neg_ids'] = neg_ids
        batch_entry['attention_masks'] = attention_masks
        batch_entry['neg_num_total'] = self.neg_num


        batch_entry['targets'] = targets
        batch_entry['sources'] = sources


        return batch_entry



def get_loader(args, split='train', mode='train', tokenizer=None,
               batch_size=32, workers=4, topk=-1, entity_dataset=None):
    if entity_dataset is None:
        entity_dataset = EntityDict(args.dataset_dir, tokenizer)

    sampler = None
    dataset = T5Dataset(
                split,
                topk=topk,
                args=args,
                tokenizer=tokenizer,
                entity_dataset=entity_dataset)
    if args.distributed:
        if mode == 'train':
            sampler = DistributedSampler(dataset)
        else:
            sampler = DistributedSampler(dataset, shuffle=False)
    else:
        sampler = None
    if mode == 'train':
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=(sampler is None),
            num_workers=workers, 
            pin_memory=True, 
            sampler=sampler,
            collate_fn=dataset.collate_fn)
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers, 
            pin_memory=True,
            sampler=sampler,
            shuffle=None if (sampler is not None) else False,
            collate_fn=dataset.collate_fn,
            drop_last=False)


    return loader, sampler, entity_dataset