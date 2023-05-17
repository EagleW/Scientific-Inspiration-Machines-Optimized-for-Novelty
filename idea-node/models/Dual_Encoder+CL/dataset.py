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


class SimKGCDBERTLocalDataset(Dataset):
    def __init__(self, split='train', topk=-1, args=None, tokenizer=None, entity_dataset=None):
        super().__init__()
        fname = args.dataset_dir + '/%s.json' % split
        self.topk = topk
        self.tokenizer = tokenizer
        self.entity_dataset = entity_dataset
        self.neg_num = args.neg_num
        self.data = self.loadData(fname)

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
                entity = cur_data['entity']
                context = cur_data['context']
                neg_sample = cur_data["neg_sample"]

                l_neg = len(neg_sample)
                if l_neg < self.neg_num:
                    random_select = random.sample(self.entity_dataset.id2entity, k=self.neg_num + 2 - l_neg)
                    for e in random_select:
                        if e != entity and e != output:
                            neg_sample.append(e)


                tokenized_s = self.tokenizer(text=input_, text_pair=context,truncation=True, max_length=512)
                source_id = tokenized_s.input_ids
                s_token_type_id = tokenized_s.token_type_ids
                hr_mask = tokenized_s.attention_mask              

                tokenized_t = self.tokenizer(output, truncation=True, max_length=128)
                target_id = tokenized_t.input_ids
                tail_mask = tokenized_t.attention_mask   

                tokenized_e = self.tokenizer(entity, truncation=True, max_length=128)
                entity_id = tokenized_e.input_ids
                head_mask = tokenized_e.attention_mask   

                out_dict = {
                    'source': {
                        'input':input_, 
                        'context':context, 
                        'forward':cur_data['forward'], 
                        'id': ids, 
                        'year': year, 
                        'rel_sent':rel_sent },
                    'target': [output],
                    'entity': entity,
                    'neg_sample': neg_sample,
                    'input_length': len(source_id),
                    'hr_token_ids': torch.LongTensor(source_id),
                    'tail_token_ids': torch.LongTensor(target_id),
                    'head_token_ids': torch.LongTensor(entity_id),
                    'tail_length': len(target_id),
                    'head_length': len(entity_id),
                    'hr_mask': torch.LongTensor(hr_mask),
                    'tail_mask': torch.LongTensor(tail_mask),
                    'head_mask': torch.LongTensor(head_mask),
                    'hr_token_type_ids': torch.LongTensor(s_token_type_id),
                }
                data.append(out_dict)
                if len(data) > self.topk and self.topk != -1:
                    return data
        return data
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datum = self.data[idx]
        negs = random.sample(datum['neg_sample'], k= self.neg_num)
        neg_length = []
        neg_ids = []
        neg_masks = []

        for neg in negs:
            tokenized_n = self.tokenizer(neg, truncation=True, max_length=128)
            neg_id = tokenized_n.input_ids
            neg_mask = tokenized_n.attention_mask
            neg_length.append(len(neg_id))
            neg_ids.append(torch.LongTensor(neg_id))
            neg_masks.append(torch.LongTensor(neg_mask))
        
        datum['neg_length'] = neg_length
        datum['neg_ids'] = neg_ids
        datum['neg_mask'] = neg_masks
        return datum

    def collate_fn(self, batch):
        batch_entry = {}
        B = len(batch)

        S_L = max(entry['input_length'] for entry in batch)
        hr_token_ids = torch.ones(B, S_L, dtype=torch.long) * self.tokenizer.pad_token_id
        hr_mask = torch.zeros(B, S_L, dtype=torch.long)
        hr_token_type_ids = torch.zeros(B, S_L, dtype=torch.long)

        T_L = max(entry['tail_length'] for entry in batch)
        tail_token_ids = torch.ones(B, T_L, dtype=torch.long) * self.tokenizer.pad_token_id
        tail_mask = torch.zeros(B, T_L, dtype=torch.long) 

        H_L = max(entry['head_length'] for entry in batch)
        head_token_ids = torch.ones(B, H_L, dtype=torch.long) * self.tokenizer.pad_token_id
        head_mask = torch.zeros(B, H_L, dtype=torch.long) 

        N_L = max(neg_length for entry in batch for neg_length in entry['neg_length'])
        neg_ids = torch.ones(B * self.neg_num, N_L, dtype=torch.long) * self.tokenizer.pad_token_id
        neg_mask = torch.zeros(B * self.neg_num, N_L, dtype=torch.long) 

        targets = []
        sources= []
        for i, entry in enumerate(batch):
            hr_token_ids[i, :entry['input_length']] = entry['hr_token_ids']
            tail_token_ids[i, :entry['tail_length']] = entry['tail_token_ids']
            head_token_ids[i, :entry['head_length']] = entry['head_token_ids']

            hr_mask[i, :entry['input_length']] = entry['hr_mask']
            tail_mask[i, :entry['tail_length']] = entry['tail_mask']
            head_mask[i, :entry['head_length']] = entry['head_mask']

            for j in range(self.neg_num):
                index = i  * self.neg_num + j
                neg_ids[index, :entry['neg_length'][j]] = entry['neg_ids'][j]
                neg_mask[index, :entry['neg_length'][j]] = entry['neg_mask'][j]

            hr_token_type_ids[i, :entry['input_length']] = entry['hr_token_type_ids']

            sources.append(entry['source'])
            targets.append(entry['target'])
        
        batch_entry['hr_token_ids'] = hr_token_ids
        batch_entry['tail_token_ids'] = tail_token_ids
        batch_entry['head_token_ids'] = head_token_ids   
        batch_entry['neg_ids'] = neg_ids

        batch_entry['hr_token_type_ids'] = hr_token_type_ids

        batch_entry['hr_mask'] = hr_mask
        batch_entry['tail_mask'] = tail_mask
        batch_entry['head_mask'] = head_mask
        batch_entry['neg_mask'] = neg_mask

        batch_entry['targets'] = targets
        batch_entry['sources'] = sources

        return batch_entry



class SimKGCDRoBERTaLocalDataset(Dataset):
    def __init__(self, split='train', topk=-1, args=None, tokenizer=None,entity_dataset=None):
        super().__init__()
        fname = args.dataset_dir + '/%s.json' % split
        self.topk = topk
        self.tokenizer = tokenizer
        self.neg_num = args.neg_num
        self.entity_dataset = entity_dataset
        self.data = self.loadData(fname)

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
                entity = cur_data['entity']
                context = cur_data['context']
                neg_sample = cur_data["neg_sample"]

                l_neg = len(neg_sample)
                if l_neg < self.neg_num:
                    random_select = random.sample(self.entity_dataset.id2entity, k=self.neg_num + 2 - l_neg)
                    for e in random_select:
                        if e != entity and e != output:
                            neg_sample.append(e)

                tokenized_s = self.tokenizer(text=input_, text_pair=context,truncation=True, max_length=512)
                source_id = tokenized_s.input_ids
                hr_mask = tokenized_s.attention_mask              

                tokenized_t = self.tokenizer(output, truncation=True, max_length=128)
                target_id = tokenized_t.input_ids
                tail_mask = tokenized_t.attention_mask   

                tokenized_e = self.tokenizer(entity, truncation=True, max_length=128)
                entity_id = tokenized_e.input_ids
                head_mask = tokenized_e.attention_mask   

                out_dict = {
                    'source': {
                        'input':input_, 
                        'context':context, 
                        'forward':cur_data['forward'], 
                        'id': ids, 
                        'year': year, 
                        'rel_sent':rel_sent },
                    'target': [output],
                    'entity': entity,
                    'neg_sample': neg_sample,
                    'input_length': len(source_id),
                    'hr_token_ids': torch.LongTensor(source_id),
                    'tail_token_ids': torch.LongTensor(target_id),
                    'head_token_ids': torch.LongTensor(entity_id),
                    'tail_length': len(target_id),
                    'head_length': len(entity_id),
                    'hr_mask': torch.LongTensor(hr_mask),
                    'tail_mask': torch.LongTensor(tail_mask),
                    'head_mask': torch.LongTensor(head_mask),
                }
                data.append(out_dict)
                if len(data) > self.topk and self.topk != -1:
                    return data
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datum = self.data[idx]
        negs = random.sample(datum['neg_sample'], k= self.neg_num)
        neg_length = []
        neg_ids = []
        neg_masks = []

        for neg in negs:
            tokenized_n = self.tokenizer(neg, truncation=True, max_length=128)
            neg_id = tokenized_n.input_ids
            neg_mask = tokenized_n.attention_mask
            neg_length.append(len(neg_id))
            neg_ids.append(torch.LongTensor(neg_id))
            neg_masks.append(torch.LongTensor(neg_mask))
        
        datum['neg_length'] = neg_length
        datum['neg_ids'] = neg_ids
        datum['neg_mask'] = neg_masks
        return datum

    def collate_fn(self, batch):
        batch_entry = {}
        B = len(batch)

        S_L = max(entry['input_length'] for entry in batch)
        hr_token_ids = torch.ones(B, S_L, dtype=torch.long) * self.tokenizer.pad_token_id
        hr_mask = torch.zeros(B, S_L, dtype=torch.long)

        T_L = max(entry['tail_length'] for entry in batch)
        tail_token_ids = torch.ones(B, T_L, dtype=torch.long) * self.tokenizer.pad_token_id
        tail_mask = torch.zeros(B, T_L, dtype=torch.long) 

        H_L = max(entry['head_length'] for entry in batch)
        head_token_ids = torch.ones(B, H_L, dtype=torch.long) * self.tokenizer.pad_token_id
        head_mask = torch.zeros(B, H_L, dtype=torch.long) 

        N_L = max(neg_length for entry in batch for neg_length in entry['neg_length'])
        neg_ids = torch.ones(B * self.neg_num, N_L, dtype=torch.long) * self.tokenizer.pad_token_id
        neg_mask = torch.zeros(B * self.neg_num, N_L, dtype=torch.long) 

        targets = []
        sources= []
        for i, entry in enumerate(batch):
            hr_token_ids[i, :entry['input_length']] = entry['hr_token_ids']
            tail_token_ids[i, :entry['tail_length']] = entry['tail_token_ids']
            head_token_ids[i, :entry['head_length']] = entry['head_token_ids']

            hr_mask[i, :entry['input_length']] = entry['hr_mask']
            tail_mask[i, :entry['tail_length']] = entry['tail_mask']
            head_mask[i, :entry['head_length']] = entry['head_mask']

            for j in range(self.neg_num):
                index = i  * self.neg_num + j
                neg_ids[index, :entry['neg_length'][j]] = entry['neg_ids'][j]
                neg_mask[index, :entry['neg_length'][j]] = entry['neg_mask'][j]

            sources.append(entry['source'])
            targets.append(entry['target'])
        
        batch_entry['hr_token_ids'] = hr_token_ids
        batch_entry['tail_token_ids'] = tail_token_ids
        batch_entry['head_token_ids'] = head_token_ids        
        batch_entry['neg_ids'] = neg_ids

        batch_entry['hr_mask'] = hr_mask
        batch_entry['tail_mask'] = tail_mask
        batch_entry['head_mask'] = head_mask
        batch_entry['neg_mask'] = neg_mask

        batch_entry['targets'] = targets
        batch_entry['sources'] = sources

        return batch_entry


def get_loader(args, split='train', mode='train', tokenizer=None,
               batch_size=32, workers=4, topk=-1, entity_dataset=None):
    if entity_dataset is None:
        entity_dataset = EntityDict(args.dataset_dir, tokenizer)
        
    if 'roberta' in args.model:
        dataset =SimKGCDRoBERTaLocalDataset(
                split,
                topk=topk,
                args=args,
                tokenizer=tokenizer,
                entity_dataset=entity_dataset
                )
    else:
        dataset =SimKGCDBERTLocalDataset(
                split,
                topk=topk,
                args=args,
                tokenizer=tokenizer,
                entity_dataset=entity_dataset
                )
    if args.distributed:
        if mode == 'train':
            sampler = DistributedSampler(dataset)
        else:
            sampler = DistributedSampler(dataset, shuffle=False)
    else:
        sampler = None

    if mode == 'test':
        entity_loader = DataLoader(
            entity_dataset,
            batch_size=max(batch_size, 1024),
            num_workers=workers, 
            pin_memory=True,
            sampler=sampler,
            shuffle=False,
            collate_fn=entity_dataset.collate_fn,
            drop_last=False)
    else:
        entity_loader = None
    if mode == 'train':
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=(sampler is None),
            num_workers=workers, 
            pin_memory=True, 
            sampler=sampler,
            collate_fn=dataset.collate_fn,
            # drop_last=True
            )
    else:
        if 'roberta' in args.model:
            loader = DataLoader(
                dataset,
                batch_size=batch_size//2,
                num_workers=workers, 
                pin_memory=True,
                sampler=sampler,
                shuffle=None if (sampler is not None) else False,
                collate_fn=dataset.collate_fn,
                drop_last=False)
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
    return loader, sampler, (entity_dataset, entity_loader)