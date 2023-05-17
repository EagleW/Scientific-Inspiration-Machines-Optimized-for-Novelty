from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import json
import torch
import random
from bleu.bleu import Bleu
from rouge.rouge import Rouge
from torch.utils.data.distributed import DistributedSampler


class T5Dataset(Dataset):
    def __init__(self, split='train', topk=-1, args=None, tokenizer=None):
        super().__init__()
        fname = args.dataset_dir + '/%s.json' % split
        self.topk = topk
        self.tokenizer = tokenizer
        self.neg_num = args.neg_num
        self.data = self.loadData(fname)

    def __len__(self):
        return len(self.data)

    def loadData(self, filename):
        data = []
        with open(filename, 'r') as f:
            for line in tqdm(f):
                cur_data = json.loads(line)
                input_ = cur_data["input"]
                output = cur_data["rel_sent"].lower()
                ids = cur_data["id"]
                year = cur_data["year"]
                rel_sent = cur_data["rel_sent"].lower()
                src_ids = cur_data["src_ids"]


                source_id = self.tokenizer(input_, truncation=True, max_length=512).input_ids

                target_id = self.tokenizer(output, truncation=True, max_length=128).input_ids
                out_dict = {
                    'source': {'input':input_, 'forward':cur_data['forward'], 'id': ids, 'year': year, 'rel_sent':rel_sent },
                    'target': output.lower(),
                    'entity': cur_data['entity'],
                    'src_id': src_ids,
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


        return datum

    def collate_fn(self, batch):
        batch_entry = {}

        B = len(batch)

        S_L = max(entry['input_length'] for entry in batch)
        input_ids = torch.ones(B, S_L, dtype=torch.long) * self.tokenizer.pad_token_id
        attention_masks = torch.zeros(B, S_L, dtype=torch.long)

        T_L = max(entry['target_length'] for entry in batch)
        target_ids = torch.ones(B, T_L, dtype=torch.long) * self.tokenizer.pad_token_id


        targets = []
        sources= []
        src_ids = []

        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']
            target_ids[i, :entry['target_length']] = entry['target_ids']
            attention_masks[i, :entry['input_length']] = 1

            sources.append(entry['source'])
            targets.append(entry['target'])
            src_ids.append(entry['src_id'])

        batch_entry['input_ids'] = input_ids
        word_mask = target_ids != self.tokenizer.pad_token_id
        target_ids[~word_mask] = -100
        batch_entry['target_ids'] = target_ids
        batch_entry['attention_masks'] = attention_masks


        batch_entry['targets'] = targets
        batch_entry['sources'] = sources
        batch_entry['src_ids'] = src_ids


        return batch_entry


def get_loader(args, split='train', mode='train', tokenizer=None,
               batch_size=32, workers=4, topk=-1):

    sampler = None
    dataset = T5Dataset(
                split,
                topk=topk,
                args=args,
                tokenizer=tokenizer)
    
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

    loader.evaluator = Evaluator()

    return loader, sampler


class Evaluator:
    def __init__(self):
        self.scorers = [
            (Bleu(4),  ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L")
            ]

    

    def score(self, ref, hypo):
        final_scores = {}
        for scorer, method in self.scorers:
            score, scores = scorer.compute_score(ref, hypo)
            if type(score) == list:
                for m, s in zip(method, score):
                    final_scores[m] = s
            else:
                final_scores[method] = score

        return final_scores
    def evaluate(self, quesid2ans):
        hypo = {}
        ref = {}
        i = 0
        for k in quesid2ans:
            ans, tgt = quesid2ans[k]
            hypo[i] = [ans]
            ref[i] = [tgt]
            i += 1

        score = self.score(ref, hypo)
        print(score)
        
        return {'score':2*score['ROUGE_L']*score['Bleu_4']/(score['Bleu_4']+ score['ROUGE_L']), 'bleu':score['Bleu_4'], 'rogue':score['ROUGE_L']}