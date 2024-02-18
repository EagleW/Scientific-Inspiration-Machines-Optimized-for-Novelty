import json
from collections import defaultdict
from tqdm import tqdm
import copy
import os
import shutil
from collections import defaultdict, Counter



def Sorting(lst):
    lst2 = sorted(lst, key=len)
    return lst2


input_tmp = '../../data/kg/kg_%d.json'

entities = set()
entity2type_all = defaultdict(list)
entity2type = {}

for ii in [2020,2021,2022]: 
    with open(input_tmp % ii, 'r') as file_j:
        for idxn, line in tqdm(enumerate(file_j), "Encoding", position=0):
            pdf_dict = json.loads(line)
            ners = pdf_dict["ner"]
            mention2cluster = pdf_dict["mention2cluster"]
            cluster2txt = pdf_dict["cluster2txt"]
            for ner in ners:
                for tmp in ner:
                    if len(tmp) > 0:
                        (s, e, t, p, _, offset) = tmp 
                        pos = str(s) + '-' + str(e)                        
                        cls_idx = mention2cluster[pos]
                        token = cluster2txt[cls_idx].lower()
                        entity2type_all[token].append(t)

print('Finish read entities')
for entity in entity2type_all:
    tags = entity2type_all[entity]
    c = Counter(tags)
    entity2type[entity] = c.most_common(1)[0][0]

f = open('../../data/kg/generic.txt','r')
generic_vocab = json.load(f)


relations = defaultdict(dict)
reversed_relations = defaultdict(dict)
input_tmp = '../../data/kg/kg_2020.json'
with open(input_tmp, 'r') as  file_j:
    for idxn, line in tqdm(enumerate(file_j), "Encoding", position=0):
        pdf_dict = json.loads(line)
        mention2cluster = pdf_dict["mention2cluster"]
        cluster2txt = pdf_dict["cluster2txt"]
        relations_t = pdf_dict['relations']
        for rel in relations_t:
            for tmp in rel:
                if len(tmp) > 0:
                    (s1, e1, s2, e2, t, p) = tmp
                    pos1 = str(s1) + '-' + str(e1)
                    pos2 = str(s2) + '-' + str(e2)
                    cls_idx1 = mention2cluster[pos1]
                    cls_idx2 = mention2cluster[pos2]
                    token1 = cluster2txt[cls_idx1].lower()
                    token2 = cluster2txt[cls_idx2].lower()
                    if  token1.lower() == '' or token2.lower() == '':
                        continue
                    if token2 in relations[token1]:
                        relations[token1][token2].add(t)
                    else:
                        relations[token1][token2] = set()
                        relations[token1][token2].add(t)

                    if token1 in reversed_relations[token2]:
                        reversed_relations[token2][token1].add(t)
                    else:
                        reversed_relations[token2][token1] = set()
                        reversed_relations[token2][token1].add(t)


os.makedirs('../models/T5+CL+NBR/local_kg_dataset', exist_ok=True)
out_name = '../models/T5+CL+NBR/local_kg_dataset/%s.json'
in_name = '../models/T5/local_context_dataset/%s.json'
fnames = ['train', 'valid', 'test']

for fname in fnames:
    input_fname = in_name % fname
    output_fname = out_name % fname
    c = 0
    with open(input_fname, 'r') as file_j:
        wf = open(output_fname, 'w')
        for idxn, line in tqdm(enumerate(file_j), "Encoding"):
            pdf_dict = json.loads(line)
            input_, context_ = pdf_dict['input'].split('| context: ')

            
            entity = pdf_dict['entity']
            type_ = input_.split()[-1]
            entities = []
            retrieve = {}
            if pdf_dict["forward"]:
                if entity in relations:
                    for key in relations[entity]:
                        retrieve[key] = entity2type[key]
                tmp_retrieve = {}
                if len(retrieve) > 20:
                    for key in relations[entity]:
                        if "USED-FOR" in relations[entity][key]:
                            tmp_retrieve[key] = entity2type[key]
                retrieve = tmp_retrieve

            else:
                if entity in reversed_relations:
                    for key in reversed_relations[entity]:
                        retrieve[key] = entity2type[key]
                if len(retrieve) > 20:
                    for key in reversed_relations[entity]:
                        if "USED-FOR" in reversed_relations[entity][key]:
                            tmp_retrieve[key] = entity2type[key]
                retrieve = tmp_retrieve

            for entity in retrieve:
                if retrieve[entity] == type_:
                    entities.append(entity)
            tmp = copy.deepcopy(pdf_dict)
            if len(entities) > 10:

                entities = Sorting(entities)[-10:]

            tmp['retrieve'] = entities
            if len(entities) > 0:
                c +=1
                tmp['input']  = input_ + ' | retrieve: ' + ', '.join(tmp["retrieve"]) + ' | context: ' + context_
            else:                
                tmp['input']  = input_  + ' | context: ' + context_
            wf.write(json.dumps(tmp) + '\n')
        wf.close()
        print(c)
