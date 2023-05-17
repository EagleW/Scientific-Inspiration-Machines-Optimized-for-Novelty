from sentence_transformers import SentenceTransformer, util
import json
from collections import defaultdict
from tqdm import tqdm
import copy
import os
import shutil



model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

context2cands = defaultdict(dict)
context2c = defaultdict(set)
context2ids = {}
contexts = []

in_name = '../models/Dual_Encoder/local_dataset/train.json'
with open(in_name, 'r') as file_j:
    for idxn, line in tqdm(enumerate(file_j), "Encoding"):
        pdf_dict = json.loads(line)
        type_ = pdf_dict['input'].split()[-1]
        context = pdf_dict['input'] + ' context: ' + pdf_dict['context']
        context2ids[context] = pdf_dict['id']
        context2cands[context][pdf_dict['output']] = type_
        context2c[context].add(pdf_dict['context'])
        contexts.append(context)

context_embeddings = model.encode(contexts, batch_size=256, convert_to_tensor=True, device='cuda')

def get_retrieve(current_context, contexts, context_embeddings, ids, type_, split):
    c_embedding = model.encode([current_context], convert_to_tensor=True, device='cuda')
    hits = util.semantic_search(c_embedding, context_embeddings, top_k=100)
    retrieve = {}
    count = 0
    if split:
        for i in range(100):
            cur_id = hits[0][i]['corpus_id']
            context = contexts[cur_id]
            rid = context2ids[context]
            if rid != ids:
                count += 1
                retrieve.update(context2cands[context])

                if count == 20:
                    break

    else:
        for i in range(20):
            cur_id = hits[0][i]['corpus_id']
            context = contexts[cur_id]
            retrieve.update(context2cands[context])
    
    entities = []
    for entity in retrieve:
        if retrieve[entity] == type_:
            entities.append(entity)

        
    return entities


os.makedirs('../models/Dual_Encoder+CL+NBR/local_sn_dataset', exist_ok=True)
src = '../models/Dual_Encoder/local_dataset/entities.json'
dst = '../models/Dual_Encoder+CL+NBR/local_sn_dataset/entities.json'
shutil.copyfile(src, dst)

out_name = '../models/Dual_Encoder+CL+NBR/local_sn_dataset/%s.json'
in_name = '../models/Dual_Encoder/local_dataset/%s.json'
fnames = ['train', 'valid', 'test']
for fname in fnames:
    input_fname = in_name % fname
    output_fname = out_name % fname
    with open(input_fname, 'r') as file_j:
        wf = open(output_fname, 'w')
        for idxn, line in tqdm(enumerate(file_j), "Encoding"):
            pdf_dict = json.loads(line)
            type_ = pdf_dict['input'].split()[-1]
            tmp = copy.deepcopy(pdf_dict)
            current_context = pdf_dict['input'] + ' context: ' + pdf_dict['context']
            tmp['retrieve'] = get_retrieve(current_context, contexts, context_embeddings, pdf_dict['id'], type_, fname == 'train')
            wf.write(json.dumps(tmp) + '\n')
        wf.close()
