from sentence_transformers import SentenceTransformer, util
import json
from collections import defaultdict
from tqdm import tqdm
import copy
import os
from transformers import GPT2TokenizerFast
import random
import shutil

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')


context2cands = defaultdict(dict)
relation_template1 = [
    'In that context, which %s can be used for %s, and why?\n%s',
    'In that context, which %s do we use %s, and why?\n%s'
]

context2c = defaultdict(dict)
context2forward = defaultdict(dict)
context2ids = {}
contexts = []

f = open('e2t.json','r')
e2t = json.load(f)

in_name = '../models/T5/local_context_dataset/train.json'
with open(in_name, 'r') as file_j:
    for idxn, line in tqdm(enumerate(file_j), "Encoding"):
        pdf_dict = json.loads(line)
        input_t, cc = pdf_dict['input'].split('| context: ')

        type_ = input_t.split()[-1]

        context = input_t + ' context: ' + cc
        context2ids[context] = pdf_dict['id']

        context2cands[context][pdf_dict['output']] = type_


        entity = pdf_dict['entity']
        output = pdf_dict['output']
        rel_sent = pdf_dict['rel_sent']
        type_e = e2t[entity].lower()
        type_o = e2t[output].lower()

        prompt = []
        cc1 = 'Consider the following context: ' + cc
        if pdf_dict['forward']:
            prompt.append(cc1)
            prompt.append(relation_template1[0] % (type_o, entity, rel_sent))
            prompt = '\n'.join(prompt)
        else:
            prompt.append(cc1)
            prompt.append(relation_template1[1] % (type_o, entity, rel_sent))
            prompt = '\n'.join(prompt)

        context2c[context][pdf_dict['output']] = prompt
        contexts.append(context)

context_embeddings = model.encode(contexts, batch_size=256, convert_to_tensor=True, device='cuda')

def get_retrieve(current_context, contexts, context_embeddings, type_):
    c_embedding = model.encode([current_context], convert_to_tensor=True, device='cuda')
    hits = util.semantic_search(c_embedding, context_embeddings, top_k=100)
    retrieve = []
    all = []
    count = 0
    for i in range(100):
        cur_id = hits[0][i]['corpus_id']
        context1 = contexts[cur_id]
        for output in context2cands[context1]:
            nprompt = context2c[context1][output]
            all.append(nprompt)
            if context2cands[context1][output] == type_ :

                tmp = retrieve[:]
                tmp.append(nprompt)
                
                txts = ' '.join(tmp)
                if len(tokenizer(txts)['input_ids'])> 2000:
                    continue
                else:
                    count += 1
                    retrieve = tmp
                    if count > 4:
                        return ' '.join(retrieve)

    txts = "\n".join(random.sample(all, k=5))
    while len(tokenizer(txts)['input_ids']) > 2000:
        txts = "\n".join(random.sample(all, k=5))
    return txts


    
os.makedirs('../models/GPT3.5Retr/local_dataset', exist_ok=True)
out_name = '../models/GPT3.5Retr/local_dataset/test.json'
in_name = '../models/T5/local_context_dataset/test.json'
with open(in_name, 'r') as file_j:
    wf = open(out_name, 'w')
    for idxn, line in tqdm(enumerate(file_j), "Encoding"):
        pdf_dict = json.loads(line)
        input_t, cc = pdf_dict['input'].split('| context: ')

        type_ = input_t.split()[-1]

        tmp = copy.deepcopy(pdf_dict)
        current_context = input_t + ' context: ' + cc
        tmp.pop("neg_sample")
        tmp["input"] = input_t
        tmp["context"] = cc
        tmp['retrieve'] = get_retrieve(current_context, contexts, context_embeddings, type_)
        wf.write(json.dumps(tmp) + '\n')
    wf.close()
src =  '../models/GPT3.5Retr/local_dataset/'
dst = '../models/GPT3.5RND/local_dataset/'
shutil.copytree(src, dst)