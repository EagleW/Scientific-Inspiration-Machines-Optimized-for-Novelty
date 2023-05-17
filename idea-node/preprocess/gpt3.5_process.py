from sentence_transformers import SentenceTransformer, util
import json
from collections import defaultdict
from tqdm import tqdm
import copy
import os
import shutil
from transformers import GPT2TokenizerFast
import random
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

context2cands = defaultdict(dict)
relation_template1 = [
    'Q: Which %s can be used for %s? A: %s is done by using %s.',
    'Q: Which %s do we use %s?  A: We use %s for %s.'
]

context2c = defaultdict(dict)
context2forward = defaultdict(dict)
context2ids = {}
contexts = []

f = open('e2t.json','r')
e2t = json.load(f)

in_name = '../models/Dual_Encoder/local_dataset/train.json'
with open(in_name, 'r') as file_j:
    for idxn, line in tqdm(enumerate(file_j), "Encoding"):
        pdf_dict = json.loads(line)
        type_ = pdf_dict['input'].split()[-1]
        context = pdf_dict['input'] + ' context: ' + pdf_dict['context']
        context2ids[context] = pdf_dict['id']

        context2cands[context][pdf_dict['output']] = type_


        entity = pdf_dict['entity']
        output = pdf_dict['output']
        type_e = e2t[entity].lower()
        type_o = e2t[output].lower()

        prompt = []
        cc = 'Context: ' + pdf_dict['context']
        if pdf_dict['forward']:
            prompt.append(cc)
            prompt.append(relation_template1[0] % (type_o, entity, entity, output))
            prompt = '\n'.join(prompt)
        else:
            prompt.append(cc)
            prompt.append(relation_template1[1] % (type_o, entity, entity, output))
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
out_name = './models/GPT3.5Retr/local_dataset/test.json'
in_name = '../models/Dual_Encoder/local_dataset/test.json'
with open(in_name, 'r') as file_j:
    wf = open(out_name, 'w')
    for idxn, line in tqdm(enumerate(file_j), "Encoding"):
        pdf_dict = json.loads(line)
        type_ = pdf_dict['input'].split()[-1]
        tmp = copy.deepcopy(pdf_dict)
        current_context = pdf_dict['input'] + ' context: ' + pdf_dict['context']
        tmp['retrieve'] = get_retrieve(current_context, contexts, context_embeddings, type_)
        wf.write(json.dumps(tmp) + '\n')
    wf.close()