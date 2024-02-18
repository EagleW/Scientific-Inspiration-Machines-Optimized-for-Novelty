from sentence_transformers import SentenceTransformer, util
import json
from collections import defaultdict
from tqdm import tqdm
import copy
import os
import shutil


forward_gpt3 = defaultdict(dict)
backward_gpt3 = defaultdict(dict)

f = open('e2t.json','r')
e2t = json.load(f)

prompt_ = 'Consider the following context: '
relation_template = [
    'In that context, which %s can be used for %s, and why?',
    'In that context, which %s do we use %s, and why?'
]

model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

in_name = '../T5/local_context_dataset/train.json'
relmap = {}
with open(in_name, 'r') as file_j:
    for idxn, line in tqdm(enumerate(file_j), "Encoding"):
        pdf_dict = json.loads(line)
        rel_sent = pdf_dict['rel_sent'].lower() 
        relmap[rel_sent] = pdf_dict['rel_sent']
        
contexts = []
for key in relmap:
    contexts.append(key)
context_embeddings = model.encode(contexts, batch_size=256, convert_to_tensor=True, device='cuda')

map_gt = {
    'forward': 'tail',
    'backward': 'head'
}

os.makedirs('sim_dataset1', exist_ok=True)

foldernames = [
    'GPT4FS_checkpoint/',
    'GPT4FS+sn_checkpoint/', 
    'GPT4FS+kg_checkpoint/', 
    'GPT4FS+ct_checkpoint/'
    ]
def get_retrieve(current_context, contexts, context_embeddings):
    c_embedding = model.encode([current_context], convert_to_tensor=True, device='cuda')
    hits = util.semantic_search(c_embedding, context_embeddings, top_k=20)
    sents = []
    scores = []
    for i in range(20):
        hh = hits[0][i]
        cur_id = hh['corpus_id']
        score = hh['score']
        context = relmap[contexts[cur_id]]
        sents.append(context)
        scores.append(score)
    return sents, scores
        
    
f_template = 'eval_%s.json'

for foldername in foldernames:
    similarity = []
    print(foldername)
    os.makedirs('sim_dataset/' + foldername, exist_ok=True)
    for fname in ['forward', 'backward']: 
        print(fname)
        input_fname = '1st' + foldername + f_template % fname
        output_fname = 'sim_dataset/' + foldername + f_template % fname
        predictions= json.load(open(input_fname, 'r'))
        new_prediction = []
        for tmp in tqdm(predictions):
            input_ = tmp['input']
            input_split = input_.split('\n')
            ppt_ = input_split[-2]
            cur_cxt_all = input_split[-3]
            if ' The retrieval results are: ' in cur_cxt_all:
                cur_cxt, retrival = cur_cxt_all.split(' The retrieval results are: ')
            else:
                cur_cxt = cur_cxt_all
            ground_truth = tmp['ground_truth'][0][map_gt[fname]].lower()
            p1 = ppt_.split('In that context, which ')[1]
            
            if ' do we use ' in p1:
                type_, p2 = p1.split(' do we use ')
            else:
                type_, p2 = p1.split(' can be used for ')
            entity_ = p2.split(', and why?')[0]
            
            pred = tmp['pred'].lower()
            sents, scores = get_retrieve(pred, contexts, context_embeddings)
            similarity.extend(scores)
            tmp['type'] = type_
            tmp['entity'] = entity_
            tmp['cur_cxt'] = cur_cxt
            tmp['cur_cxt_all'] = cur_cxt_all
            tmp['similar'] = sents
            tmp['similar_score'] = scores
            new_prediction.append(tmp)
        
        with open(output_fname, 'w', encoding='utf-8') as writer:
            writer.write(json.dumps(new_prediction, ensure_ascii=False, indent=4))
    output_fname = 'sim_dataset/' + foldername + 'simscores.json'
        
    with open(output_fname, 'w', encoding='utf-8') as writer:
        writer.write(json.dumps(similarity, ensure_ascii=False, indent=4))
