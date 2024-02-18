from sentence_transformers import SentenceTransformer, util
import json
from collections import defaultdict
from tqdm import tqdm
import copy
import os
import csv
from collections import defaultdict

model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

path_i = '../../data/ct/cited_paper_filter.csv'
fieldnames = ["corpus_paper_id","title","venue","year"]
cite_paper = {}
with open(path_i, 'r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    for row in tqdm(csv_reader):
        if line_count == 0:
            pass
        else:
            cite_paper[row["corpus_paper_id"]] = row["title"]
        line_count += 1

path_i = '../../data/ct/citations_filter.csv'
citemap = defaultdict(set)
fieldnames = ["corpus_paper_id","source_id","cited_corpus_paperid"]
with open(path_i, 'r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    rows = []
    for row in tqdm(csv_reader):
        if line_count == 0:
            pass
        else:
            if row["cited_corpus_paperid"] in cite_paper:
                title = cite_paper[row["cited_corpus_paperid"] ]
                citemap[row["source_id"]].add(title)
        line_count += 1

def get_retrieve(current_context, id_):
    c_embedding = model.encode([current_context], convert_to_tensor=True, device='cuda')

    contexts = list(citemap[id_])
    if len(contexts) == 0:
        return []
    context_embeddings = model.encode(contexts, batch_size=256, convert_to_tensor=True, device='cuda')

    hits = util.semantic_search(c_embedding, context_embeddings, top_k=100)
    titles = []
    for i in range(min(5, len(hits[0]))):
        cur_id = hits[0][i]['corpus_id']
        context = contexts[cur_id]
        titles.append(context)

        
    return titles
os.makedirs('../models/T5+CL+NBR/local_ct_dataset', exist_ok=True)
out_name = '../models/T5+CL+NBR/local_ct_dataset/%s.json'
in_name = '../models/T5/local_context_dataset/%s.json'

fnames = ['train', 'valid', 'test']
for fname in fnames:
    input_fname = in_name % fname
    output_fname = out_name % fname
    with open(input_fname, 'r') as file_j:
        wf = open(output_fname, 'w')
        for idxn, line in tqdm(enumerate(file_j), "Encoding"):
            pdf_dict = json.loads(line)
            input_, context_ = pdf_dict['input'].split('| context: ')

            type_ = input_.split()[-1]
            tmp = copy.deepcopy(pdf_dict)
            current_context = input_ + ' context: ' + context_
            tmp['retrieve'] = get_retrieve(current_context, pdf_dict['id'])
            tmp['neg_sample'].extend(tmp['retrieve'] )
            tmp['input']  = input_ + ' | retrieve: ' + ', '.join(tmp["retrieve"]) + ' | context: ' + context_
            wf.write(json.dumps(tmp) + '\n')
        wf.close()
