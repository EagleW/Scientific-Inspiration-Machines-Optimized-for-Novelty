import json
from tqdm import tqdm
import os
import shutil
import copy


os.makedirs('../models/T5/local_dataset', exist_ok=True)
src = '../models/Dual_Encoder/local_dataset/entities.json'
dst = '../models/T5/local_dataset/entities.json'
shutil.copyfile(src, dst)

out_name = '../models/T5/local_dataset/%s.json'
in_name = '../models/Dual_Encoder/local_dataset/%s.json'
fnames = ['train', 'valid', 'test']
for fname in fnames:
    input_fname = in_name % fname
    output_fname = out_name % fname
    with open(input_fname, 'r') as file_j:
        wf = open(output_fname, 'w')
        for idxn, line in tqdm(enumerate(file_j), "Encoding"):
            pdf_dict = json.loads(line)
            tmp = copy.deepcopy(pdf_dict)
            tmp['input'] = tmp['input'] + '| context: ' + tmp['context']
            tmp.pop('context')
            wf.write(json.dumps(tmp) + '\n')
        wf.close()

src = '../models/T5/local_dataset/'
dst = '../models/T5+CL/local_dataset/'
shutil.copytree(src, dst)


for fname in ['ct', 'sn', 'kg']:
    fpath = "local_%s_dataset" % fname

    os.makedirs('../models/T5+CL+NBR/' + fpath, exist_ok=True)
    src = '../models/Dual_Encoder/local_dataset/entities.json'
    dst = '../models/T5+CL+NBR/' + fpath + '/entities.json'
    shutil.copyfile(src, dst)

    out_name = '../models/T5+CL+NBR/' + fpath + '/%s.json'
    in_name =  '../models/Dual_Encoder+CL+NBR/' + fpath + '/%s.json'
    fnames = ['train', 'valid', 'test']
    for fname in fnames:
        input_fname = in_name % fname
        output_fname = out_name % fname
        with open(input_fname, 'r') as file_j:
            wf = open(output_fname, 'w')
            for idxn, line in tqdm(enumerate(file_j), "Encoding"):
                pdf_dict = json.loads(line)
                tmp = copy.deepcopy(pdf_dict)
                if len(tmp["retrieve"]) > 0:
                    tmp['input'] = tmp['input'] + ' | retrieve: ' + ', '.join(tmp["retrieve"]) + ' | context: ' + tmp['context']
                else:
                    tmp['input'] = tmp['input']  + ' | context: ' + tmp['context']
                tmp.pop('context')
                wf.write(json.dumps(tmp) + '\n')
            wf.close()