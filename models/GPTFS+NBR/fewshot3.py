import os
from tqdm import tqdm
import openai
import json
from time import time
import random

from eval import Evaluator
from transformers import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
openai.organization = "" # type your own organization
openai.api_key = "" # type your own api key



evaluator = Evaluator()
f = open('e2t.json','r')
e2t = json.load(f)

relation_template = [
    'In that context, which %s can be used for %s, and why?\n',
    'In that context, which %s do we use %s, and why?\n'
]
relation_template1 = [
    'In that context, which %s can be used for %s, and why?\n%s',
    'In that context, which %s do we use %s, and why?\n%s'
]
dir_path = '../T5+CL+NBR/local_%s_dataset/%s.json'
for fname in ['ct', 'sn', 'kg']:
    start_time = time()
    forward_ = []
    backward_ = []
    quesid2ans_forward = {}
    quesid2ans_backward = {}

    examples = [[],[]]
    in_name = dir_path % (fname, 'train')
    print(in_name)
    with open(in_name, 'r') as file_j:
        for idxn, line in tqdm(enumerate(file_j), "Encoding"):
            pdf_dict = json.loads(line)
            input_ = pdf_dict['input'].lower()
            if '| retrieve:' in input_:
                input_t, other = input_.split(' | retrieve: ')
                retrieve, context =  other.split(' | context: ')
                cc1 = 'Consider the following context: ' + context + ' The retrieval results are: ' + retrieve
            elif ' | context:' in input_:
                retrieve = ''
                input_t, context = pdf_dict['input'].split('| context: ')
                cc1 = 'Consider the following context: ' + context 
            type_ = input_t.split()[-1]


            entity = pdf_dict['entity']
            output = pdf_dict['output']
            rel_sent = pdf_dict['rel_sent']
            type_e = e2t[entity].lower()
            type_o = e2t[output].lower()
            prompt = []

            if pdf_dict['forward']:
                prompt.append(cc1)
                prompt.append(relation_template1[0] % (type_o, entity, rel_sent))
                prompt = '\n'.join(prompt)
                examples[0].append(prompt)
            else:
                prompt.append(cc1)
                prompt.append(relation_template1[1] % (type_o, entity, rel_sent))
                prompt = '\n'.join(prompt)
                examples[1].append(prompt)
    print('Finish processsing samples')
    filename = dir_path % (fname, 'test')
    with open(filename, 'r') as f:
        for line in tqdm(f):
            cur_data = json.loads(line)
            input_ = cur_data['input'].lower()
            entity = cur_data['entity']
            output = cur_data['output'].lower()
            if '| retrieve:' in input_:
                input_t, other = input_.split(' | retrieve: ')
                retrieve, context =  other.split(' | context: ')
                cc = 'Consider the following context: ' + context + ' The retrieval results are: ' + retrieve
            elif ' | context:' in input_:
                retrieve = ''
                input_t, context = input_.split('| context: ')
                cc = 'Consider the following context: ' + context 


            type_e = e2t[entity].lower()
            type_o = e2t[output].lower()

            if cur_data['forward']:
                txts = "\n".join(random.sample(examples[0], k=5))
                while len(tokenizer(txts)['input_ids']) > 1600:
                    txts = "\n".join(random.sample(examples[0], k=5))

                prompt = [txts,cc]
                prompt.append(relation_template[0] % (type_o, entity))
            else:
                txts = " ".join(random.sample(examples[1], k=5))
                while len(tokenizer(txts)['input_ids']) > 1600:
                    txts = " ".join(random.sample(examples[1], k=5))

                prompt = [txts,cc]
                prompt.append(relation_template[1] % (type_o, entity))

            prompt = '\n'.join(prompt)
            if len(tokenizer(prompt)['input_ids']) > 2048:

                if cur_data['forward']:
                    txts = "\n".join(random.sample(examples[0], k=5))
                    while len(tokenizer(txts)['input_ids']) > 1024:
                        txts = "\n".join(random.sample(examples[0], k=5))

                    prompt = [txts,cc]
                    prompt.append(relation_template[0] % (type_o, entity))
                else:
                    txts = " ".join(random.sample(examples[1], k=5))
                    while len(tokenizer(txts)['input_ids']) > 1024:
                        txts = " ".join(random.sample(examples[1], k=5))

                    prompt = [txts,cc]
                    prompt.append(relation_template[1] % (type_o, entity))
                prompt = '\n'.join(prompt)

            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt,
                temperature=1,
                top_p=1,
                n=1,
                max_tokens=100,
                best_of=10,
                frequency_penalty=0,
                presence_penalty=0,
                stop=["."]
            )

            choices = response['choices']
            pred = choices[0]['text'].strip()
            if len(pred) > 0:
                if pred[-1] != '.':
                    pred += '.'
            else:
                pred = ' '
            ref = cur_data['rel_sent']    
            tmp = {
                'input': prompt,
                'pred': pred,
                'ground_truth': []
            }
            qid = cur_data["src_ids"]
            quesid2ans_tmp = {qid: (pred, ref)}
            if cur_data['forward']:
                tttmp = {
                    'tail': ref,
                    'sentence': json.dumps([cur_data['id'], cur_data['year'], cur_data['rel_sent']])
                }
                tmp['ground_truth'].append(tttmp)
                forward_.append(tmp)
                quesid2ans_forward.update(quesid2ans_tmp)
            else:
                tttmp = {
                    'head': ref,
                    'sentence': json.dumps([cur_data['id'], cur_data['year'], cur_data['rel_sent']])
                }
                tmp['ground_truth'].append(tttmp)
                backward_.append(tmp)
                quesid2ans_backward.update(quesid2ans_tmp)
    
    output_ = 'GPT3.5FS+%s_checkpoint' % fname
    os.makedirs(output_, exist_ok=True)

    with open('{}/eval_forward.json'.format(output_), 'w', encoding='utf-8') as writer:
        writer.write(json.dumps(forward_, ensure_ascii=False, indent=4))

    with open('{}/eval_backward.json'.format(output_), 'w', encoding='utf-8') as writer:
        writer.write(json.dumps(backward_, ensure_ascii=False, indent=4))


    forward_metrics = evaluator.evaluate(quesid2ans_forward)
    backward_metrics = evaluator.evaluate(quesid2ans_backward)

    quesid2ans_total = quesid2ans_forward.copy()
    quesid2ans_total.update(quesid2ans_backward)

    metrics =  evaluator.evaluate(quesid2ans_total)


    with open('{}/metrics.json'.format(output_), 'w', encoding='utf-8') as writer:
        writer.write('forward metrics: {}\n'.format(json.dumps(forward_metrics)))
        writer.write('backward metrics: {}\n'.format(json.dumps(backward_metrics)))
        writer.write('average metrics: {}\n'.format(json.dumps(metrics)))
    print('Evaluation takes {} seconds'.format(round(time() - start_time, 3)))