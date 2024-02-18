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

prompt_ = 'Consider the following context: '
relation_template = [
    'In that context, which %s can be used for %s, and why?\n',
    'In that context, which %s do we use %s, and why?\n'
]
start_time = time()
forward_ = []
backward_ = []
quesid2ans_forward = {}
quesid2ans_backward = {}
filename = 'local_dataset/test.json'
with open(filename, 'r') as f:
    for line in tqdm(f):
        cur_data = json.loads(line)
        input_ = cur_data['input']
        entity = cur_data['entity']
        output = cur_data['output']
        context = cur_data['context']

        retrieve = cur_data['retrieve']

        type_e = e2t[entity].lower()
        type_o = e2t[output].lower()

        cc = prompt_ + cur_data['context']
        prompt = [retrieve,cc]
        if cur_data['forward']:
            prompt.append(relation_template[0] % (type_o, entity))
        else:
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
            pred = ''

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


output_ = 'GPT3.5Retr_checkpoint'
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