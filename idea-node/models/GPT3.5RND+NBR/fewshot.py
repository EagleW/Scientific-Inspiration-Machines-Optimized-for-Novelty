import os
from tqdm import tqdm
import openai
import json
from  pprint import pprint
from time import time
import random

from pprint import pprint
from transformers import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
openai.organization = "" # type your own organization
openai.api_key = "" # type your own api key


f = open('e2t.json','r')
e2t = json.load(f)

prompts = [
    'Suggest a %s that can be used for a natural language processing %s.',
    'Suggest a %s for a natural language processing %s.']
relation_template1 = [
    'Q: Which %s can be used for %s? A: %s is done by using %s.',
    'Q: Which %s do we use %s?  A: We use %s for %s.'
]
relation_template = [
    'Q: Which %s can be used for %s ? A: %s is done by using ',
    'Q: Which %s do we use %s?  A: We use %s for '
]

dir_path = '../T5+CL+NBR/local_%s_dataset/%s.json'
for fname in ['ct', 'sn', 'kg']:

    start_time = time()
    forward_ = []
    backward_ = []
    mrrf, hit1f, hit3f, hit5f, hit10f = 0, 0, 0, 0, 0
    mrrb, hit1b, hit3b, hit5b, hit10b = 0, 0, 0, 0, 0
    count_f = 0
    count_b = 0
    examples = [[],[]]

    in_name = dir_path % (fname, 'train')

    with open(in_name, 'r') as file_j:
        for idxn, line in tqdm(enumerate(file_j), "Encoding"):
            pdf_dict = json.loads(line)
            input_ = pdf_dict['input'].lower()
            if '| retrieve:' in input_:
                input_t, other = input_.split(' | retrieve: ')
                if ' | context: ' in other:
                    retrieve, context =  other.split(' | context: ')
                else:
                    retrieve, context =  other.split('| context: ')
                cc1 = 'Consider the following context: ' + context + ' The retrieval results are: ' + retrieve
            elif ' | context:' in input_:
                retrieve = ''
                input_t, context = input_.split('| context: ')
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
                prompt.append(relation_template1[0] % (type_o, entity, entity, output))
                prompt = '\n'.join(prompt)
                examples[0].append(prompt)
            else:
                prompt.append(cc1)
                prompt.append(relation_template1[1] % (type_o, entity, entity, output))
                prompt = '\n'.join(prompt)
                examples[1].append(prompt)
    print('Finish processsing samples')
    filename = dir_path % (fname, 'test')
    with open(filename, 'r') as f:
        for line in tqdm(f):
            cur_data = json.loads(line)
            input_ = cur_data['input'].lower()
            if '| retrieve:' in input_:
                input_t, other = input_.split(' | retrieve: ')
                if ' | context: ' in other:
                    retrieve, context =  other.split(' | context: ')
                else:
                    retrieve, context =  other.split('| context: ')
                cc = 'Consider the following context: ' + context + ' The retrieval results are: ' + retrieve
            elif ' | context:' in input_:
                retrieve = ''
                input_t, context = input_.split('| context: ')
                cc = 'Consider the following context: ' + context 

            entity = cur_data['entity']
            output = cur_data['output']

            type_e = e2t[entity].lower()
            type_o = e2t[output].lower()

            if cur_data['forward']:
                pt1 = prompts[0] % (type_o, type_e)
                txts = "\n".join(random.sample(examples[0], k=5))
                while len(tokenizer(txts)['input_ids']) > 1600:
                    txts = "\n".join(random.sample(examples[0], k=5))

                prompt = [pt1, txts,cc]
                prompt.append(relation_template[0] % (type_o, entity, entity))
            else:
                pt2 = prompts[1] % (type_o, type_e)
                txts = " ".join(random.sample(examples[1], k=5))
                while len(tokenizer(txts)['input_ids']) > 1600:
                    txts = " ".join(random.sample(examples[1], k=5))

                prompt = [pt2,txts,cc]
                prompt.append(relation_template[1] % (type_o, entity, entity))
            prompt = '\n'.join(prompt)

            if len(tokenizer(prompt)['input_ids']) > 2048:
                if cur_data['forward']:
                    pt1 = prompts[0] % (type_o, type_e)
                    txts = "\n".join(random.sample(examples[0], k=5))
                    while len(tokenizer(txts)['input_ids']) > 1024:
                        txts = "\n".join(random.sample(examples[0], k=5))

                    prompt = [pt1, txts, cc]
                    prompt.append(relation_template[0] % (type_o, entity, entity))
                else:
                    pt2 = prompts[1] % (type_o, type_e)
                    txts = " ".join(random.sample(examples[1], k=5))
                    while len(tokenizer(txts)['input_ids']) > 1024:
                        txts = " ".join(random.sample(examples[1], k=5))

                    prompt = [pt2, txts, cc]
                    prompt.append(relation_template[1] % (type_o, entity, entity))
                prompt = '\n'.join(prompt)
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt,
                temperature=1,
                top_p=1,
                n=15,
                best_of=15,
                frequency_penalty=0,
                presence_penalty=0,
                stop=[".", '\n']
            )
            choices = response['choices']
            topk_score_info = []
            ttmp = []
            tcount, tmrr, thit1, thit3, thit5, thit10 = 0, 0, 0, 0, 0, 0
            cc = 0
            for tmp in choices:
                txt = tmp['text'].strip()
                idx = tmp['index']
                if len(txt) != 0:
                    ttmp.append((txt, idx))
                    cc += 1
                    if cc > 9:
                        break
            ttmp.sort(key = lambda x: x[1])
            for x,_ in ttmp:
                topk_score_info.append(x)
                if txt == output:
                    tmrr += 1/(idx + 1)
                    thit1 += 1 if idx + 1 <= 1 else 0
                    thit3 += 1 if idx + 1 <= 3 else 0
                    thit5 += 1 if idx + 1 <= 5 else 0
                    thit10 += 1 if idx + 1 <= 10 else 0

            if len(topk_score_info) < 10:
                for _ in range(10-len(ttmp)):
                    topk_score_info.append('')

            tcount += 1
            tmp = {
                'input': prompt,
                'pred': topk_score_info[0],
                'ground_truth': [],
                'topk_score_info': json.dumps(topk_score_info)
            }
            if cur_data['forward']:
                mrrf += tmrr
                count_f += tcount
                    
                hit1f += thit1
                hit3f += thit3
                hit5f += thit5
                hit10f += thit10
                tttmp = {
                    'tail': output,
                    'sentence': json.dumps([cur_data['id'], cur_data['year'], cur_data['rel_sent']])
                }
                tmp['ground_truth'].append(tttmp)
                forward_.append(tmp)
            else:
                mrrb += tmrr
                count_b += tcount

                hit1b += thit1
                hit3b += thit3
                hit5b += thit5
                hit10b += thit10

                tttmp = {
                    'head': output,
                    'sentence': json.dumps([cur_data['id'], cur_data['year'], cur_data['rel_sent']])
                }
                tmp['ground_truth'].append(tttmp)
                backward_.append(tmp)
            tmp['hit1'] = thit1
            tmp['hit5'] = thit5
            tmp['hit10'] = thit10

    output_ = 'GPT3.5Rnd+%s_checkpoint' % fname
    os.makedirs(output_, exist_ok=True)
    with open('{}/eval_forward.json'.format(output_), 'w', encoding='utf-8') as writer:
        writer.write(json.dumps(forward_, ensure_ascii=False, indent=4))

    with open('{}/eval_backward.json'.format(output_), 'w', encoding='utf-8') as writer:
        writer.write(json.dumps(backward_, ensure_ascii=False, indent=4))

    forward_metrics = {'mrr': mrrf, 'hit@1': hit1f, 'hit@3': hit3f, 'hit@5': hit5f, 'hit@10': hit10f}
    forward_metrics = {k: round(v / count_f, 4) for k, v in forward_metrics.items()}


    backward_metrics = {'mrr': mrrb, 'hit@1': hit1b, 'hit@3': hit3b, 'hit@5': hit5b, 'hit@10': hit10b}
    backward_metrics = {k: round(v / count_b, 4) for k, v in backward_metrics.items()}

    metrics = {k: round((forward_metrics[k] * count_f + backward_metrics[k] * count_b) / (count_b + count_f), 4) for k in forward_metrics}

    with open('{}/metrics.json'.format(output_), 'w', encoding='utf-8') as writer:
        writer.write('forward metrics: {}\n'.format(json.dumps(forward_metrics)))
        writer.write('backward metrics: {}\n'.format(json.dumps(backward_metrics)))
        writer.write('average metrics: {}\n'.format(json.dumps(metrics)))

    print('Evaluation takes {} seconds'.format(round(time() - start_time, 3)))
