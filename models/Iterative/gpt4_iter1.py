import os
from tqdm import tqdm
import openai
import json
from time import time
import random
import glob

from pprint import pprint
from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
openai.organization = "" # type your own organization
openai.api_key = "" # type your own api key


os.makedirs('1st', exist_ok=True)

f_template = 'eval_%s.json'
path = 'sim_dataset/'

system_header = "You are an expert in natural language processing whose task is to come up with new hypotheses based on your existing knowledge.\n\n"

query2_template = "Your prediction has similarities with existing research as demonstrated by these %d sentences:\n%s\nMake sure the idea you suggest is significantly different from the existing research mentioned in the above sentences. Let's give it another try.\n"
unfinished = 0

foldernames = [
    'GPT4FS_checkpoint/',
    'GPT4FS+sn_checkpoint/', 
    'GPT4FS+kg_checkpoint/', 
    'GPT4FS+ct_checkpoint/'
    ]

for foldername in foldernames:
    os.makedirs('1st' + foldername, exist_ok=True)
    for fname in ['forward', 'backward']: 
        print(fname)
        filename = path + foldername + f_template % fname
        predictions= json.load(open(filename, 'r'))
        ffname = '1st' + foldername  + f_template % fname
        start = time()
        count = 0
        cc = 0
        new_pp = []
        for data in tqdm(predictions):
            cc += 1
            if time() - start > 59:
                if count % 200 == 199:
                    time.sleep(60)
                count = 0
                start = time()
            else:
                count += 1
                
            query1 = data['input']
            answer1 = data['pred']
            similar_score = data['similar_score']
            similar_exp = data['similar']
            cur_cxt_all = data['cur_cxt_all']
            quest = query1.split('\n')[-2]
            threshold = 0.6
            sim_txts = []
            for txt, score in zip(similar_exp, similar_score):
                if score > threshold:
                    sim_txts.append(txt)
                else:
                    break
            number = len(sim_txts)
            similar_txts = '\t\n'.join(sim_txts) #+  cur_cxt_all + '\n' + quest
            
            if number == 0:
                data['regenerate'] = False
                data['final'] = answer1
                writer.write(json.dumps(data) + '\n')
                continue
            
            query2 = query2_template % (number, similar_txts)       
            chat = [
                {"role": "system", "content": system_header},
                {"role": "user", "content": query1},
                {"role": "assistant", "content": answer1},
                {"role": "user", "content": query2},
            ]

                
            response = openai.ChatCompletion.create(
                model="gpt-4-0314", # The deployment name you chose when you deployed the ChatGPT or GPT-4 model.
                messages=chat,
                temperature=1,
                top_p=1,
                n=1,
                max_tokens=100,
                frequency_penalty=0,
                presence_penalty=0,
            )
            txt = response['choices'][0]['message']['content']

            if len(txt) == 0:
                unfinished += 1
                response = openai.ChatCompletion.create(
                    model="gpt-4-0314", # The deployment name you chose when you deployed the ChatGPT or GPT-4 model.
                    messages=chat,
                    temperature=1,
                    top_p=1,
                    n=1,
                    max_tokens=100,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                txt = response['choices'][0]['message']['content']
                
            data['regenerate'] = True
            data['final'] = txt
            new_pp.append(data)
        with open(ffname, 'w', encoding='utf-8') as writer: 
            writer.write(json.dumps(new_pp, ensure_ascii=False, indent=4))
    print(unfinished)