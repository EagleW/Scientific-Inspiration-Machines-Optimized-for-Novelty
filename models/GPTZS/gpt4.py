import os
from tqdm import tqdm
import openai
import json
from time import time
import random
import glob

from eval import Evaluator
from pprint import pprint
from transformers import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
openai.organization = "" # type your own organization
openai.api_key = "" # type your own api key


foldername = 'GPT4ZS_checkpoint/'

os.makedirs(foldername, exist_ok=True)

f_template = 'eval_%s.json'
path = 'GPT3.5ZS_checkpoint/'
unfinished = 0
for fname in ['forward','backward']: 
    print(fname)
    filename = path + f_template % fname
    predictions= json.load(open(filename, 'r'))
    ffname = foldername  + f_template % fname
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
        prompt = data['input']
        data.pop('pred', None)
        response = openai.ChatCompletion.create(
            model="gpt-4-0314", # The deployment name you chose when you deployed the ChatGPT or GPT-4 model.
            messages=[
                {"role": "system", "content": prompt}
            ],
            temperature=1,
            top_p=1,
            n=1,
            max_tokens=100,
            frequency_penalty=0,
            presence_penalty=0,
        )
        txt = response['choices'][0]['message']['content']

        if len(txt) == 0:
            print(prompt)
            unfinished += 1
            response = openai.ChatCompletion.create(
                model="gpt-4-0314", # The deployment name you chose when you deployed the ChatGPT or GPT-4 model.
                messages=[
                    {"role": "system", "content": prompt}
                ],
                temperature=1,
                top_p=1,
                n=1,
                max_tokens=100,
                frequency_penalty=0,
                presence_penalty=0,
            )
            txt = response['choices'][0]['message']['content']
        data['pred'] = txt
        new_pp.append(data)
    with open(ffname, 'w', encoding='utf-8') as writer:
        writer.write(json.dumps(new_pp, ensure_ascii=False, indent=4))
            
print(unfinished)