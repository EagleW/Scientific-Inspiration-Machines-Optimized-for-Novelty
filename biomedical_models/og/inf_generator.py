import numpy as np
import pandas as pd
from typing import Dict, List, Mapping, Union
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig


import datasets
import json

from accelerate import Accelerator, DistributedType
import accelerate
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from collections import defaultdict
import time

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    MistralForCausalLM,
    default_data_collator,
    get_scheduler,
)
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
import warnings
import torch

def main():
    accelerator = Accelerator()
    accelerator.wait_for_everyone()
    accelerator.print("Start loading dataset")
    raw_datasets = load_dataset('json', data_files={"train": 'data/og/test.json'})
    accelerator.print("Finish loading dataset")
    config = AutoConfig.from_pretrained(
        "models/final"
    )
    tokenizer = transformers.LlamaTokenizer.from_pretrained(
        "models/final"
    )
    accelerator.print('Finish loading config')
    accelerator.print("Start loading pretrained")
        
    model = transformers.LlamaForCausalLM.from_pretrained(
            "models/final",
            config=config
    )
    accelerator.print("Finish loading pretrained")
    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        # model.resize_token_embeddings(len(tokenizer))
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
        accelerator.print(f"Embedding size updated to {model.get_input_embeddings().weight.shape[0]}")
    accelerator.wait_for_everyone()
    
    prompt_template =     ("Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        "You are a researcher. You can come up with new hypotheses based on your existing knowledge. "
        "Hypotheses are given against the following input. You should be as detailed as possible.\n\n### Input:\n%s\n\n### Response:")
    
    input_template = "Context: %s \nIn that context, could you formulate a hypothesis, in a single sentence, about %s?\n'"
    
    def tokenize_function(examples):
        txt = input_template % (examples["context"], examples["entity"])
        prompt_ = prompt_template % txt
        p_token = tokenizer(prompt_).input_ids
        return  {
            "input_ids": p_token
        }
    column_names = raw_datasets["train"].column_names
    column_names.append("input_ids")
        
    with accelerator.main_process_first():
        lm_datasets = raw_datasets.map(
            tokenize_function,
            batched=False,
            num_proc=32,
            # remove_columns=column_names,
            # load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    block_size = 420
    tokenizer.pad_token_id = 0
    
    def conv_collator(
        batch
    ) -> Dict:
        IGNORE_TOKEN_ID = -100  # -100 is the default value for ignore_index in CrossEntropyLoss
        def round_to_multiple_of(x: int, y: int) -> int:
            return ((x + y - 1) // y) * y
        
        batch_seq_length = block_size
        batch_seq_length = max(len(x["input_ids"]) for x in batch)
        # batch_seq_length = min(block_size, round_to_multiple_of(batch_seq_length, 16))
        
        # pad data to seq_len, create attention mask
        batch_size = len(batch)
        
        # Need to use pad instead of IGNORE_TOKEN_ID since latter will be sent to model and cause index error
        input_ids = torch.zeros((batch_size, batch_seq_length), dtype=torch.long) #* tokenizer.pad_token_id
        
        other_infos = defaultdict(list)
        
        for batch_idx, tokenized_conv in enumerate(batch):
            
            cur_input_ids = tokenized_conv["input_ids"]
            cur_seq_len = len(cur_input_ids)
            # assert len(cur_loss_masks) == cur_seq_len
            
            # Truncate if necessary
            if cur_seq_len > batch_seq_length:
                cur_input_ids = cur_input_ids[:batch_seq_length]
                cur_loss_masks = cur_loss_masks[:batch_seq_length]
                cur_seq_len = batch_seq_length

            # assert cur_seq_len < batch_seq_length
            input_ids[batch_idx, -cur_seq_len:] = torch.LongTensor(cur_input_ids)
            
            for col_name in column_names:
                other_infos[col_name].append(tokenized_conv[col_name])

        return {
            "input_ids": input_ids,
            "other_infos": other_infos
        }
    valid_dataset = lm_datasets["train"]
    val_dataloader = DataLoader(
        valid_dataset, collate_fn=conv_collator, batch_size=1
    )
    
    # model, val_dataloader = accelerator.prepare(
    #     model, val_dataloader
    # )
    # if accelerator.is_main_process:
    model = model.cuda()
    output_dir = 'models/result.json'
    accelerator.print('start inference')
    
    # wf = open(output_dir, 'a')
    wf = open(output_dir, 'w')
    remain = []
    with torch.no_grad():
        for step, batch in tqdm(enumerate(val_dataloader), total = len(val_dataloader)):
            # if step < 112:
            #     continue
            inputs = batch["input_ids"].cuda()
            other_infos = batch["other_infos"]
            # input_ids = other_infos["input_ids"]
            contexts = other_infos["context"]
            idx_txt = other_infos["idx"]
            targets = other_infos["target"]
            # pmcids = other_infos["pmcid"]
            pmids = other_infos["pmid"]
            entitys = other_infos['entity']
            biotypes = other_infos['biotype']
            outputs = model.generate(inputs = inputs, max_new_tokens=400, do_sample=True, top_k=50)

            generated_sents = tokenizer.batch_decode(outputs)
            # for pmid, pmcid, entity, biotype, input_txt, qid, ans, tgt in zip(pmids, pmcids, entitys, biotypes, contexts, idx_txt, generated_sents, targets):
            for pmid, entity, biotype, input_txt, qid, ans, tgt in zip(pmids, entitys, biotypes, contexts, idx_txt, generated_sents, targets):
                wf.write(json.dumps({
                    # 'input': input_txt,
                    'pmid': pmid,
                    # 'pmcid': pmcid,
                    'context': input_txt,
                    'entity': entity,
                    'biotype': biotype,
                    'idx': qid,
                    'pred': ans,
                    'ground_truth': tgt,
                    
                }) + '\n')
            # print(generated_sents[0])
            # # print(outputs[0]["generated_text"])
            # input()
        wf.close()
        
if __name__ == "__main__":
    main()
