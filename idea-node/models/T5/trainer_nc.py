from transformers import T5Config
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration

from tqdm import tqdm
import torch
import logging
import torch.distributed as dist
from torch.distributed import ReduceOp
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import  get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, AutoTokenizer
import os
from torch.optim import AdamW
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast
from param import parse_args
from dataset_nc import get_loader
from utils_ import LossMeter, set_global_logging_level, reduce_dict, setup_for_distributed, accuracy
from time import time
import json
import wandb
import pickle
import math
from statistics import mean

def Log2(x):
    return (math.log10(x) /
            math.log10(2))


set_global_logging_level(logging.ERROR, ["transformers"])
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_total_params(module: torch.nn.Module):
    total_params = 0
    for param in module.parameters():
        total_params += param.numel()
    return total_params

class Trainer:
    def __init__(self, args, train_loader=None, val_loader=None, test_loader=None, tokenizer=None, sampler=None, train=True):
        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.tokenizer = tokenizer
        self.verbose = True
        if self.args.distributed:
            if dist.get_rank() != 0:
                self.verbose = False  
        self.sampler = sampler    
        if self.verbose:
            from time import time
            start = time()
        
        self.model = T5ForConditionalGeneration.from_pretrained("t5-large")
        if self.verbose:
            print(get_total_params(self.model))
        if args.load is not None:
            ckpt_path = args.load
            self.load_checkpoint(ckpt_path)
            print('Load pretrained model')

        # GPU Options
        if torch.cuda.is_available():
            print(f'Model Launching at GPU {self.args.gpu}')
            self.model = self.model.to(args.gpu)
        if args.distributed:
            self.model = DDP(self.model, device_ids=[args.gpu],
                                output_device=args.gpu
                                )
            self.unwrap_model = self.model.module
        else:
            self.unwrap_model = self.model
        # Optimizer
        if train:
            if self.args.fp16:
                print('Run in half precision')
                self.scaler = torch.cuda.amp.GradScaler()
            self.create_optimizer_and_scheduler()

        self.device =  next(self.unwrap_model.parameters()).device

        if self.verbose:
            print(f'It took {time() - start:.1f}s')
            if args.wandb:
                wandb.watch(self.model)
    def create_optimizer_and_scheduler(self):
        if self.verbose:
            print('Building Optimizer')

        self.optim = AdamW([p for p in self.model.parameters() if p.requires_grad],
                            lr=self.args.lr, eps=self.args.adam_eps, betas=(0.9, 0.98))
        num_training_steps = self.args.epochs * len(self.train_loader)
        self.lr_scheduler = self._create_lr_scheduler(num_training_steps)
        

    def _create_lr_scheduler(self, num_training_steps):
        self.args.warmup = min(self.args.warmup, num_training_steps // 10)
        if self.args.lr_scheduler == 'linear':
            return get_linear_schedule_with_warmup(optimizer=self.optim,
                                                   num_warmup_steps=self.args.warmup,
                                                   num_training_steps=num_training_steps)
        elif self.args.lr_scheduler == 'cosine':
            return get_cosine_schedule_with_warmup(optimizer=self.optimizer,
                                                   num_warmup_steps=self.args.warmup,
                                                   num_training_steps=num_training_steps)
        else:
            assert False, 'Unknown lr scheduler: {}'.format(self.args.scheduler)

        

    def load_checkpoint(self, ckpt_path):
        if self.verbose:
            print("Load model from %s" % ckpt_path)
        pretrained_dict = torch.load("%s.pth" % ckpt_path)

        model_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict, strict=False)

    def save(self, name):
        if not os.path.isdir(self.args.output):
            os.makedirs(self.args.output, exist_ok=True)
        torch.save(self.unwrap_model.state_dict(),
            os.path.join(self.args.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.unwrap_model.load_state_dict(state_dict) 

    def train(self):
        if self.verbose:
            loss_meter = LossMeter()
            best_valid = 0.
            best_epoch = 0

        global_step = 0
        update_epoch = 0
        if self.args.distributed:
            dist.barrier()

        for epoch in range(self.args.epochs):
            flag_tensor = torch.zeros(1).to(self.device)
            if self.args.distributed:
                self.sampler.set_epoch(epoch)

            self.model.train()
            if self.verbose:
                pbar = tqdm(total=len(self.train_loader), ncols=150)

            epoch_results = {
                'loss': 0.
            }

            for batch in self.train_loader:
                self.model.train()
                self.model.zero_grad(set_to_none=True)

                if self.args.fp16:
                    with autocast():
                        input_ids = batch['input_ids'].to(self.device)
                        lm_labels = batch["target_ids"].to(self.device)
                        attention_mask = batch["attention_masks"].to(self.device)

                        results = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=lm_labels,
                            return_dict=True
                        )
                        loss = results.loss
                        self.scaler.scale(loss).backward()
                else:
                    input_ids = batch['input_ids'].to(self.device)
                    lm_labels = batch["target_ids"].to(self.device)
                    attention_mask = batch["attention_masks"].to(self.device)

                    results = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=lm_labels,
                        return_dict=True
                    )
                    loss = results.loss
                    loss.backward()

                # Update Parameters
                if self.args.clip_grad_norm > 0:
                    if self.args.fp16:

                        self.scaler.unscale_(self.optim)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)

                        self.scaler.step(self.optim)

                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.clip_grad_norm)
                        self.optim.step()
                else:

                    if self.args.fp16:
                        self.scaler.step(self.optim)
                        self.scaler.update()
                    else:
                        self.optim.step()

                if self.lr_scheduler:
                    self.lr_scheduler.step()
                for param in self.model.parameters():
                    param.grad = None

                global_step += 1
                l = results.loss.detach().item()
                epoch_results['loss'] += l
                lr=self.optim.param_groups[0]["lr"] 

                if self.verbose:
                    loss_meter.update(l)
                    desc_str = f'Epoch {epoch} | LR {lr:.10f}'
                    desc_str += f' | Loss {loss_meter.val:6f}'

                    pbar.set_description(desc_str)
                    pbar.update(1)

            if self.verbose:
                pbar.close()
            len_train_loader = len(self.train_loader)
            epoch_results['loss'] /= len_train_loader
            if self.args.distributed:
                dist.barrier()
                epoch_results = reduce_dict(epoch_results, average=True)
            
            # Validation
            score_dict = self.evaluate(self.val_loader)
            if self.args.distributed:
                dist.barrier()
                score_dict = reduce_dict(score_dict, average=True)


            if self.verbose:
                valid_score = score_dict['acc10']
                if valid_score > best_valid or epoch == 0:
                    best_valid = valid_score
                    best_epoch = epoch
                    self.save("BEST")
                    update_epoch  = epoch

                log_str = ''
                log_str += "\nEpoch %d: Best Acc@10 Score %0.2f\n" % (best_epoch, best_valid)

                wandb_log_dict = {}
                wandb_log_dict['Train/Loss'] = epoch_results['loss'] 

                wandb_log_dict['Valid/Loss'] = score_dict['loss']
                wandb_log_dict['Valid/Acc@1'] = score_dict['acc1']
                wandb_log_dict['Valid/Acc@10'] = score_dict['acc10']

                if self.args.wandb:
                    wandb.log(wandb_log_dict, step=epoch)
                print("\nEpoch %d: Valid Acc@1 %0.4f Valid Acc@10 %0.4f Valid loss %0.4f Train loss %0.4f \n" % (epoch, wandb_log_dict['Valid/Acc@1'], wandb_log_dict['Valid/Acc@10'], wandb_log_dict['Valid/Loss'], wandb_log_dict['Train/Loss']))
                print(log_str)
                print()
            if self.args.distributed:
                dist.barrier()
        
                if self.verbose:
                    if epoch - update_epoch > self.args.patient:
                        flag_tensor += 1
                dist.all_reduce(flag_tensor,op=ReduceOp.SUM)
                if flag_tensor > 0:
                    break
            else:
                if epoch - update_epoch > self.args.patient:
                    break        
            if self.args.distributed:
                dist.barrier()

        if self.args.distributed:
            dist.barrier()
        if self.verbose:
            self.save("LAST")
        # Test Set
        if self.args.distributed:
            dist.barrier()
        torch.cuda.empty_cache()
        best_path = os.path.join(self.args.output, 'BEST')
        self.load(best_path)
        
        score_dict = self.evaluate(self.test_loader)
        if self.args.distributed:
            dist.barrier()
            score_dict = reduce_dict(score_dict, average=True)

        if self.verbose:

            wandb_log_dict = {}
            wandb_log_dict['Test/Loss'] = score_dict['loss']
            wandb_log_dict['Test/Acc@1'] = score_dict['acc1']
            wandb_log_dict['Test/Acc@10'] = score_dict['acc10']

            print(wandb_log_dict)
            if self.args.wandb:
                wandb.log(wandb_log_dict)
                wandb.log({'finished': True})

                print('save prediction file')

        if self.args.distributed:
            dist.barrier()
            exit()

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        epoch_results = {
            'loss': 0.,
            'acc1': 0.,
            'acc10': 0
        }
        if self.verbose:
            pbar = tqdm(total=len(loader), ncols=120, desc="Prediction")

        for batch in loader:
            input_ids = batch['input_ids'].to(self.device)

            output = self.unwrap_model.generate(input_ids = input_ids, num_beams=10, num_return_sequences=10, output_scores=True, return_dict_in_generate=True)

            prediction = self.tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
            references = batch["targets"]
            predictions = []

            batch_size = input_ids.size(0)
            for i in range(batch_size):
                predictions.append(prediction[i*10:(i+1)*10])
            for pred, ref in zip(predictions, references):
                for idx, p in enumerate(pred):
                    if p in ref:
                        if idx < 1:
                            epoch_results['acc1'] += 1
                        if idx < 10:
                            epoch_results['acc10'] += 1


            if self.verbose:
                pbar.update(1)


        len_loader = len(loader)
        epoch_results['loss'] /= len_loader
        epoch_results['acc1'] /= len_loader
        epoch_results['acc10'] /= len_loader
        if self.verbose:
            pbar.close()
        return epoch_results

    @torch.no_grad()
    def predict(self):
        start_time = time()
        self.model.eval()

        forward_ = []
        backward_ = []
        mrrf, hit1f, hit3f, hit5f, hit10f = 0, 0, 0, 0, 0
        mrrb, hit1b, hit3b, hit5b, hit10b = 0, 0, 0, 0, 0
        count_f = 0
        count_b = 0

        for batch in tqdm(self.test_loader):
            input_ids = batch['input_ids'].to(self.device)
            batch_size = input_ids.size(0)
            output = self.unwrap_model.generate(
                input_ids = input_ids, 
                num_beams=self.args.beam_size, 
                num_return_sequences=self.args.num_predictions, 
                num_beam_groups=self.args.num_predictions, 
                diversity_penalty=15.0,  
                output_scores=True, 
                return_dict_in_generate=True)

            prediction = self.tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
            score = output.sequences_scores.cpu().tolist()
            predictions = []
            scores = []
            for i in range(batch_size):
                predictions.append(prediction[i*10:(i+1)*10])
                scores.append(score[i*10:(i+1)*10])
            references =  batch["targets"]
            sources =  batch["sources"]

            for pred, ref, source, score in zip(predictions, references, sources, scores):
                tcount, tmrr, thit1, thit3, thit5, thit10 = 0, 0, 0, 0, 0, 0
                topk_score_info = {}
                for idx, (p, s) in enumerate(zip(pred, score)):
                    topk_score_info[p] = s
                    if p in ref:
                        tmrr += 1/(idx + 1)
                        thit1 += 1 if idx + 1 <= 1 else 0
                        thit3 += 1 if idx + 1 <= 3 else 0
                        thit5 += 1 if idx + 1 <= 5 else 0
                        thit10 += 1 if idx + 1 <= 10 else 0
                tcount += 1
                tmp = {
                    'input': source['input'],
                    'pred': pred[0],
                    'ground_truth': [],
                    'topk_score_info': json.dumps(topk_score_info)
                }
                if source['forward']:
                    mrrf += tmrr
                    count_f += tcount
                        
                    hit1f += thit1
                    hit3f += thit3
                    hit5f += thit5
                    hit10f += thit10
                    for r in ref:
                        tttmp = {
                            'tail': r,
                            'sentence': json.dumps([source['id'], source['year'], source['rel_sent']])
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
                    for r in ref:
                        tttmp = {
                            'head': r,
                            'sentence': json.dumps([source['id'], source['year'], source['rel_sent']])
                        }
                        tmp['ground_truth'].append(tttmp)
                    backward_.append(tmp)
                tmp['hit1'] = thit1
                tmp['hit5'] = thit5
                tmp['hit10'] = thit10

        os.makedirs(self.args.output, exist_ok=True)

        with open('{}/eval_forward.json'.format(self.args.output), 'w', encoding='utf-8') as writer:
            writer.write(json.dumps(forward_, ensure_ascii=False, indent=4))

        with open('{}/eval_backward.json'.format(self.args.output), 'w', encoding='utf-8') as writer:
            writer.write(json.dumps(backward_, ensure_ascii=False, indent=4))
        
        forward_metrics = {'mrr': mrrf, 'hit@1': hit1f, 'hit@3': hit3f, 'hit@5': hit5f, 'hit@10': hit10f}
        forward_metrics = {k: round(v / count_f, 4) for k, v in forward_metrics.items()}


        backward_metrics = {'mrr': mrrb, 'hit@1': hit1b, 'hit@3': hit3b, 'hit@5': hit5b, 'hit@10': hit10b}
        backward_metrics = {k: round(v / count_b, 4) for k, v in backward_metrics.items()}

        metrics = {k: round((forward_metrics[k] * count_f + backward_metrics[k] * count_b) / (count_b + count_f), 4) for k in forward_metrics}

        with open('{}/metrics.json'.format(self.args.output), 'w', encoding='utf-8') as writer:
            writer.write('forward metrics: {}\n'.format(json.dumps(forward_metrics)))
            writer.write('backward metrics: {}\n'.format(json.dumps(backward_metrics)))
            writer.write('average metrics: {}\n'.format(json.dumps(metrics)))

        print('Evaluation takes {} seconds'.format(round(time() - start_time, 3)))




def main_worker(gpu, args):
    # GPU is assigned
    args.gpu = gpu
    args.rank = gpu
    print(f'Process Launching at GPU {gpu}')
    if args.distributed:
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl',
                                     init_method='env://', rank=args.rank, world_size=args.world_size)
    tokenizer =  AutoTokenizer.from_pretrained(args.model)

    if args.test_only:
        print(f'Building submit test loader at GPU {gpu}')

        split = f'submit_{gpu}'
        print('Loading', split)

        test_loader, sampler, entity_dataset = get_loader(
            args,
            split='test', 
            mode='test', 
            tokenizer=tokenizer,
            batch_size=args.valid_batch_size,
            workers=args.workers,
            topk=args.valid_topk,
        )
        train_loader = None
        val_loader = None

        trainer = Trainer(args, train_loader, val_loader, test_loader, tokenizer, sampler, train=False)
        trainer.predict()

    else:

        print(f'Building train loader at GPU {gpu}')

        train_loader, sampler, entity_dataset = get_loader(
            args,
            split='train', 
            mode='train', 
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            workers=args.workers,
            topk=args.train_topk,
        )

        if args.valid_batch_size is not None:
            valid_batch_size = args.valid_batch_size
        else:
            valid_batch_size = args.batch_size
        print(f'Building val loader at GPU {gpu}')
        val_loader, sampler, _ = get_loader(
            args,
            split='valid', 
            mode='val', 
            tokenizer=tokenizer,
            batch_size=valid_batch_size,
            workers=args.workers,
            topk=args.valid_topk,
            entity_dataset=entity_dataset
        )

        print(f'Building test loader at GPU {gpu}')
        test_loader, sampler, _ = get_loader(
            args,
            split='test', 
            mode='val', 
            tokenizer=tokenizer,
            batch_size=valid_batch_size,
            workers=args.workers,
            topk=args.valid_topk,
            entity_dataset=entity_dataset
        )

        trainer = Trainer(args, train_loader, val_loader, test_loader, tokenizer, sampler, train=True)

        trainer.train()


if __name__ == "__main__":
    args = parse_args()
    if torch.cuda.is_available() and args.distributed:
        ngpus_per_node = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
        cudnn.benchmark = True
        args.distributed = args.distributed and ngpus_per_node>1
        args.world_size =  ngpus_per_node
        args.rank = int(os.environ["RANK"])
    else:
        args.world_size = 0
        args.local_rank = -1
        args.distributed = False
    project_name = "kgt5_contrastive"

    if args.local_rank in [0, -1]:
        comments = []
        if args.load is not None:
            ckpt_str = "_".join(args.load.split('/')[-3:])
            comments.append(ckpt_str)
        comment = '_'.join(comments)

        from datetime import datetime
        current_time = datetime.now().strftime('%b%d_%H-%M')

        run_name = f'{current_time}_GPU{args.world_size}'
        if len(comments) > 0:
            run_name += f'_{comment}'

        args.run_name = run_name

        if args.wandb:
            wandb.init(project=project_name,  resume="allow")
            wandb.config.update(args)
            config = wandb.config
        else:
            config=args

    if args.distributed:
        main_worker(args.local_rank, args)
    else:
        main_worker(0, args)
