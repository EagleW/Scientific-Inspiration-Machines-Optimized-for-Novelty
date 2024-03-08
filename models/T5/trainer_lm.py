from transformers import T5Config, T5ForConditionalGeneration

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
from dataset_lm import get_loader
from utils_ import LossMeter, set_global_logging_level, reduce_dict, setup_for_distributed, accuracy
from time import time
import json
import wandb


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
      
        self.model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-large")
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
            quesid2ans = self.evaluate(self.val_loader)
            quesid2ans_all = None
            output = [None for _ in range(self.args.world_size)]
            if self.args.distributed:
                dist.barrier()
                dist.gather_object(
                        quesid2ans,
                        output if dist.get_rank() == 0 else None,
                        dst=0
                            )
                
                if self.verbose:
                    quesid2ans_all = {}
                    for quesid2ans_ in output:
                        quesid2ans_all.update(quesid2ans_)
                    # print()
                    # print(len(quesid2ans_all))
            else:
                quesid2ans_all = quesid2ans


            if self.verbose:
                score_dict = self.val_loader.evaluator.evaluate(quesid2ans)
                valid_score = score_dict['score']
                if valid_score > best_valid or epoch == 0:
                    best_valid = valid_score
                    best_epoch = epoch
                    self.save("BEST")
                    update_epoch  = epoch

                log_str = ''
                log_str += "\nEpoch %d: Best Score Score %0.2f\n" % (best_epoch, best_valid)

                wandb_log_dict = {}
                wandb_log_dict['Train/Loss'] = epoch_results['loss'] 

                wandb_log_dict['Valid/Score'] = score_dict['score']
                wandb_log_dict['Valid/Bleu_4'] = score_dict['bleu']
                wandb_log_dict['Valid/ROUGE_L'] = score_dict['rogue']

                if self.args.wandb:
                    wandb.log(wandb_log_dict, step=epoch)
                print("\nEpoch %d: Valid Score %0.4f Valid Bleu_4 %0.4f Valid ROUGE_L %0.4f Train loss %0.4f \n" % (epoch, wandb_log_dict['Valid/Score'], wandb_log_dict['Valid/Bleu_4'], wandb_log_dict['Valid/ROUGE_L'], wandb_log_dict['Train/Loss']))
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
        
        quesid2ans = self.evaluate(self.test_loader)
        quesid2ans_all = None
        output = [None for _ in range(self.args.world_size)]
        if self.args.distributed:
            dist.barrier()
            dist.gather_object(
                    quesid2ans,
                    output if dist.get_rank() == 0 else None,
                    dst=0
                        )
            
            if self.verbose:
                quesid2ans_all = {}
                for quesid2ans_ in output:
                    quesid2ans_all.update(quesid2ans_)
                # print()
                # print(len(quesid2ans_all))
        else:
            quesid2ans_all = quesid2ans

        if self.verbose:
            score_dict = self.val_loader.evaluator.evaluate(quesid2ans)

            wandb_log_dict = {}
            wandb_log_dict['Test/Score'] = score_dict['score']
            wandb_log_dict['Test/Bleu_4'] = score_dict['bleu']
            wandb_log_dict['Test/ROUGE_L'] = score_dict['rogue']

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
        quesid2ans = {}
        if self.verbose:
            pbar = tqdm(total=len(loader), ncols=120, desc="Prediction")

        for batch in loader:
            input_ids = batch['input_ids'].to(self.device)

            output = self.unwrap_model.generate(
                input_ids = input_ids, 
                num_beams=self.args.beam_size,
                repetition_penalty=1.5,)

            predictions = self.tokenizer.batch_decode(output, skip_special_tokens=True)
            ques_ids = batch['src_ids']
            references = batch["targets"]
            for qid, ans, tgt in zip(ques_ids, predictions, references):
                quesid2ans[qid] = (ans, tgt)



            if self.verbose:
                pbar.update(1)


        if self.verbose:
            pbar.close()
            print()
            qid = ques_ids[-1]
            print(qid)
            print('ans: ', quesid2ans[qid][0])
            print('target: ', quesid2ans[qid][1])
        return quesid2ans

    @torch.no_grad()
    def predict(self):
        start_time = time()
        self.model.eval()

        forward_ = []
        backward_ = []
        quesid2ans_forward = {}
        quesid2ans_backward = {}
        evaluator = self.test_loader.evaluator

        for batch in tqdm(self.test_loader):
            input_ids = batch['input_ids'].to(self.device)
            output = self.unwrap_model.generate(
                input_ids = input_ids, 
                max_length=100,
                repetition_penalty=1.5,
                num_beams=self.args.beam_size)

            predictions = self.tokenizer.batch_decode(output, skip_special_tokens=True)
            references =  batch["targets"]
            sources =  batch["sources"]
            ques_ids = batch['src_ids']
            for source, pred, ref, qid in zip(sources, predictions, references, ques_ids):
                tmp = {
                    'input': source['input'],
                    'pred': pred,
                    'ground_truth': []
                }
                quesid2ans_tmp = {qid: (pred, ref)}
                score_tmp = evaluator.evaluate(quesid2ans_tmp)
                tmp.update(score_tmp)
                if source['forward']:
                    tttmp = {
                        'tail': ref,
                        'sentence': json.dumps([source['id'], source['year'], source['rel_sent']])
                    }
                    tmp['ground_truth'].append(tttmp)
                    forward_.append(tmp)
                    quesid2ans_forward.update(quesid2ans_tmp)
                else:
                    tttmp = {
                        'head': ref,
                        'sentence': json.dumps([source['id'], source['year'], source['rel_sent']])
                    }
                    tmp['ground_truth'].append(tttmp)
                    backward_.append(tmp)
                    quesid2ans_backward.update(quesid2ans_tmp)

        os.makedirs(self.args.output, exist_ok=True)

        with open('{}/eval_forward.json'.format(self.args.output), 'w', encoding='utf-8') as writer:
            writer.write(json.dumps(forward_, ensure_ascii=False, indent=4))

        with open('{}/eval_backward.json'.format(self.args.output), 'w', encoding='utf-8') as writer:
            writer.write(json.dumps(backward_, ensure_ascii=False, indent=4))
        
        forward_metrics = evaluator.evaluate(quesid2ans_forward)
        backward_metrics = evaluator.evaluate(quesid2ans_backward)

        quesid2ans_total = quesid2ans_forward.copy()
        quesid2ans_total.update(quesid2ans_backward)

        metrics =  evaluator.evaluate(quesid2ans_total)

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
    tokenizer =  AutoTokenizer.from_pretrained("google-t5/t5-large")

    if args.test_only:
        print(f'Building submit test loader at GPU {gpu}')

        split = f'submit_{gpu}'
        print('Loading', split)

        test_loader, sampler = get_loader(
            args,
            split='test', 
            mode='test', 
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            workers=args.workers,
            topk=args.valid_topk,
        )
        train_loader = None
        val_loader = None

        trainer = Trainer(args, train_loader, val_loader, test_loader, tokenizer, sampler, train=False)
        trainer.predict()

    else:

        print(f'Building train loader at GPU {gpu}')

        train_loader, sampler = get_loader(
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
        val_loader, sampler = get_loader(
            args,
            split='valid', 
            mode='val', 
            tokenizer=tokenizer,
            batch_size=valid_batch_size,
            workers=args.workers,
            topk=args.valid_topk
        )

        print(f'Building test loader at GPU {gpu}')
        test_loader, sampler = get_loader(
            args,
            split='test', 
            mode='val', 
            tokenizer=tokenizer,
            batch_size=valid_batch_size,
            workers=args.workers,
            topk=args.valid_topk
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
    project_name = "kgt5_language_contrastive"

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
