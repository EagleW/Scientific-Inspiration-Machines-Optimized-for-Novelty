import torch.backends.cudnn as cudnn
import os

from tqdm import tqdm
import torch
import logging
import torch.distributed as dist
from torch.distributed import ReduceOp
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
from torch.optim import AdamW
from model import ContrastiveDualEncoder
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.cuda.amp import autocast
from param import parse_args
from dataset import get_loader
from utils_ import LossMeter, set_global_logging_level, reduce_dict, setup_for_distributed, accuracy
from time import time
import json
import math
from statistics import mean

def Log2(x):
    return (math.log10(x) /
            math.log10(2))

set_global_logging_level(logging.ERROR, ["transformers"])
os.environ["TOKENIZERS_PARALLELISM"] = "true"

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

        if 'roberta' in self.args.model.lower():
            self.keys = ['hr_token_ids', 'hr_mask', 'tail_token_ids', 'tail_mask', 'head_token_ids', 'head_mask', 'neg_ids', 'neg_mask']
        else:
            self.keys = ['hr_token_ids', 'hr_mask', 'tail_token_ids', 'tail_mask', 'head_token_ids', 'head_mask', 'hr_token_type_ids', 'neg_ids', 'neg_mask']

        self.model = ContrastiveDualEncoder(args)
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
                
    def train(self):
        if self.verbose:
            LOSSES_NAME = ['loss', 'acc1', 'acc2']
            loss_meters = [LossMeter() for _ in range(3)]

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
                'loss': 0.,
                'acc1': 0.,
                'acc10': 0
            }

            for batch in self.train_loader:
                self.model.train()
                self.model.zero_grad(set_to_none=True)


                if self.args.fp16:
                    with autocast():
                        batch = self.to_cuda(batch)
                        results = self.model(**batch)
                        loss = results.loss
                        self.scaler.scale(loss).backward()
                else:
                    batch = self.to_cuda(batch)
                    results = self.model(**batch)

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
                acc1 = results.acc[0].detach().item()
                acc10 = results.acc[1].detach().item()

                epoch_results['loss'] += l
                epoch_results['acc1'] += acc1
                epoch_results['acc10'] += acc10

                lr=self.optim.param_groups[0]["lr"] 

                if self.verbose:
                    loss_meters[0].update(l)
                    desc_str = f'Epoch {epoch} | LR {lr:.10f}'
                    desc_str += f' | Loss {loss_meters[0].val:6f}'

                    loss_meters[1].update(acc1)
                    desc_str += f' | Acc@1 {loss_meters[1].val:6f}'

                    loss_meters[2].update(acc10)
                    desc_str += f' | Acc@10 {loss_meters[2].val:6f}'

                    pbar.set_description(desc_str)
                    pbar.update(1)

            
            if self.verbose:
                pbar.close()

            len_train_loader = len(self.train_loader)
            epoch_results['loss'] /= len_train_loader
            epoch_results['acc1'] /= len_train_loader
            epoch_results['acc10'] /= len_train_loader
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
                wandb_log_dict['Train/Acc@1'] = epoch_results['acc1'] 
                wandb_log_dict['Train/Acc@10'] = epoch_results['acc10'] 

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
            batch = self.to_cuda(batch)
            results = self.model(**batch)
            epoch_results['loss'] += results.loss.detach().item()
            epoch_results['acc1'] += results.acc[0].detach().item()
            epoch_results['acc10'] += results.acc[1].detach().item()
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
    def predict(self, entity_dataset, entity_loader):
        k = 10
        start_time = time()
        prob_total = []
        self.model.eval()
        ent_tensor_list = []
        for batch in tqdm(entity_loader):
            outputs = self.model(
                hr_token_ids = batch['hr_token_ids'].to(self.device),
                hr_mask = batch['hr_mask'].to(self.device),
                only_ent_embedding=True
            )
            ent_tensor_list.append(outputs['ent_vectors'])
        entity_tensor = torch.cat(ent_tensor_list, dim=0)
        print('Finish encoding entities')
        # input()

        mean_rankf, mean_rankb = 0, 0
        forward_ = []
        backward_ = []
        mrrf, hit1f, hit3f, hit5f, hit10f = 0, 0, 0, 0, 0
        mrrb, hit1b, hit3b, hit5b, hit10b = 0, 0, 0, 0, 0
        count_f = 0
        count_b = 0
        for batch in tqdm(self.test_loader):
            targets = batch['targets']
            sources =  batch["sources"]

            batch = self.to_cuda(batch)
            batch['only_hr_embedding'] = True
            outputs = self.model(**batch)
            hr_vector = outputs['hr_vector']
            batch_score = torch.log_softmax(hr_vector.mm(entity_tensor.t()), dim=-1)

            batch_sorted_score, batch_sorted_indices = torch.sort(batch_score, dim=-1, descending=True)

            for idx in range(batch_score.size(0)):
                source = sources[idx]
                target = targets[idx]

                target_id = entity_dataset.entity_to_idx(target[0])
                # print()
                # print(target_id)
                # print(batch_sorted_indices[idx].size())
                cur_rank = torch.nonzero(batch_sorted_indices[idx].eq(target_id).long(), as_tuple=False).tolist()[0][0]
                # print(cur_rank)
                # input()
                cur_rank += 1
                prob = batch_score[idx,target_id].item()
                prob_total.append(prob)
                # prob_total.append(batch_sorted_score[idx,cur_rank-1].item())
                # prob_total.append(Log2(math.exp(batch_sorted_score[idx,cur_rank-1].item())))

                tmean_rank = cur_rank
                tmrr = 1.0 / cur_rank
                thit1 = 1 if cur_rank <= 1 else 0
                thit3 = 1 if cur_rank <= 3 else 0
                thit5 = 1 if cur_rank <= 5 else 0
                thit10 = 1 if cur_rank <= 10 else 0
                topk_scores = batch_sorted_score[idx, :k].tolist()
                topk_indices = batch_sorted_indices[idx, :k].tolist()
                # print(topk_indices)
                # input()
                pred_idx = topk_indices[0]
                pred = entity_dataset.idx_to_entity(pred_idx)

                topk_score_info = {}
                for (top_p, s) in zip(topk_indices, topk_scores):
                    pred_t = entity_dataset.idx_to_entity(top_p)
                    topk_score_info[pred_t] = s

                tmp = {
                    'input': source['input'],
                    'context': source['context'],
                    'pred': pred,
                    'rank': cur_rank,
                    'log_prob': prob,
                    'ground_truth': [],
                    'topk_score_info': json.dumps(topk_score_info)
                }
                if source['forward']:
                    mrrf += tmrr
                    count_f += 1
                    mean_rankf += tmean_rank
                    hit1f += thit1
                    hit3f += thit3
                    hit5f += thit5
                    hit10f += thit10
                    tttmp = {
                        'tail': target[0],
                        'sentence': json.dumps([source['id'], source['year'], source['rel_sent']])
                    }
                    tmp['ground_truth'].append(tttmp)
                    forward_.append(tmp)
                else:
                    mrrb += tmrr
                    count_b += 1
                    mean_rankb += tmean_rank

                    hit1b += thit1
                    hit3b += thit3
                    hit5b += thit5
                    hit10b += thit10
                    tttmp = {
                        'head': target[0],
                        'sentence': json.dumps([source['id'], source['year'], source['rel_sent']])
                    }
                    tmp['ground_truth'].append(tttmp)
                    backward_.append(tmp)
                tmp['hit1'] = thit1
                tmp['hit5'] = thit5
                tmp['hit10'] = thit10
        
        forward_ = sorted(forward_, key=lambda d: d['log_prob'], reverse=True) 

        with open('{}/eval_forward.json'.format(self.args.output), 'w', encoding='utf-8') as writer:
            writer.write(json.dumps(forward_, ensure_ascii=False, indent=4))

        backward_ = sorted(backward_, key=lambda d: d['log_prob'], reverse=True) 
        with open('{}/eval_backward.json'.format(self.args.output), 'w', encoding='utf-8') as writer:
            writer.write(json.dumps(backward_, ensure_ascii=False, indent=4))
        
        forward_metrics = {'mrr': mrrf, 'mean_rank': mean_rankf,'hit@1': hit1f, 'hit@3': hit3f, 'hit@5': hit5f, 'hit@10': hit10f}
        forward_metrics = {k: round(v / count_f, 4) for k, v in forward_metrics.items()}

        backward_metrics = {'mrr': mrrb, 'mean_rank': mean_rankb, 'hit@1': hit1b, 'hit@3': hit3b, 'hit@5': hit5b, 'hit@10': hit10b}
        backward_metrics = {k: round(v / count_b, 4) for k, v in backward_metrics.items()}

        metrics = {k: round((forward_metrics[k] * count_f + backward_metrics[k] * count_b) / (count_b + count_f), 4) for k in forward_metrics}
        # metrics['ppl'] = math.exp(-mean(prob_total))
        # metrics['ppl'] = 2 ** (-mean(prob_total))

        metrics['avg_logprob'] = mean(prob_total)
        metrics['ppl'] = math.exp(-metrics['avg_logprob'])
        with open('{}/metrics.json'.format(self.args.output), 'w', encoding='utf-8') as writer:
            writer.write('forward metrics: {}\n'.format(json.dumps(forward_metrics)))
            writer.write('backward metrics: {}\n'.format(json.dumps(backward_metrics)))
            writer.write('average metrics: {}\n'.format(json.dumps(metrics)))

        print('Evaluation takes {} seconds'.format(round(time() - start_time, 3)))
    
    @torch.no_grad()
    def ppl(self, entity_dataset, entity_loader):
        k = 10
        start_time = time()
        prob_total = []
        self.model.eval()
        ent_tensor_list = []
        for batch in tqdm(entity_loader):
            outputs = self.model(
                hr_token_ids = batch['hr_token_ids'].to(self.device),
                hr_mask = batch['hr_mask'].to(self.device),
                only_ent_embedding=True
            )
            ent_tensor_list.append(outputs['ent_vectors'])
        entity_tensor = torch.cat(ent_tensor_list, dim=0)
        print('Finish encoding entities')
        # input()
        ppls = []

        for batch in tqdm(self.test_loader):
            targets = batch['targets']
            sources =  batch["sources"]

            batch = self.to_cuda(batch)
            batch['only_hr_embedding'] = True
            outputs = self.model(**batch)
            hr_vector = outputs['hr_vector']
            batch_score = torch.log_softmax(hr_vector.mm(entity_tensor.t()), dim=-1)

            batch_sorted_score, batch_sorted_indices = torch.sort(batch_score, dim=-1, descending=True)

            for idx in range(batch_score.size(0)):
                source = sources[idx]
                target = targets[idx]

                target_id = entity_dataset.entity_to_idx(target[0])
                # print()
                # print(target_id)
                # print(batch_sorted_indices[idx].size())
                cur_rank = torch.nonzero(batch_sorted_indices[idx].eq(target_id).long(), as_tuple=False).tolist()[0][0]
                # print(cur_rank)
                # input()
                cur_rank += 1
                prob_total.append(batch_score[idx,target_id].item())
                # prob_total.append(batch_sorted_score[idx,cur_rank-1].item())
                # prob_total.append(Log2(math.exp(batch_sorted_score[idx,cur_rank-1].item())))

                tmp = {
                    'input': source['input'],
                    'context': source['context'],
                    'rank': cur_rank,
                    'ground_truth': {
                        'entity': target[0],
                        'sentence': json.dumps([source['id'], source['year'], source['rel_sent']])},
                    'log_prob': batch_sorted_score[idx,cur_rank-1].item()
                }
                ppls.append(tmp)

        with open('{}/eval_ppl.json'.format(self.args.output), 'w', encoding='utf-8') as writer:
            writer.write(json.dumps(ppls, ensure_ascii=False, indent=4))
        

        metrics = {}
        metrics['avg_logprob'] = mean(prob_total)
        metrics['ppl'] = math.exp(-metrics['avg_logprob'])
        # metrics['ppl'] = 2 ** (-mean(prob_total))

        with open('{}/metrics_ppl.json'.format(self.args.output), 'w', encoding='utf-8') as writer:
            writer.write('average metrics: {}\n'.format(json.dumps(metrics)))

        print('Evaluation takes {} seconds'.format(round(time() - start_time, 3)))

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

    def to_cuda(self, batch):
        for key in self.keys:
            batch[key] = batch[key].to(self.device)
        return batch


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

        test_loader, sampler, (entity_dataset, entity_loader) = get_loader(
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
        if args.ppl:
            trainer.ppl(entity_dataset, entity_loader)
        else:
            trainer.predict(entity_dataset, entity_loader)

    else:

        print(f'Building train loader at GPU {gpu}')

        train_loader, sampler, (entity_dataset, _) = get_loader(
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
    project_name = "SimKGC"

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
