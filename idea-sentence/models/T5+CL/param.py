import os
import random
import torch
import argparse
import warnings
import numpy as np

import torch.backends.cudnn as cudnn

def parse_args():
    parser = argparse.ArgumentParser()
    # Model Loading
    parser.add_argument('--model', 
                        default='t5-large', 
                        type=str, help='path to pretrained model')
    parser.add_argument('--load', type=str, default=None,
                        help='Load the model (usually the fine-tuned model).')
    parser.add_argument('--output', type=str, 
                        default='kgt5_local_language_contrastive_checkpoint',
                        help='Save the model (usually the fine-tuned model).')

    # Training Hyper-parameters
    parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training. ')
    parser.add_argument('-b', '--batch_size', default=128, type=int,
                        help='mini-batch size (default: 256)')
    parser.add_argument('--valid_batch_size', type=int, default=None)
    parser.add_argument('--neg_num', default=10, type=int,
                        help='number of context entities used for negatives')
    parser.add_argument('--beam_size', type=int, default=50
                        )
    parser.add_argument('--num_predictions',type=int, default=10)

    # CPU/GPU
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument("--distributed", action='store_true')

    # Data Splits
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--ppl', action='store_true')
    parser.add_argument("--dataset_dir", 
                        default='local_language_contrastive_dataset', 
                        type=str,  help='which dataset')

    # Quick experiments
    parser.add_argument('--train_topk', type=int, default=-1)
    parser.add_argument('--valid_topk', type=int, default=-1)


    # Training configuration
    parser.add_argument('--clip_grad_norm', type=float, default=10.0)
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_eps", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--lr-scheduler', default='linear', type=str,
                    help='Lr scheduler to use')
    parser.add_argument('--patient', type=int, default=4)
    parser.add_argument('--epochs', default=10, type=int,
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=10, type=int,
                    help='number of data loading workers')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--warmup', default=400, type=int,
                        help='warmup steps')


    args = parser.parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    return args

if __name__ == '__main__':
    args = parse_args()