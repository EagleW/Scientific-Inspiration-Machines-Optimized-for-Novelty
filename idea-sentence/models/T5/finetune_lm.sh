torchrun \
    --nproc_per_node=4 \
    trainer_lm.py \
    --model t5-large \
    --batch_size 8 \
    --valid_batch_size 16 \
    --beam_size 10 \
    --neg_num 2 \
    --dataset_dir local_context_dataset \
    --lr 6e-6 \
    --output t5_checkpoint \
    --wandb \
    --distributed \
    --epochs 10 "$@"