torchrun \
    --nproc_per_node=4 \
    trainer_nc.py \
    --model t5-large \
    --batch_size 8 \
    --valid_batch_size 4 \
    --dataset_dir local_ct_dataset \
    --lr 6e-6 \
    --output t5_div_ct_checkpoint \
    --wandb \
    --distributed \
    --epochs 10 