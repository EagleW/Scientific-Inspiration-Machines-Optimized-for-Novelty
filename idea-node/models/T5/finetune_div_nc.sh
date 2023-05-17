torchrun \
    --nproc_per_node=4 \
    trainer_nc.py \
    --model t5-large \
    --batch_size 16 \
    --valid_batch_size 4 \
    --lr 6e-6 \
    --dataset_dir local_dataset \
    --output t5_div_checkpoint \
    --wandb \
    --distributed \
    --epochs 10 