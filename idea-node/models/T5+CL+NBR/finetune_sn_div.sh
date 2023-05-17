torchrun \
    --nproc_per_node=4 \
    trainer_nc.py \
    --model t5-large \
    --batch_size 8 \
    --valid_batch_size 4 \
    --dataset_dir local_sn_dataset \
    --lr 6e-6 \
    --output t5_div_sn_checkpoint \
    --wandb \
    --distributed \
    --epochs 10 