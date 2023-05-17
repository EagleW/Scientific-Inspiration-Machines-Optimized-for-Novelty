python \
    trainer_nc.py \
    --model t5-large \
    --batch_size 4 \
    --valid_batch_size 2 \
    --dataset_dir local_dataset \
    --lr 6e-6 \
    --load t5_div_checkpoint/BEST \
    --output t5_div_checkpoint \
    --test_only 