python \
    trainer_nc_cbs.py \
    --model t5-large \
    --batch_size 4 \
    --dataset_dir local_dataset \
    --valid_batch_size 2 \
    --lr 6e-6 \
    --load t5_cbs_checkpoint/BEST \
    --output t5_cbs_checkpoint \
    --test_only 