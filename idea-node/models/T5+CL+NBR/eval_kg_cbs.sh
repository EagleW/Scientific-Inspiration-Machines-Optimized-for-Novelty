python \
    trainer_nc_cbs.py \
    --model t5-large \
    --batch_size 4 \
    --valid_batch_size 2 \
    --lr 6e-6 \
    --dataset_dir local_kg_dataset \
    --load t5_cbs_kg_checkpoint/BEST \
    --output t5_cbs_kg_checkpoint \
    --test_only  