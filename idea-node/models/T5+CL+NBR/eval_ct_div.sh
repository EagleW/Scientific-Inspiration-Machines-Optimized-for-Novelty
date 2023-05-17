python \
    trainer_lm.py \
    --model t5-large \
    --batch_size 8 \
    --valid_batch_size 4 \
    --lr 6e-6 \
    --beam_size 20 \
    --neg_num 2 \
    --dataset_dir local_ct_dataset \
    --load t5_cl_ct_checkpoint/BEST \
    --output t5_cl_ct_checkpoint \
    --test_only  