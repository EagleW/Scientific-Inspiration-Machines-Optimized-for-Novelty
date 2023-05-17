torchrun \
    --nproc_per_node=4 \
    trainer.py \
    --model allenai/cs_roberta_base \
    --batch_size 100 \
    --valid_batch_size 100 \
    --lr 2.5e-6 \
    --fp16 \
    --pooling mean \
    --additive_margin 0.02 \
    --pre_batch 2 \
    --dataset_dir local_ct_dataset \
    --neg_num 5 \
    --output roberta_ct_checkpoint \
    --wandb \
    --distributed \
    --epochs 100  