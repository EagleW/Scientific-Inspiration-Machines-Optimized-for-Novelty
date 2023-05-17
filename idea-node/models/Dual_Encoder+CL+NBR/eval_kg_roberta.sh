python \
    trainer.py \
    --model allenai/cs_roberta_base \
    --batch_size 150 \
    --valid_batch_size 140 \
    --lr 2.5e-5 \
    --load roberta_kg_checkpoint/BEST \
    --output roberta_kg_checkpoint \
    --dataset_dir local_kg_dataset \
    --pooling mean \
    --additive_margin 0.02 \
    --pre_batch 2 \
    --test_only 