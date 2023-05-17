torchrun \
    --nproc_per_node=4 \
    trainer.py \
    --model allenai/scibert_scivocab_uncased \
    --batch_size 100 \
    --valid_batch_size 100 \
    --lr 2.5e-6 \
    --fp16 \
    --pooling mean \
    --additive_margin 0.02 \
    --pre_batch 2 \
    --dataset_dir local_kg_dataset \
    --neg_num 5 \
    --output scibert_kg_checkpoint \
    --wandb \
    --distributed \
    --epochs 100 