torchrun \
    --nproc_per_node=4 \
    trainer_nc.py \
    --model allenai/scibert_scivocab_uncased \
    --batch_size 150 \
    --valid_batch_size 150 \
    --lr 2.5e-6 \
    --fp16 \
    --pooling mean \
    --additive_margin 0.02 \
    --pre_batch 2 \
    --neg_num 5 \
    --output scibert_cl_checkpoint \
    --wandb \
    --distributed \
    --epochs 100 
       
       