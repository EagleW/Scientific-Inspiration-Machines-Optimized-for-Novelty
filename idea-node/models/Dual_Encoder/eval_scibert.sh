python \
      trainer_nc.py \
      --model allenai/scibert_scivocab_uncased \
      --batch_size 150 \
      --valid_batch_size 140 \
      --lr 2.5e-5 \
      --load scibert_checkpoint/BEST \
      --output scibert_checkpoint \
      --pooling mean \
      --additive_margin 0.02 \
      --pre_batch 2 \
      --test_only