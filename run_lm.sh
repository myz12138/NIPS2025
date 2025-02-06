#!/bin/bash

python3 ./src/LM/lm_ft.py --lm_file_path "./roberta-large" \
                     --lm_name "roberta" \
                     --folder_name "./TAG_data/" \
                     --dataset_name "cora" \
                     --batch_size 64 \
                     --epoch 10 \
                     --lm_dim 1024 \
                     --label_dim 10 \
                     --learning_rate 1e-5 \
                     --train_ratio 0.6 \
                     --val_ratio 0.2 \
                     --cuda_number 0
