#!/bin/bash

#example for cora dataset
python3 main.py --lm_file_path "./roberta-large" \
                 --lm_dim 1024 \
                 --e_hidden_size 128 \
                 --m_hidden_size 128 \
                 --num_classes 7 \
                 --batch_size 16 \
                 --e1_lr 1e-5 \
                 --e2_lr 1e-2 \
                 --m1_lr 1e-3 \
                 --m2_lr 5e-3 \
                 --e_dropout 0.5 \
                 --m_dropout 0.5 \
                 --epoch_all 50 \
                 --initial_dim 1024 \
                 --latent_dim 256 \
                 --early_epoch_e 0 \
                 --early_epoch_m 2 \
                 --epoch_e_class 3 \
                 --epoch_m_class 3 \
                 --e2_loss_Lweigh 0.2 \
                 --m2_loss_Lweigh 0.2 \
                 --folder_name "./TAG_data/" \
                 --dataset_name "cora" \
                 --sample_time 10 \
                 --gnn_name "GCN" \
                 --lm_name "roberta"
