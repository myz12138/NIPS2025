
import argparse   
from transformers import AutoTokenizer,AutoModel
import torch

parser = argparse.ArgumentParser(description='parser example')

parser.add_argument('--batch_size', default=16, type=int, help='batch size')#0.9457 0.9507 0.7616

parser.add_argument('--e1_lr', default=1e-5, type=float, help='learning rate of e1-step: approximate Q to P for kl-loss')

parser.add_argument('--e2_lr', default=1e-2, type=float, help='learning rate of e2-step: approximate Q to P for crossstropy-loss')

parser.add_argument('--m1_lr', default=1e-3, type=float, help='learning rate of m1-step: approximate P to Q for kl-loss')

parser.add_argument('--m2_lr', default=5e-3, type=float, help='learning rate of m2-step: approximate P to Q for crossstropy-loss')

parser.add_argument('--epoch_all', default=50, type=int, help='epoch of train for classifier')

parser.add_argument('--early_epoch_e', default=0, type=int, help='early epoch for kl_loss in e step')

parser.add_argument('--early_epoch_m', default=2, type=int, help='early epoch for kl_loss in m step')

parser.add_argument('--epoch_e_class', default=3, type=int, help='epoch for class_loss in e step for each train step')

parser.add_argument('--epoch_m_class', default=3, type=int, help='epoch for class_loss in m step for each train step')

parser.add_argument('--e2_loss_Lweigh', default=0.6, type=float, help='the weight for observed data for classification in step e2')

parser.add_argument('--m2_loss_Lweigh', default=0.6, type=float, help='the weight for observed data for classification in step m2')

parser.add_argument('--lm_name', default='roberta', type=str, help='path of lm_model')

parser.add_argument('--e_hidden_size', default=128, type=int, help='size of hidden layer in e-step-classification')

parser.add_argument('--m_hidden_size', default=128, type=int, help='size of hidden layer in m-step-classification')

parser.add_argument('--num_classes', default=7, type=int, help='the number of nodes-class')

parser.add_argument('--lm_dim', default=1024, type=int, help='dim of lm_model for e-step')

parser.add_argument('--initial_dim', default=1024, type=int, help='dim of initial embedding for m-step')

parser.add_argument('--e_dropout', default=0.5, type=float, help='dropout for e_step in class-loss')

parser.add_argument('--m_dropout', default=0.5, type=float, help='dropout for m_step in class-loss')

parser.add_argument('--latent_dim', default=256, type=int, help='dim of lattent feature')

parser.add_argument('--gnn_name', default='GCN', type=str, help='dim of lattent feature')

parser.add_argument('--dataset_name', default='cora', type=str, help='name of dataset')#arxiv-2023ogbn_products(subset)

parser.add_argument('--lm_file_path', default='./roberta-large', type=str, help='path of lm_model')

parser.add_argument('--folder_name', default='./TAG_data/', type=str, help='name of folder for dataset and their model')

parser.add_argument('--flag', default='em2', type=str, help='name of folder for dataset and their model')

parser.add_argument('--sample_time', default=10, type=int, help='number of cuda')

parser.add_argument('--cuda_number', default='1', type=str, help='number of cuda')
args = parser.parse_args()


