import torch
import torch.nn as nn
import time
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader as TorchDataLoader
from my_em_models import TextEncoder,TaskOnlyModel,AlignforClassModel,AlignOnlyModel,TextDataset,device,TextGraphDataset
from lm_ft import ft_Model
from my_em_train import EMTrainer
from transformers import AutoTokenizer, AutoModel, AdamW
from data_pro import read_data_products,read_data_arxiv23,read_data
import argparse
#First: TaskonlyModel: Only with task,without align, Concat embeds of LM and GCN,and mlp for classifying
class AblationTrainer:
    def __init__(self, config,tokenizer,Lmft_Model,flag):
        self.text_encoder = TextEncoder(tokenizer=tokenizer,ft_model=Lmft_Model)
        self.device =device
        self.flag=flag
        if flag=='align':
            self.task_model =TaskOnlyModel(
                ft_model=Lmft_Model,
                lm_dim=config['lm_dim'],
                concat_size=config['concat_size'],
                num_classes=config['num_classes'],
                dropout=config['dropout'],
            ).to(self.device)
            self.task_optimizer=torch.optim.Adam(self.task_model.parameters(), lr=config['align_lr'])

        self.criterion_class = nn.CrossEntropyLoss()
        self.criterion_mse=nn.MSELoss()
        self.config = config

    def compute_accuracy(self, pred_labels, true_labels):
        correct = (pred_labels == true_labels).sum().item()
        return correct 

    def align_setp(self,data_loader,x_graph,edge_index):
        for epoch in range(self.config['align_epoch']):
            best_preds=0
            self.task_model.train()
            all_loss=0
            for step,batch in enumerate(data_loader):
                batch_ids,batch_token_mask,batch_idx = batch['ids'].to(self.device),batch['mask'].to(self.device),batch['idx']
                train_mask=batch['train_mask']
                batch_label=batch['label']
                if len(batch_idx)!=0:
                    logits,_,_=self.task_model(batch_ids,batch_token_mask,x_graph,edge_index,batch_idx,train_mask)
                    loss=self.criterion_class(logits[train_mask],batch_label[train_mask])
                    all_loss+=loss.item()
                    self.task_optimizer.zero_grad()
                    loss.backward()
                    self.task_optimizer.step() 
                    if step%30==0:
                        print(f" step:{step},Loss: {loss}")
            print(f"Training Loss: {all_loss / len(data_loader)}")
            preds=0
            self.task_model.eval()
            embs_e,embs_m=torch.zeros(x_graph.shape[0],256).to(device),torch.zeros(x_graph.shape[0],256).to(device)
            print(embs_e.shape,embs_m.shape)
            with torch.no_grad():
                for batch in data_loader:
                    batch_ids,batch_token_mask,batch_idx = batch['ids'].to(self.device),batch['mask'].to(self.device),batch['idx']
                    val_mask=batch['test_mask']
                    batch_label=batch['label']
                    logits,embs_e_batch,embs_m_batch=self.task_model(batch_ids,batch_token_mask,\
                                                                     x_graph,edge_index,batch_idx,val_mask)
                    
                    embs_e[batch_idx],embs_m[batch_idx]=embs_e_batch,embs_m_batch
                    if val_mask.sum().item()!=0:
                        preds+=self.compute_accuracy(logits[val_mask].argmax(dim=-1), batch_label[val_mask])
                print(f'valpreds:{preds}')
                if preds>best_preds:
                    best_preds=preds
                    best_model_state =self.task_model.state_dict()
                    torch.save(best_model_state,'./ablation_concat_model.path')
                
        pred_test=0
        with torch.no_grad():
            for batch in data_loader:
                batch_ids,batch_token_mask,batch_idx = batch['ids'].to(self.device),batch['mask'].to(self.device),batch['idx']
                test_mask=batch['test_mask']
                batch_label=batch['label']
                if len(batch_idx)!=0:
                    logits=self.task_model(batch_ids,batch_token_mask,x_graph,edge_index,batch_idx,test_mask)
                    pred_test+=self.compute_accuracy(logits[test_mask].argmax(dim=-1), batch_label[test_mask])
            print(f'testpreds:{pred_test}')
        return pred_test

    def train(self,texts, edge_index, train_mask,val_mask, test_mask, labels):
        x_graph,all_ids,all_mask= self.text_encoder.encode(texts)
        x_graph,edge_index=x_graph.to(self.device),edge_index.to(self.device)
        dataset = TextGraphDataset(all_ids,all_mask,labels,train_mask,val_mask, test_mask)
        dataloader = TorchDataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=False
        )
        if self.flag=='align':
            preds=self.align_setp(dataloader,x_graph,edge_index)
            print(f'best_test:{preds/test_mask.sum().item()}')        
            
if __name__=='__main__': 
    tokenizer = AutoTokenizer.from_pretrained('./roberta-large')
    lm_model = AutoModel.from_pretrained('./roberta-large')
    config = {
        'lm_dim':1024,
        'concat_size': 256,
        'align_size':256,
        'num_classes':  6,
        'batch_size':16,
        'align_lr':1e-5,
        'task_lr':1e-5,
        'task_class_lr':1e-3,
        'dropout':0.5,
        "align_epoch":5,
        "task_epoch":1,
        'task_class_epoch':5,
        'gnn_name':'GCN',
        'dataset_name':'',
        'folder_name':'./TAG_data/',
        'lm_name':'roberta',
    }
    parser = argparse.ArgumentParser(description='parser example')
    parser.add_argument('--dataset_name', default=config['dataset_name'], type=str, help='name of dataset')
    parser.add_argument('--lm_file_path', default='./roberta', type=str, help='path of lm_model')
    parser.add_argument('--folder_name', default=config['folder_name'], type=str, help='name of folder for dataset and their model')
    parser.add_argument('--flag', default='em2', type=str, help='name of folder for dataset and their model')
    args_ablation = parser.parse_args()

    Lmft_Model=ft_Model(lm_model,hidden_dim=config['lm_dim'],labels_dim=config['num_classes']).to(device)
    if config['dataset_name'] in ['cora','citeseer','pubmed']:
        Data_splited=read_data(args=args_ablation,device=device)
    elif config['dataset_name']=='arxiv-2023':
        Data_splited=read_data_arxiv23(device=device)
    elif config['dataset_name']=='ogbn_products(subset)':
        Data_splited=read_data_products(device=device)
    else:
        print('error')
        
    Lmft_Model=ft_Model(lm_model,hidden_dim=config['lm_dim'],labels_dim=config['num_classes']).to(device)
    load_ft_model_path=config['folder_name']+config['dataset_name']+'/'+'best_'+config['lm_name']+'_model_'+config['dataset_name']+'.pth'
    print(load_ft_model_path)
    Lmft_Model.load_state_dict(torch.load(load_ft_model_path,map_location=device))
    Lmft_Model_trans=Lmft_Model.lm_model

    trainer = AblationTrainer(config,tokenizer,Lmft_Model_trans,flag='align')
    print(trainer.flag)

    trainer.train(
        texts=Data_splited.texts,
        edge_index=Data_splited.edge_index,
        train_mask=Data_splited.train_mask,
        val_mask=Data_splited.val_mask,
        test_mask=Data_splited.test_mask,
        labels=Data_splited.true_labels,
    )
    
            