import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv
from transformers import AutoTokenizer, AutoModel, AdamW
from transformers import RobertaModel, RobertaTokenizer
from torch_geometric.data import Data, DataLoader
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from data_pro import read_data
from lm_ft import ft_Model
import time
import numpy as np
from em_models import EStepModel,MStepModel,TextDataset,device,TextEncoder
from utils import args
device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
class EMTrainer:
    def __init__(self, config,tokenizer,Lmft_Model):
        self.device = device
        
        # 初始化模型
        self.text_encoder = TextEncoder(tokenizer=tokenizer,ft_model=Lmft_Model)
        
        self.e_model = EStepModel(
            ft_model=Lmft_Model,
            lm_dim=config['lm_dim'],
            e_hidden_size=config['e_hidden_size'],
            num_classes=config['num_classes'],
            dropout=config['e_dropout']
        ).to(self.device)

        self.m_model = MStepModel(
            lm_dim=config['lm_dim'],
            m_hidden_size=config['m_hidden_size'],
            num_classes=config['num_classes'],
            dropout=config['m_dropout']
        ).to(self.device)
        
        # 优化器
        self.e_optimizer = torch.optim.Adam(self.e_model.parameters(), lr=config['e_lr'])
        self.m_optimizer = torch.optim.Adam(self.m_model.parameters(), lr=config['m_lr'])
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.config = config
    
    def initialize_pseudo_labels(self,x,graph_data):
        """初始化伪标签"""
        self.m_model.eval()
        with torch.no_grad():
            _,pseudo_logits = self.m_model(
                x.to(self.device),
                graph_data.to(self.device)
            )
            pseudo_labels=pseudo_logits.argmax(dim=-1)
        return pseudo_labels
    

    def e_step(self, e_dataloader,train_mask, test_mask, true_labels, pseudo_labels):
        self.e_model.train()
        e_predictions = torch.zeros(len(true_labels), dtype=torch.long).to(device)
        embs_e= torch.zeros(len(true_labels),self.config['e_hidden_size']).to(device)
        avg_loss=0
        t=time.time()
        for step,batch in enumerate(e_dataloader):
            batch_ids,batch_mask = batch['ids'].to(self.device),batch['mask'].to(self.device)
            batch_idx = batch['idx']
            
            # 获取当前批次的标签
            batch_labels = torch.where(
                train_mask[batch_idx],
                true_labels[batch_idx],
                pseudo_labels[batch_idx]
            ).to(self.device)
            
            e,logits_e = self.e_model(batch_ids,batch_mask)
            embs_e[batch_idx]=e
            loss = self.criterion(logits_e, batch_labels)
            avg_loss+=loss.item()
            # 反向传播
            self.e_optimizer.zero_grad()
            loss.backward()
            self.e_optimizer.step()
            # 保存预测结果
            e_predictions[batch_idx] = logits_e.argmax(dim=-1)
            #pesudo_emb_m[batch_idx],new_emb_m[batch_idx]=emb_e,emb_m
            if step%20==0:
                print(f'e_step:{step},train_loss:{loss},time:{time.time()-t}')
        e_test_acc = self.compute_accuracy(e_predictions, true_labels, test_mask)
        
        print(f'e_test: {e_test_acc},train_avg_loss:{avg_loss/len(e_dataloader)}')

        self.e_model.eval()
        return e_predictions,e_test_acc,embs_e
    
    def m_step(self, x,graph_data, train_mask, test_mask, true_labels, pseudo_labels):
        self.m_model.train()
        labels = torch.where(train_mask, true_labels, pseudo_labels).to(self.device)
        
        # 前向传播
        embs_m,logits_m = self.m_model(
            x.to(self.device),
            graph_data.to(self.device)
        )
        loss = self.criterion(logits_m, labels)
        m_predictions=logits_m.argmax(dim=-1)
        # 反向传播
        self.m_optimizer.zero_grad()
        loss.backward()
        self.m_optimizer.step()

        m_test_acc = self.compute_accuracy(m_predictions, true_labels, test_mask)
        print(f'm_test: {m_test_acc},train_m_loss:{loss}')
        self.m_model.eval()
        return m_predictions,m_test_acc,embs_m
    
    def compute_accuracy(self, pred_labels, true_labels, mask):
        
        correct = (pred_labels[mask] == true_labels[mask]).sum().item()
        total = mask.sum().item()
        return correct / total if total > 0 else 0.0
    
    def train(self, texts, graph_data, train_mask, test_mask, labels, num_iterations):
        # 获取文本嵌入
        m_initial_embeddings,all_ids,all_mask= self.text_encoder.encode(texts)
        
        dataset = TextDataset(all_ids,all_mask, torch.arange(len(all_ids)))

        dataloader = TorchDataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=True
        )
        ti=time.time()
        pseudo_labels = self.initialize_pseudo_labels(m_initial_embeddings,graph_data)
        best_acc=0
        
        for iteration in range(num_iterations):

            print(iteration)
            e_predictions,acc_e,embs_e = self.e_step(
                dataloader,
                train_mask,
                test_mask,
                labels,
                pseudo_labels,
    
                
            )
            # M-step
            m_predictions,acc_m,embs_m = self.m_step(
                m_initial_embeddings,
                graph_data,
                train_mask,
                test_mask,
                labels,
                e_predictions,
                
            )
            pseudo_labels=m_predictions.detach()
            print(f'iteration:{iteration},time:{time.time()-ti}')
            if acc_e>best_acc and acc_e>acc_m :
                np.save('./TAG_data/citeseer/embs_e_citeseer_GLEM.npy', embs_e.cpu().detach().numpy())
            elif acc_m>best_acc and acc_m>acc_e :
                np.save('./TAG_data/citeseer/embs_m_citeseer_GLEM.npy', embs_m.cpu().detach().numpy())
            #np.save('./TAG_data/citeseer/labels_citeseer_GLEM.npy', labels.cpu().detach().numpy())
            #np.save('./TAG_data/citeseer/test_mask_citeseer_GLEM.npy', test_mask.cpu().detach().numpy())
def indices_to_mask(indices, num_nodes):
    
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[indices] = True
    return mask

# 使用示例
def main():
    
    # 初始化tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained('./roberta-large')
    lm_model = AutoModel.from_pretrained('./roberta-large')
    config = {
        'e_hidden_size': 256,
        'm_hidden_size': 256,
        'num_classes': 6,
        'batch_size': 16,
        'e_lr': 1e-5,
        'm_lr': 1e-2,
        'e_dropout':0.5,
        'm_dropout':0.5,
        "epoch":50,
        'lm_dim':1024, 
    }

    #加载微调好的lm模型
    Lmft_Model=ft_Model(lm_model,hidden_dim=config['lm_dim'],labels_dim=config['num_classes'])#已经微调好的lm模型
    Lmft_Model.load_state_dict(torch.load('./TAG_data/citeseer/best_roberta_model_citeseer.pth',map_location=device))
    
    if args.dataset_name in ['cora','citeseer','pubmed']:
        Data_splited=read_data(args=args,device=device)
    trainer = EMTrainer(config,tokenizer,Lmft_Model)
    
    

    trainer.train(
        texts=Data_splited.texts,
        graph_data=Data_splited.edge_index,
        train_mask=Data_splited.train_mask.to(device),
        test_mask=Data_splited.test_mask.to(device),
        labels=Data_splited.true_labels.to(device),
        num_iterations=config['epoch']
    )

if __name__=="__main__":
    main()