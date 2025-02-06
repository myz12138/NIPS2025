import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import Data, DataLoader
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from data_pro import encode_texts
from utils import args
import time
device = torch.device('cuda:'+args.cuda_number if torch.cuda.is_available() else 'cpu')
class TextEncoder:
    def __init__(self, tokenizer,ft_model):
        self.tokenizer =tokenizer
        self.model = ft_model.to(device)
        self.model.eval()  #initilize embeddings,do not update parameters
        
    @torch.no_grad()
    def encode(self, texts):
        batch_size=int(len(texts)/100)
        t=time.time()
        embeddings,all_ids,all_mask = [],[],[]
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            input_ids,attention_mask = encode_texts(
                batch_texts,
                device,
                self.tokenizer,
                truncation_length=512
            )
            batch_embeddings= self.model(**{'input_ids':input_ids,"attention_mask":attention_mask}).last_hidden_state
            batch_embeddings=torch.mean(batch_embeddings,dim=-2)
            all_ids.append(input_ids.cpu())
            all_mask.append(attention_mask.cpu())
            embeddings.append(batch_embeddings.cpu())
            if i%(batch_size)==0:
                print(f'initial_step:{i},time:{time.time()-t}')
        return torch.cat(embeddings,dim=0),torch.cat(all_ids, dim=0),torch.cat(all_mask, dim=0)
        
class E1StepModel(nn.Module):
    def __init__(self,ft_model,lm_dim,latent_dim):
        super().__init__()
        self.emb_model=ft_model
        self.mlp_latent_u=nn.Sequential(
            nn.Linear(lm_dim,latent_dim),
            
        )
        self.mlp_latent_sigma=nn.Sequential(
            nn.Linear(lm_dim,latent_dim),
            nn.ReLU(),
        )
    def forward(self, e_ids,e_mask):
        x=self.emb_model(**{'input_ids':e_ids,"attention_mask":e_mask}).last_hidden_state
        x=torch.mean(x,dim=-2)
        u,sigma=self.mlp_latent_u(x),self.mlp_latent_sigma(x)
        sigma=torch.clamp(sigma, min=1e-2) 
        return u,sigma

class E2StepModel(nn.Module):
    def __init__(self,latent_dim,e_hidden_size, num_classes,dropout):
        super().__init__()
        self.mlp_class = nn.Sequential(
            nn.Linear(latent_dim, e_hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(e_hidden_size, num_classes),
        )

    def forward(self, samples_emb):
        logits=self.mlp_class(samples_emb)
        return logits


class M1StepModel(nn.Module):
    def __init__(self, initial_dim,latent_dim,dropout,gnn_name='GCN'):
        super().__init__()
        if gnn_name=='GCN':
            self.gnn_1= GCNConv(initial_dim,latent_dim)
            self.gnn_2=GCNConv(latent_dim,latent_dim)
        elif gnn_name=='GAT':
            self.gnn_1= GATConv(initial_dim, latent_dim, heads=4)
            self.gnn_2= GATConv(latent_dim, latent_dim, heads=4)
        elif gnn_name=='MLP':
            self.gnn_1= nn.Linear(initial_dim, latent_dim)
            self.gnn_2= nn.Linear(latent_dim, latent_dim)
        elif gnn_name=='SAGE':
            self.gnn_1= SAGEConv(initial_dim, latent_dim)
            self.gnn_2= SAGEConv(latent_dim, latent_dim)             
        self.mlp_u=nn.Linear(latent_dim,latent_dim)
        self.dropout=nn.Dropout(dropout)
        self.mlp_sigma=nn.Linear(latent_dim,latent_dim)
        self.act=nn.ReLU()
        
    def forward(self, initial_e_emb, edge_index):
        initial_e_emb=self.act(self.gnn_1(initial_e_emb,edge_index))
        initial_e_emb=self.gnn_2(self.dropout(initial_e_emb),edge_index)
        u,sigma=self.mlp_u(initial_e_emb),self.mlp_sigma(initial_e_emb)
        sigma=torch.clamp(self.act(sigma),min=1e-2) 
        return u,sigma


class M2StepModel(nn.Module):
    def __init__(self, latent_dim, m_hidden_size, num_classes,dropout,gnn_name='GCN'):
        super().__init__()
        if gnn_name=='GCN':
            self.gnn_1= GCNConv(latent_dim,m_hidden_size)
            self.gnn_2=GCNConv(m_hidden_size, num_classes)
        elif gnn_name=='GAT':
            self.gnn_1= GATConv(latent_dim,m_hidden_size, heads=4)
            self.gnn_2= GATConv(m_hidden_size, num_classes, heads=4)
        elif gnn_name=='MLP':
            self.gnn_1= nn.Linear(latent_dim,m_hidden_size)
            self.gnn_2= nn.Linear(m_hidden_size, num_classes)
        elif gnn_name=='SAGE':
            self.gnn_1= SAGEConv(latent_dim,m_hidden_size)
            self.gnn_2= SAGEConv(m_hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, last_e_emb, edge_index):
        x1 = F.relu(self.gnn_1(last_e_emb, edge_index))
        x2 = self.dropout(x1)
        logits = self.gnn_2(x2, edge_index)
        return logits
    
class TextDataset(Dataset):
    def __init__(self, idss,masks, idxs):
        self.idss = idss
        self.masks=masks
        self.idxs =idxs
        
    def __len__(self):
        return len(self.idxs)
    
    def __getitem__(self, index):
        return {
            'ids': self.idss[index],
            'mask':self.masks[index],
            'idx': self.idxs[index]
        }

class EDataset(Dataset):
    def __init__(self, embs, idxs):
        self.embs = embs
        self.idxs =idxs
        
    def __len__(self):
        return len(self.idxs)
    
    def __getitem__(self, index):
        return {
            'embs': self.embs[index],
            'idx': self.idxs[index]
        }
    
class TaskOnlyModel(nn.Module):#model for ablation study (Concat) 
    def __init__(self,ft_model,lm_dim,concat_size,num_classes,dropout):
        super().__init__()
        self.emb_model=ft_model
        self.mlp_text=nn.Sequential(
            nn.Linear(lm_dim,concat_size),
            nn.LeakyReLU(),
        )
        self.gnn_layers1=GCNConv(lm_dim,2*concat_size)
        self.act=nn.LeakyReLU()
        self.dropout=nn.Dropout(dropout)
        self.gnn_layers2=GCNConv(2*concat_size,concat_size)
        
        self.mlp_class=nn.Sequential(
            nn.Linear(2*concat_size,concat_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(concat_size,num_classes),
        )

    def forward(self, e_ids,e_mask,x_graph,edge_index,batch_idx,data_mask):
        x_text=self.emb_model(**{'input_ids':e_ids,"attention_mask":e_mask}).last_hidden_state
        x_text=torch.mean(x_text,dim=-2)
        x_text,x_graph=self.mlp_text(x_text),self.gnn_layers1(x_graph,edge_index)
        x_graph=self.gnn_layers2(self.dropout(self.act(x_graph)),edge_index)
        x_concat=torch.cat([x_text,x_graph[batch_idx]],dim=-1)
        logits=self.mlp_class(x_concat)
        return logits,x_text,x_graph[batch_idx]
    
class AlignOnlyModel(nn.Module):#model for ConGraT (MSE)
    def __init__(self,ft_model,lm_dim,align_size,dropout):
        super().__init__()
        self.emb_model=ft_model
        self.mlp_text=nn.Linear(lm_dim,align_size)
        self.gnn_layers1=GCNConv(lm_dim,2*align_size)
        self.act=nn.LeakyReLU()
        self.dropout=nn.Dropout(dropout)
        self.gnn_layers2=GCNConv(2*align_size,align_size)

    def forward(self, e_ids,e_mask,x_graph,edge_index,batch_idx,data_mask):
        x_text=self.emb_model(**{'input_ids':e_ids,"attention_mask":e_mask}).last_hidden_state
        x_text=torch.mean(x_text,dim=-2)
        x_text,x_graph=self.mlp_text(x_text),self.gnn_layers1(x_graph,edge_index)
        x_graph=self.gnn_layers2(self.dropout(self.act(x_graph)),edge_index)
        return x_text,x_graph[batch_idx][data_mask]

class AlignforClassModel(nn.Module):
    def __init__(self,align_size,num_classes):
        super().__init__()
        self.classfy=nn.Linear(align_size,num_classes)

    def forward(self,x_t):
        return self.classfy(x_t)
