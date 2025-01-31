import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from data_pro import encode_texts
device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
class TextEncoder:
    def __init__(self, tokenizer,ft_model):#ft_model是包含分类器的，.lm_model是不包含分类器的
        self.tokenizer =tokenizer
        self.model = ft_model.lm_model.to(device)
        self.model.eval()  # 设置为评估模式，不更新参数
        
    @torch.no_grad()
    def encode(self, texts,batch_size=8):
        embeddings,all_ids,all_mask = [],[],[]
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(device)
            input_ids,attention_mask=inputs['input_ids'],inputs['attention_mask']
            batch_embeddings= self.model(**{'input_ids':input_ids,"attention_mask":attention_mask}).pooler_output
            all_ids.append(input_ids.cpu())
            all_mask.append(attention_mask.cpu())
            embeddings.append(batch_embeddings.cpu())
            if i%50==0:
                print(f'initial_step:{i}')
        return torch.cat(embeddings, dim=0),torch.cat(all_ids, dim=0),torch.cat(all_mask, dim=0)
    
    def encode_optimized(self,optimized_texts):
        optimized_ids,optimized_mask=encode_texts(optimized_texts,device,self.tokenizer,truncation_length=512)
        return optimized_ids,optimized_mask
        
class EStepModel(nn.Module):
    def __init__(self,ft_model,lm_dim,e_hidden_size, num_classes,dropout):
        super().__init__()
        self.emb_model=ft_model.lm_model
    
        self.mlp_class1 = nn.Linear(lm_dim, e_hidden_size)
        self.act=nn.LeakyReLU()
        self.dropout=nn.Dropout(dropout)
        self.mlp_class2=nn.Linear(e_hidden_size, num_classes)        
        
    def forward(self, e_ids,e_mask):
        x=self.emb_model(**{'input_ids':e_ids,"attention_mask":e_mask}).pooler_output
        x=self.mlp_class1(x)
        logits=self.dropout(self.act(x))
        logits=self.mlp_class2(logits)
        return x,logits


class MStepModel(nn.Module):
    def __init__(self, lm_dim, m_hidden_size, num_classes,dropout):
        super().__init__()
        self.gcn1 = GCNConv(lm_dim, m_hidden_size)
        self.gcn2 = GCNConv( m_hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    
    def forward(self, last_e_emb, edge_index):
    
        x1 = self.gcn1( last_e_emb, edge_index)
        x2 = self.dropout(F.relu(x1))
        logits = self.gcn2(x2, edge_index)
        return x1,logits



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

