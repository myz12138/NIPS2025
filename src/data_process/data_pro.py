import torch
import json
import csv
import numpy as np
from torch_geometric.data import Data
from torch.utils.data import DataLoader,Dataset
from sklearn.model_selection import train_test_split

class LmftDataset(Dataset):
    def __init__(self, ids,mask, labels):
        self.ids = ids
        self.mask = mask
        self.labels=labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.ids[idx], self.mask[idx],self.labels[idx]


class emDataset(Dataset):
    def __init__(self,input_ids,attention_mask,label):
        self.input_ids=input_ids
        self.attention_mask=attention_mask
        self.label=label
        self.len=len(label)

    def __getitem__(self, index) :
        return self.input_ids[index],self.attention_mask[index],self.label[index]
    def __len__(self):
        return self.len
        
def indices_to_mask(indices, num_nodes):
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[indices] = True
    return mask

def encode_texts(texts,device,tokenizer,truncation_length):
    inputs = tokenizer(texts, return_tensors='pt',  truncation=True,max_length=truncation_length,padding='max_length').to(device)
    return inputs['input_ids'], inputs['attention_mask']

#read data for cora and citeseer dataset
def read_data(args,device):
    data= torch.load(args.folder_name+args.dataset_name+'/'+args.dataset_name+'_fixed_tfidf.pt')
    x_feature=data.x
    edge_index=data.edge_index
    true_labels=data.y
    texts=data.node_stores[0]._mapping['raw_texts']

    if args.flag=='lm_ft':
        n_samples=len(texts)
        #split the dataset
        split_and_save_dataset(n_samples,args.train_ratio, args.val_ratio,save_path=args.folder_name+args.dataset_name+'/splited_indices.json')
    train_idx,val_idx,test_idx=load_split_indices(path=args.folder_name+args.dataset_name+'/splited_indices.json')

    train_mask,val_mask,test_mask=indices_to_mask(train_idx, len(texts)),indices_to_mask(val_idx, len(texts)),indices_to_mask(test_idx, len(texts))
    Data_splited=Data(
        train_mask=train_mask.to(device),
        val_mask=val_mask.to(device),
        test_mask=test_mask.to(device),
        x_feature=x_feature.to(device),
        edge_index=edge_index.to(device),
        true_labels=true_labels.to(device),
        texts=texts
    )
    return Data_splited

def split_and_save_dataset(n_samples, train_ratio,val_ratio,save_path):
    
    indices = np.arange(n_samples)
    train_indices, temp_indices = train_test_split(
        indices, 
        train_size=train_ratio,
        random_state=42
    )
    val_size = val_ratio / (1 - train_ratio)
    # Split temp into validation and test
    val_indices, test_indices = train_test_split(
        temp_indices,
        train_size=val_size,
        random_state=42
    )
    
    split_indices = {
        'train': train_indices.tolist(),
        'val': val_indices.tolist(),
        'test': test_indices.tolist()
    }
    
    with open(save_path, 'w') as f:
        json.dump(split_indices, f)
    return 

def load_split_indices(path):
   
    with open(path, 'r') as f:
        split_indices = json.load(f)
    
    train_indices = torch.tensor(split_indices['train'])
    val_indices = torch.tensor(split_indices['val'])
    test_indices = torch.tensor(split_indices['test'])
    return train_indices, val_indices, test_indices

#read data for ogbn_products(subset)
def read_data_products(device):
    text_list=[]
    load_data_path='./TAG_data/ogbn_products(subset)/ogbn-products_subset.pt'
    with open('./TAG_data/ogbn_products(subset)/ogbn-products_subset.csv','r') as file:
        f=csv.reader(file)
        for line in f:
            text_list.append(line[-1]+'.'+line[-2])
    data= torch.load(load_data_path)
    
    x_feature=data['x']
    edge_index=data.node_stores[0].edge_index
    true_labels=data.y.squeeze(1)
    train_mask,val_mask,test_mask=data.node_stores[0]._mapping['train_mask'],\
                                    data.node_stores[0]._mapping['val_mask'],\
                                    data.node_stores[0]._mapping['test_mask']
    texts=text_list[1:]
    Data_splited=Data(
        train_mask=train_mask.to(device),
        val_mask=val_mask.to(device),
        test_mask=test_mask.to(device),
        x_feature=x_feature.to(device),
        edge_index=edge_index.to(device),
        true_labels=true_labels.to(device),
        texts=texts
    )
    return Data_splited

#read data for arxiv-2023
def read_data_arxiv23(device):
    text_list=[]
    load_data_path='./TAG_data/arxiv-2023/graph.pt'
    with open('./TAG_data/arxiv-2023/paper_info.csv','r') as file:
        f=csv.reader(file)
        for line in f:
            text_list.append(line[1]+'. '+line[2])
    data= torch.load(load_data_path)
    
    x_feature=data.x
    edge_index=data.edge_index
    true_labels=data.y
    train_mask,val_mask,test_mask=data.node_stores[0]._mapping['train_mask'],\
                                    data.node_stores[0]._mapping['val_mask'],\
                                    data.node_stores[0]._mapping['test_mask']
    texts=text_list[1:]
    Data_splited=Data(
        train_mask=train_mask.to(device),
        val_mask=val_mask.to(device),
        test_mask=test_mask.to(device),
        x_feature=x_feature.to(device),
        edge_index=edge_index.to(device),
        true_labels=true_labels.to(device),
        texts=texts
    )
    return Data_splited

#read data for wikics
def read_data_wikics(device):
    data= torch.load('./TAG_data/wiki-cs/wikics_fixed_sbert.pt')
    x_feature=data.x
    edge_index=data.edge_index
    true_labels=data.y
    texts=data.node_stores[0]._mapping['raw_texts']

    train_mask,val_mask,test_mask=data.node_stores[0]._mapping['train_mask'].T[0].T,\
                                    data.node_stores[0]._mapping['val_mask'].T[0].T,\
                                    data.node_stores[0]._mapping['test_mask']
    print(train_mask.sum().item(),val_mask.sum().item(),test_mask.sum().item(),x_feature.shape[0])
    Data_splited=Data(
        train_mask=train_mask.to(device),
        val_mask=val_mask.to(device),
        test_mask=test_mask.to(device),
        x_feature=x_feature.to(device),
        edge_index=edge_index.to(device),
        true_labels=true_labels.to(device),
        texts=texts
    )
    return Data_splited

#read data for Ele-Photo
def read_data_photo(device):
    data= torch.load('./TAG_data/photo/photo.pt')
    x_feature=data.x
    edge_index=data.edge_index
    true_labels=data.y
    texts=data.node_stores[0]._mapping['raw_texts']

    train_mask,val_mask,test_mask=data.node_stores[0]._mapping['train_mask'],\
                                    data.node_stores[0]._mapping['val_mask'],\
                                    data.node_stores[0]._mapping['test_mask']
    Data_splited=Data(
        train_mask=train_mask.to(device),
        val_mask=val_mask.to(device),
        test_mask=test_mask.to(device),
        x_feature=x_feature.to(device),
        edge_index=edge_index.to(device),
        true_labels=true_labels.to(device),
        texts=texts
    )
    return Data_splited

#read data for ogbn_arxiv
def read_data_ogbn_arxiv(device):

    with open('./TAG_data/ogbn_arxiv/edge.json','r')  as file:
        edge_index=json.load(file)
        edge_index=[torch.tensor(edge) for edge in edge_index]
        edge_index=torch.stack(edge_index).T
        maxl,minr=torch.max(edge_index[0]),torch.min(edge_index[0])
        max_r,min_r=torch.max(edge_index[1]),torch.min(edge_index[1])
    with open('./TAG_data/ogbn_arxiv/data.json','r')  as file:
        data=json.load(file)
    
    texts,true_labels,train_idx,val_idx,test_idx=[],[],[],[],[]
    flag=0
    for data_i in data:
        texts.append(data_i['text'])
        true_labels.append(data_i['label'])
        
        if data_i['type']=='train':
            train_idx.append(data_i['id'])
        elif data_i['type']=='val':
            val_idx.append(data_i['id'])
        else:
            test_idx.append(data_i['id'])
        flag+=1
    true_labels=torch.tensor(true_labels)
    train_idx,val_idx,test_idx=torch.tensor(train_idx),torch.tensor(val_idx),torch.tensor(test_idx)
    train_mask,val_mask,test_mask=indices_to_mask(train_idx, len(texts)),indices_to_mask(val_idx, len(texts)),indices_to_mask(test_idx, len(texts))
    Data_splited=Data(
        train_mask=train_mask.to(device),
        val_mask=val_mask.to(device),
        test_mask=test_mask.to(device),
        x_feature=[],
        edge_index=edge_index.to(device),
        true_labels=true_labels.to(device),
        texts=texts
    )
    return Data_splited
