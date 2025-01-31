import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch.utils.data import DataLoader, Dataset
import torch
from utils import args
from transformers import AutoTokenizer, AutoModel, AdamW
from data_pro import get_optimized_texts,read_data_ogbn_arxiv,read_data_products,read_data_arxiv23,read_data
from torch.utils.data import DataLoader as TorchDataLoader
import time
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import accuracy_score
import numpy as np
from data_pro import encode_texts
from torch_geometric.utils import subgraph,k_hop_subgraph
from my_em_models import  TextDataset
class TextGCNModel(nn.Module):
    def __init__(self, num_classes, roberta_model_name="./roberta-large", gcn_hidden_dim=256):
        super(TextGCNModel, self).__init__()
        # Load RoBERTa model
        self.roberta = AutoModel.from_pretrained(roberta_model_name)
        self.roberta_hidden_dim = self.roberta.config.hidden_size
        self.mlp=nn.Linear(1024,256)
        # Define GCN layers
        self.gcn1 = GCNConv(256, gcn_hidden_dim)
        self.gcn2 = GCNConv(gcn_hidden_dim, num_classes)

    def forward(self, input_ids, attention_mask, edge_index):
        # Step 1: Generate embeddings using RoBERTa
        roberta_output = self.roberta(**{'input_ids':input_ids,"attention_mask":attention_mask}).last_hidden_state
        roberta_output=torch.mean(roberta_output,dim=-2)
        node_embeddings =F.relu(self.mlp(roberta_output) )  # Use [CLS] token embeddings

        # Step 2: Pass embeddings through GCN layers
        x = self.gcn1(node_embeddings, edge_index)
        x = F.relu(x)
        logits = self.gcn2(x, edge_index)

        return x, node_embeddings,logits  # Return logits and intermediate embeddings


class TextGraphDataset(Dataset):
    def __init__(self, texts, labels, masks):
        self.texts = texts
        self.labels = labels
        self.masks = masks
        self.tokenizer = AutoTokenizer.from_pretrained("./roberta-large")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Tokenize text
        encoded = self.tokenizer(self.texts[idx], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        label = self.labels[idx]
        mask = self.masks[idx]
        return input_ids, attention_mask, label, mask

def create_dataloader(texts, labels, masks, batch_size):
    dataset = TextGraphDataset(texts, labels, masks)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_model(model,  dataloader, edge_index, train_mask, test_mask, labels, device, epochs=10, lr=1e-4):
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    loss_func=nn.CrossEntropyLoss()
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        for step,batch in  enumerate(dataloader):
            batch_ids,batch_token_mask,batch_idx = batch['ids'].to(device),batch['mask'].to(device),batch['idx'].to(device)
            batch_train_mask=train_mask[batch_idx]
            optimizer.zero_grad()
            sub_edge_index,_ =subgraph(batch_idx,edge_index[0], relabel_nodes=True)
            # Forward pass
            _1, _2,logits = model(batch_ids, batch_token_mask, sub_edge_index)
            #print(logits[batch_train_mask].shape,labels[batch_idx ][batch_train_mask].shape)
            loss = loss_func(logits[batch_train_mask], labels[batch_idx ][batch_train_mask])  # Use batch-specific mask
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if step%50==0:
                print(step)
        # # Validation
        # model.eval()
        # val_logits_list = []
        # val_labels_list = []
        # with torch.no_grad():
        #     for step,batch in  enumerate(dataloader):
        #         batch_ids,batch_token_mask,batch_idx = batch['ids'].to(device),batch['mask'].to(device),batch['idx']
        #         batch_val_mask=val_mask[batch_idx]
        #         optimizer.zero_grad()

        #         # Forward pass
        #         logits, _ = model(batch_ids, batch_token_mask, edge_index)
                

        # # Concatenate all validation logits and labels
        # val_logits = torch.cat(val_logits_list, dim=0)
        # val_labels = torch.cat(val_labels_list, dim=0)
        # val_preds = val_logits.argmax(dim=1)
        # val_acc = accuracy_score(val_labels.cpu(), val_preds.cpu())

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss/train_mask.sum().item():.4f}")
        test_model(model, dataloader, edge_index, test_mask, labels, device)
def test_model(model, dataloader, edge_index, test_mask, labels, device):
    model = model.to(device)
    model.eval()
    roberta_embeddings,g_embeddings=torch.zeros(test_mask.shape[0],256).to(device),torch.zeros(test_mask.shape[0],256).to(device)
    with torch.no_grad():
        for step,batch in  enumerate(dataloader):
            batch_ids,batch_token_mask,batch_idx = batch['ids'].to(device),batch['mask'].to(device),batch['idx'].to(device)
            batch_test_mask=test_mask[batch_idx]
            sub_edge_index,_ =subgraph(batch_idx,edge_index[0], relabel_nodes=True)
            # Forward pass
            g_embeddings_batch, roberta_embeddings_batch,logits_batch = model(batch_ids, batch_token_mask,sub_edge_index)
            roberta_embeddings[batch_idx],g_embeddings[batch_idx]=roberta_embeddings_batch,g_embeddings_batch
            if test_mask[batch_idx].sum().item()>0:
                
                test_preds = logits_batch[batch_test_mask].argmax(dim=1)
                test_labels = labels[batch_idx][batch_test_mask]
                test_acc = accuracy_score(test_labels.cpu(), test_preds.cpu())
            if step%50==0:
                print(step)
            # Save embeddings
        np.save("./TAG_data/cora/embs_e_cora_simTAG.npy", roberta_embeddings.cpu().numpy())
        np.save("./TAG_data/cora/embs_m_cora_simTAG.npy", g_embeddings.cpu().numpy())

    print(f"Test Accuracy: {test_acc:.4f}")


def main():
    
    #print(torch.initial_seed())
    # 初始化tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained(args.lm_file_path)
    lm_model = AutoModel.from_pretrained(args.lm_file_path)
    config = {
        'lm_dim':args.lm_dim,
        'e_hidden_size': args.e_hidden_size,
        'm_hidden_size': args.m_hidden_size,
        'num_classes':  args.num_classes,
        'batch_size':args.batch_size,
        'e1_lr': args.e1_lr,#e步求高斯分布的（lm+mlp),对应kl_loss
        'e2_lr': args.e2_lr,#e步求分类结果的（decoder+mlp),对应分类loss
        'm1_lr': args.m1_lr,#m步求高斯分布的（GCN+mlp),对应loss2+loss3
        'm2_lr': args.m2_lr,#e步求高斯分布的（decoder+GCN),对应分类loss
        'e_dropout':args.e_dropout,
        'm_dropout':args.m_dropout,
        "epoch_all":args.epoch_all,
        "initial_dim":args.initial_dim,
        "latent_dim":args.latent_dim,
        "early_epoch_e":args.early_epoch_e,
        "early_epoch_m":args.early_epoch_m,
        "epoch_e_class":args.epoch_e_class,
        "epoch_m_class":args.epoch_m_class,
        'e2_loss_Lweight':args.e2_loss_Lweigh,
        'm2_loss_Lweight':args.m2_loss_Lweigh,
        'e1_model_path':args.folder_name+args.dataset_name+'/best_my_e1_model_'+args.lm_name+'.pth',
        'e2_model_path':args.folder_name+args.dataset_name+'/best_my_e2_model_'+args.lm_name+'.pth',
        'm1_model_path':args.folder_name+args.dataset_name+'/best_my_m1_model_'+args.lm_name+'.pth',
        'm2_model_path':args.folder_name+args.dataset_name+'/best_my_m2_model_'+args.lm_name+'.pth',
        'sample_time':args.sample_time,
        'gnn_name':args.gnn_name
    }
  
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #Lmft_Model=ft_Model(lm_model,hidden_dim=config['lm_dim'],labels_dim=config['num_classes']).to(device)#已经微调好的lm模型
    if args.dataset_name in ['cora','citeseer','pubmed']:
        Data_splited=read_data(args=args,device=device)
    elif args.dataset_name=='arxiv-2023':
        Data_splited=read_data_arxiv23(device=device)
    elif args.dataset_name=='ogbn_products(subset)':
        Data_splited=read_data_products(device=device)
    elif args.dataset_name=='ogbn_arxiv':
        Data_splited=read_data_ogbn_arxiv(device=device)
    else:
        print('error')
        
    #Lmft_Model=ft_Model(lm_model,hidden_dim=config['lm_dim'],labels_dim=config['num_classes']).to(device)
    #load_ft_model_path=args.folder_name+args.dataset_name+'/'+'best_'+args.lm_name+'_model_'+args.dataset_name+'.pth'
    #print(load_ft_model_path)
    #Lmft_Model.load_state_dict(torch.load(load_ft_model_path,map_location=device))
    #Lmft_Model_trans=Lmft_Model.lm_model
    texts=Data_splited.texts,
    edge_index=Data_splited.edge_index,
    train_mask=Data_splited.train_mask,
    val_mask=Data_splited.val_mask,
    test_mask=Data_splited.test_mask,
    labels=Data_splited.true_labels
    all_ids,all_mask=encode_texts(texts[0],device,tokenizer,truncation_length=512)
    dataset = TextDataset(all_ids,all_mask, torch.arange(len(all_ids)))
    dataloader = TorchDataLoader(
            dataset,
            batch_size=16,
            shuffle=True
        )
    batch_size = 16
    num_classes =7
    


    # 初始化模型
    model = TextGCNModel(num_classes=num_classes)

    # 训练模型
    train_model(model,  dataloader, edge_index, Data_splited.train_mask, Data_splited.test_mask, Data_splited.true_labels, device, epochs=10, lr=1e-3)

    # 测试模型
    test_model(model, dataloader, edge_index, Data_splited.test_mask, Data_splited.true_labels, device)
    
    

if __name__=="__main__":
    main()