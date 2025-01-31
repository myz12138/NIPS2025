import sys
import os

# 获取 src 的父目录路径
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(parent_dir)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from torch_geometric.nn import GCNConv
from torch.utils.data import DataLoader, Dataset
from data_pro import read_data
from utils import args
from lm_ft import ft_Model
from my_em_models import device
# 数据集定义
class TextGraphDataset(Dataset):
    def __init__(self, text_ids,texts_mask, labels,train_mask, val_mask, test_mask,x_feature):
        self.ids=text_ids,
        self.mask=texts_mask
        self.labels = labels
        
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.x_feature=x_feature

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'id':torch.tensor([i for i in range(len(self.labels))])[idx],
            'ids': self.ids[0][idx],
            'mask': self.mask[idx],
            'label': self.labels[idx],
            'train_mask': self.train_mask[idx],
            'val_mask': self.val_mask[idx],
            'test_mask': self.test_mask[idx],
            'node_feature':self.x_feature[idx]
        }

# 模型定义
class TextGraphModel(nn.Module):
    def __init__(self, lm_trans_model,gcn_input_dim, gcn_hidden_dim, gcn_output_dim, dropout=0.5):
        super(TextGraphModel, self).__init__()
        # 微调的语言模型
        self.lm = lm_trans_model
        self.mlp_lm=nn.Linear(1024,256)
        self.gcn = GCNConv(gcn_input_dim, gcn_hidden_dim)
        self.gcn_out = GCNConv(gcn_hidden_dim, gcn_output_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = gcn_output_dim

    def forward(self,ids,mask, edge_index, node_features):
        # 语言模型嵌入
        lm_outputs = self.lm(**{'input_ids':ids,"attention_mask":mask})
        lm_embeddings = lm_outputs.last_hidden_state[:, 0, :]  # 取 [CLS] token 的嵌入
        lm_embeddings = self.dropout(lm_embeddings)
        lm_embeddings=self.mlp_lm(lm_embeddings)

        # 图学习嵌入
        gcn_hidden = F.relu(self.gcn(node_features, edge_index))
        gcn_hidden = self.dropout(gcn_hidden)
        gcn_embeddings = self.gcn_out(gcn_hidden, edge_index)

        return lm_embeddings, gcn_embeddings

class TextclassModel(nn.Module):
    def __init__(self, TextGraphModel):
        super(TextclassModel, self).__init__()
        # 微调的语言模型
        self.lm =TextGraphModel.lm
        self.mlp1=TextGraphModel.mlp_lm
        self.lm.eval();self.mlp1.eval()
        self.classfi=nn.Linear(256,6)

    def forward(self,ids,mask):
        # 语言模型嵌入
        lm_outputs = self.lm(**{'input_ids':ids,"attention_mask":mask})
        lm_embeddings = lm_outputs.last_hidden_state[:, 0, :]  # 取 [CLS] token 的嵌入
        lm_embeddings=self.mlp1(lm_embeddings)
        logits=self.classfi(F.relu(lm_embeddings))

        return lm_embeddings, logits

class GraphclassModel(nn.Module):
    def __init__(self, TextGraphModel):
        super(GraphclassModel, self).__init__()
        self.gnn1 =TextGraphModel.gcn
        self.gnn2=TextGraphModel.gcn_out
        self.gnn1.eval();self.gnn2.eval()
        self.classfi=nn.Linear(256,6)

    def forward(self,node_features, edge_index):
       
        gcn_hidden = F.relu(self.gnn1(node_features, edge_index))
        gcn_embeddings = self.gnn2(gcn_hidden, edge_index)
        logits=self.classfi(F.relu(gcn_embeddings))
        return gcn_embeddings, logits
# 相似度矩阵计算
def compute_similarity_matrices(lm_embeddings, gcn_embeddings):
    # 计算 C
    C = torch.matmul(lm_embeddings, gcn_embeddings.T)

    # 计算 S_T
    S_T = torch.matmul(lm_embeddings, lm_embeddings.T)
    S_T = F.normalize(S_T, p=1, dim=1)  # 行归一化

    # 计算 S_G
    S_G = torch.matmul(gcn_embeddings, gcn_embeddings.T)
    S_G = F.normalize(S_G, p=1, dim=0)  # 列归一化

    return C, S_T, S_G

# 损失函数
def compute_loss(C, S_T, S_G, k):
    # 对 C 和 S_T 的行计算交叉熵损失
    C1=F.normalize(C, p=1, dim=1) 
    row_loss = F.cross_entropy(C1, S_T, reduction='none').sum()

    # 对 C 和 S_G 的列计算交叉熵损失
    C2=F.normalize(C.T, p=1, dim=1) 
    col_loss = F.cross_entropy(C2, S_G.T, reduction='none').sum()

    # 总损失
    loss = (row_loss + col_loss) / (2 * k)
    return loss

# 数据处理
def preprocess_data(text_list, tokenizer):
    # 对文本进行分词
    text_inputs = tokenizer(text_list, return_tensors='pt',  truncation=True,max_length=512,padding='max_length').to(device)
    
    return text_inputs['input_ids'],text_inputs['attention_mask']

def compute_accuracy( pred_labels, true_labels):
    correct = (pred_labels == true_labels).sum().item()
    
    return correct 
# 训练、验证、测试步骤
def train(model, edge_index,node_features,data_loader, optimizer, device):
    loss_f=nn.MSELoss()
    model.train()
    total_loss = 0
    for step,batch in enumerate(data_loader):
        text_ids,text_mask= batch['ids'], batch['mask']
        train_mask = batch['train_mask']
        k = train_mask.sum().item()
        id=batch['id']
        # 将数据移动到设备
        
        edge_index = edge_index.to(device)
        node_features = node_features.to(device)

        # 前向传播
        lm_embeddings, gcn_embeddings = model(text_ids,text_mask, edge_index,node_features)

        # # 计算相似度矩阵
        # C, S_T, S_G = compute_similarity_matrices(lm_embeddings[train_mask], gcn_embeddings[id][train_mask])

        # # 计算损失
        # loss = compute_loss(C, S_T, S_G, k)

        # 反向传播和优化
        loss=loss_f(lm_embeddings[train_mask], gcn_embeddings[id][train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if step%30==0:
            print(f" step:{step},Loss: {loss}")
    print(f"Training Loss: {total_loss / len(data_loader)}")

def evaluate(model, test_mask,edge_index,node_features,data_loader, device, mask_type):
    loss_f=nn.MSELoss()
    model.eval()
    total_loss = 0
    embs_text=torch.zeros(node_features.shape[0],256).to(device)
    embs_graph=torch.zeros(node_features.shape[0],256).to(device)
    with torch.no_grad():
        for batch in data_loader:
            text_ids,text_mask= batch['ids'], batch['mask']
            test_mask = batch['test_mask']

            k = test_mask.sum().item()
            id=batch['id']
            # 将数据移动到设备
            lm_embeddings, gcn_embeddings = model(text_ids,text_mask, edge_index,node_features)
            embs_text[id]=lm_embeddings
            embs_graph[id]=gcn_embeddings[id]
            # 计算相似度矩阵
            # C, S_T, S_G = compute_similarity_matrices(lm_embeddings[test_mask], gcn_embeddings[id][test_mask])

            #    计算损失
            # if k!=0:
            #     loss = compute_loss(C, S_T, S_G, k)
            if k!=0:
                loss=loss_f(lm_embeddings[test_mask], gcn_embeddings[id][test_mask])
                
                total_loss += loss.item()

    print(f"{mask_type.capitalize()} Loss: {total_loss / len(data_loader)}")
    return total_loss,embs_text,embs_graph
# 主函数
def main_embs(config):
    # 假设已经有以下数据
    tokenizer = AutoTokenizer.from_pretrained('./roberta-large')
    lm_model = AutoModel.from_pretrained('./roberta-large')
    Lmft_Model=ft_Model(lm_model,hidden_dim=config['lm_dim'],labels_dim=config['num_classes']).to(device)#已经微调好的lm模型
    if args.dataset_name in ['cora','citeseer','pubmed']:
        Data_splited=read_data(args=args,device=device)
    Lmft_Model=ft_Model(lm_model,hidden_dim=config['lm_dim'],labels_dim=config['num_classes']).to(device)
    load_ft_model_path=args.folder_name+args.dataset_name+'/'+'best_'+args.lm_name+'_model_'+args.dataset_name+'.pth'
    print(load_ft_model_path)
    Lmft_Model.load_state_dict(torch.load(load_ft_model_path,map_location=device))
    Lmft_Model_trans=Lmft_Model.lm_model
    text_ids,texts_mask = preprocess_data(
        Data_splited.texts, tokenizer
    )

    dataset = TextGraphDataset(text_ids,texts_mask,Data_splited.true_labels, Data_splited.train_mask, Data_splited.val_mask, Data_splited.test_mask, Data_splited.x_feature)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = TextGraphModel(Lmft_Model_trans, gcn_input_dim=Data_splited.x_feature.shape[1] ,gcn_hidden_dim=512, gcn_output_dim=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # 训练和验证
    loss_val_max=1e8
    for epoch in range(5):
        print(f"Epoch {epoch + 1}")
        train(model,  Data_splited.edge_index,Data_splited.x_feature,data_loader, optimizer, device)
        loss_val,embs_text,embs_graph=evaluate(model, Data_splited.test_mask,Data_splited.edge_index,Data_splited.x_feature,data_loader, device, mask_type="val_mask")
        if loss_val<loss_val_max:
            best_model_state =model.state_dict()
            torch.save(best_model_state,'./TAG_data/cora/msealign_embs_cora.path')
        np.save('./TAG_data/cora/embs_e_cora_msealign.npy', embs_text.cpu().numpy())
        np.save('./TAG_data/cora/embs_m_cora_msealign.npy', embs_graph.cpu().numpy())
    # 测试
    # evaluate(model, data_loader, device, mask_type="test_mask")

def main_class(config):
    if args.dataset_name in ['cora','citeseer','pubmed']:
        Data_splited=read_data(args=args,device=device)
    tokenizer = AutoTokenizer.from_pretrained('./roberta-large')
    lm_model = AutoModel.from_pretrained('./roberta-large')
    Lmft_Model=ft_Model(lm_model,hidden_dim=config['lm_dim'],labels_dim=config['num_classes']).to(device)
    load_ft_model_path=args.folder_name+args.dataset_name+'/'+'best_'+args.lm_name+'_model_'+args.dataset_name+'.pth'
    Lmft_Model.load_state_dict(torch.load(load_ft_model_path,map_location=device))
    Lmft_Model_trans=Lmft_Model.lm_model
    
    text_ids,texts_mask = preprocess_data(
        Data_splited.texts, tokenizer
    )
    dataset = TextGraphDataset(text_ids,texts_mask,Data_splited.true_labels, Data_splited.train_mask, Data_splited.val_mask, Data_splited.test_mask, Data_splited.x_feature)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    text_graph_model = TextGraphModel(Lmft_Model_trans, gcn_input_dim=Data_splited.x_feature.shape[1] ,gcn_hidden_dim=512, gcn_output_dim=256).to(device)
    text_graph_model.load_state_dict(torch.load('./TAG_data/cora/msealign_embs_cora.path',map_location=device))
    Text_model=TextclassModel(text_graph_model).to(device)
    Graph_model=GraphclassModel(text_graph_model).to(device)

    

    Text_model.train()
    Text_model.lm.eval();Text_model.mlp1.eval()
    total_loss = 0
    optimizer_lm = torch.optim.Adam(Text_model.parameters(), lr=1e-5)
    
    for epoch in range(5):
        for step,batch in enumerate(data_loader):
            text_ids,text_mask= batch['ids'], batch['mask']
            train_mask=batch['train_mask']
            label=batch['label']
            lm_embeddings, lm_logits =Text_model(text_ids[train_mask],text_mask[train_mask])
            loss = F.cross_entropy(lm_logits,label[train_mask])
            # 反向传播和优化
            optimizer_lm.zero_grad()
            loss.backward()
            optimizer_lm.step()

            total_loss += loss.item()
            if step%30==0:
                print(f" step:{step},Loss: {loss}")
        preds_lm=0
        print(f"Training Loss: {total_loss / len(data_loader)}")
        with torch.no_grad():
            for batch in data_loader:
                text_ids,text_mask= batch['ids'], batch['mask']
                test_mask=batch['test_mask']
                label=batch['label']
                lm_embeddings, lm_logits =Text_model(text_ids[test_mask],text_mask[test_mask])
                preds_lm+=compute_accuracy(lm_logits.argmax(dim=-1), label[test_mask])
            print(f'test_acc:{preds_lm/Data_splited.test_mask.sum().item()}')
if __name__ == "__main__":
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
    main_embs(config)
    main_class(config)
