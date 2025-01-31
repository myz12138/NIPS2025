#only with align, the task target is replaced with mlp and gcn
import torch
import torch.nn as nn
import time
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader as TorchDataLoader
from my_em_models import TextEncoder,E1StepModel,E2StepModel,M1StepModel,M2StepModel,TextDataset,device
from data_pro import get_optimized_texts,read_data_ogbn_arxiv,read_data_products,read_data_arxiv23,read_data
class Ablation2:
    def __init__(self, config,tokenizer,Lmft_Model):
        self.device =device
        
        # 初始化模型
        self.text_encoder = TextEncoder(tokenizer=tokenizer,ft_model=Lmft_Model)
    
        self.e1_model = E1StepModel(
            ft_model=Lmft_Model,
            lm_dim=config['lm_dim'],
            latent_dim=config['latent_dim'],
        ).to(self.device)

        self.e2_model = E2StepModel(
            latent_dim=config['latent_dim'],
            e_hidden_size=config['e_hidden_size'],
            num_classes=config['num_classes'],
            dropout=config['e_dropout']
        ).to(self.device)

        self.m1_model = M1StepModel(
            initial_dim=config['initial_dim'],
            latent_dim=config['latent_dim'],
            dropout=config['m_dropout'],
            gnn_name=config['gnn_name']
        ).to(self.device)

        self.m2_model = M2StepModel(
            latent_dim=config['latent_dim'],
            m_hidden_size=config['m_hidden_size'],
            num_classes=config['num_classes'],
            dropout=config['m_dropout'],
            gnn_name=config['gnn_name']
        ).to(self.device)
        
        
        # 优化器
        self.e1_optimizer = torch.optim.Adam(self.e1_model.parameters(), lr=config['e1_lr'])
        self.e2_optimizer = torch.optim.Adam(self.e2_model.parameters(), lr=config['e2_lr'])
        self.m1_optimizer = torch.optim.Adam(self.m1_model.parameters(), lr=config['m1_lr'])
        self.m2_optimizer = torch.optim.Adam(self.m2_model.parameters(), lr=config['m2_lr'])
        
        
        # 损失函数
        self.criterion_class = nn.CrossEntropyLoss()
        self.criterion_mse=nn.MSELoss()
        
        self.config = config

    @torch.no_grad()
    def sample_pro(self,u,sigma):
        
        distributions = [torch.distributions.Normal(mean, std) for mean, std in zip(u, sigma)]
        samples= torch.stack([dist.sample() for dist in distributions])
        return samples.to(self.device)
    
    def initialize_pseudo_labels(self,x,graph_data):
        #initial pseudo label e by m1 and m2 step
        self.m1_model.eval()
        self.m2_model.eval()
        with torch.no_grad():
            mu,sigma= self.m1_model(
                x,
                graph_data
            )
            sample_embs=torch.zeros(mu.shape[0],self.config['sample_time'],mu.shape[1]).to(self.device)
            for k in range(self.config['sample_time']):
                sample_embs[:,k:k+1,:]=self.sample_pro(mu,sigma).unsqueeze(1)
            sample_embs=torch.mean(sample_embs,dim=-2)
            pseudo_logits= self.m2_model(
                sample_embs,
                graph_data
            )
            pseudo_labels=pseudo_logits.argmax(dim=-1)
        return mu,sigma,pseudo_labels
    
    def kl_divergence_gaussian(self,mu0, sigma0, mu1, sigma1):
        kl_div = torch.log(sigma1/sigma0) + (sigma0**2 + (mu0 - mu1)**2) / (2 * sigma1**2) - 0.5
        kl_div=torch.mean(kl_div)
        return kl_div

    def compute_accuracy(self, pred_labels, true_labels, mask):
        correct = (pred_labels[mask] == true_labels[mask]).sum().item()
        total = mask.sum().item()
        return correct / total if total > 0 else 0.0

    def weight_edges(self,embs, edge_tensor):
        with torch.no_grad():
            src_nodes = edge_tensor[0]
            tgt_nodes = edge_tensor[1]
            weights = []
            new_embs1,new_embs2=embs[src_nodes],embs[tgt_nodes]
            weights=F.cosine_similarity(new_embs1,new_embs2,dim=-1)
            weights=torch.clamp(weights,min=0)
        return weights.to(self.device) 
     
    def e1_step(self, e_dataloader,pesudo_mu_e,pesudo_sigma_e):
        avg_loss=0
        t=time.time()
        self.e1_model.train()
        for step,batch in enumerate(e_dataloader):
            batch_ids,batch_token_mask,batch_idx = batch['ids'].to(self.device),batch['mask'].to(self.device),batch['idx']
            batch_pesudo_mu_e,batch_pesudo_sigma_e=pesudo_mu_e[batch_idx],pesudo_sigma_e[batch_idx],#pesudo mu and sigma in a batchs
            mu_e,sigma_e= self.e1_model(batch_ids,batch_token_mask)
            loss=self.criterion_mse(mu_e,batch_pesudo_mu_e.data)+self.criterion_mse(sigma_e,batch_pesudo_sigma_e.data)
            avg_loss+=loss.item()
            self.e1_optimizer.zero_grad()
            loss.backward()
            self.e1_optimizer.step()
            if step%20==0:
                print(f'e_step:{step},e_kl_loss:{loss},time:{time.time()-t}')
        avg_loss=avg_loss/pesudo_mu_e.shape[0]
        best_e1_model_state =self.e1_model.state_dict()
        torch.save(best_e1_model_state, self.config['e1_model_path'])
        
        print(f'finished e1_train step,e1_loss:{avg_loss}')
        self.e1_model.eval()
        with torch.no_grad():
            pesudo_mu_m,pesudo_sigma_m=torch.zeros_like(pesudo_mu_e).to(device),torch.zeros_like(pesudo_sigma_e).to(device)
            for step,batch in enumerate(e_dataloader):
                batch_ids,batch_token_mask,batch_idx = batch['ids'].to(self.device),batch['mask'].to(self.device),batch['idx']
                pesudo_mu_m[batch_idx],pesudo_sigma_m[batch_idx]=self.e1_model(batch_ids,batch_token_mask)
                if step%500==0:
                    print(f'step:{step}')
            print(f'finished sample e2 input')
            sample_embs=torch.zeros(pesudo_mu_m.shape[0],self.config['sample_time'],pesudo_mu_m.shape[1]).to(device)
            for k in range(self.config['sample_time']):
                sample_embs[:,k:k+1,:]=self.sample_pro(pesudo_mu_m,pesudo_sigma_m).unsqueeze(1)
            sample_embs=torch.mean(sample_embs,dim=-2)
        return pesudo_mu_m,pesudo_sigma_m,sample_embs.detach()
    
    def e2_step(self,train_mask, val_mask,test_mask, true_labels,sample_embs):
        self.e2_model.train()
        true_labels=true_labels.to(self.device)
        
        logits= self.e2_model(sample_embs.data)
        loss = self.criterion_class(logits[train_mask], true_labels[train_mask])
        
        self.e2_optimizer.zero_grad()
        loss.backward()
        self.e2_optimizer.step()
        
        e_predictions = logits.argmax(dim=-1)            
        e_test_acc,e_val_acc = self.compute_accuracy(e_predictions, true_labels, test_mask),\
                                self.compute_accuracy(e_predictions, true_labels, val_mask)
        
        print(f'e2_step:  train_loss:{loss.item()},e_val:{e_val_acc},e_test: {e_test_acc}')
            
        self.e2_model.eval()
        return e_test_acc,e_val_acc

    def m1_step(self, x,graph_data,pesudo_mu_m,pesudo_sigma_m):
        self.m1_model.train()
        mu_m,sigma_m = self.m1_model(
            x,
            graph_data
        )
        loss=self.criterion_mse(sigma_m,pesudo_sigma_m.data)+self.criterion_mse(mu_m,pesudo_mu_m.data)
        self.m1_optimizer.zero_grad()
        loss.backward()
        self.m1_optimizer.step()
        
        best_m1_model_state =self.m1_model.state_dict()
        torch.save(best_m1_model_state,self.config['m1_model_path'])
        print(f'm1_loss: train_m1_loss:{loss.item()}')
        self.m1_model.eval()
        with torch.no_grad():
            pesudo_mu_e,pesudo_sigma_e=self.m1_model(x,graph_data)
            sample_embs=torch.zeros(pesudo_mu_m.shape[0],self.config['sample_time'],pesudo_mu_m.shape[1]).to(device)
            for k in range(self.config['sample_time']):
                sample_embs[:,k:k+1,:]=self.sample_pro(pesudo_mu_e,pesudo_sigma_e).unsqueeze(1)
            sample_embs=torch.mean(sample_embs,dim=-2)
        return pesudo_mu_e,pesudo_sigma_e,sample_embs.detach()
    
    def m2_step(self,graph_data, train_mask, val_mask,test_mask, true_labels,sample_embs):
        
        self.m2_model.train()
        true_labels=true_labels.to(self.device)
        
        logits=self.m2_model(
            sample_embs,
            graph_data,
        )
        loss = self.criterion_class(logits[train_mask], true_labels[train_mask])
        
        self.m2_optimizer.zero_grad()
        loss.backward()
        self.m2_optimizer.step()
        m_predictions=logits.argmax(dim=-1)
        m_test_acc,m_val_acc = self.compute_accuracy(m_predictions, true_labels, test_mask),\
                                self.compute_accuracy(m_predictions, true_labels, val_mask)
        print(f'm2_test: {m_test_acc},m2_val:{m_val_acc},train_m2_loss:{loss}')
            
        self.m2_model.eval()
        return  m_test_acc,m_val_acc
    
    def train(self, x_feature,texts, graph_data, train_mask,val_mask, test_mask, true_labels,flag):
        #initial_embeddings=torch.rand(len(texts),self.config['lm_dim'])
        m_initial_embeddings,all_ids,all_mask= self.text_encoder.encode(texts)
        m_initial_embeddings,graph_data=m_initial_embeddings.to(self.device),graph_data.to(self.device)
        dataset = TextDataset(all_ids,all_mask, torch.arange(len(all_ids)))
        dataloader = TorchDataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=True
        )
        best_test=0
        pesudo_mu_e,pesudo_sigma_e, m_predictions = self.initialize_pseudo_labels(m_initial_embeddings,graph_data)
        for epoch in range(self.config['epoch_all']):
            print(epoch)
            if epoch<=self.config['early_epoch_e']:
                pesudo_mu_m,pesudo_sigma_m,sample_embs_e=self.e1_step(
                    dataloader,
                    pesudo_mu_e,
                    pesudo_sigma_e,
                    
                )
            if epoch<=self.config['early_epoch_m']:
                pesudo_mu_e,pesudo_sigma_e,sample_embs_m=self.m1_step(
                    m_initial_embeddings,
                    graph_data,
                    pesudo_mu_m,
                    pesudo_sigma_m)
        np.save('./TAG_data/cora/embs_e_cora_myalign.npy', sample_embs_e.cpu().numpy())
        np.save('./TAG_data/cora/embs_m_cora_myalign.npy', sample_embs_m.cpu().numpy())
        for epoch in range(20):
            if flag=='mlp':
                mlp_test,mlp_val= self.e2_step(
                        train_mask,val_mask, test_mask, true_labels,sample_embs_e
                    ) 
                if mlp_test>best_test:
                    best_test=mlp_test
            elif flag=='gcn':
                graph_test,graph_val= self.m2_step(
                    graph_data,
                    train_mask,
                    val_mask,
                    test_mask,
                    true_labels,
                    sample_embs_m,
                )  
                if graph_test>best_test:
                    best_test=graph_test
            
        
        # np.save('labels_citeseer.npy', true_labels.cpu().numpy())
        # np.save('test_mask_citeseer.npy', test_mask.cpu().numpy())
        # np.save('embs_initial_citeseer.npy', m_initial_embeddings.cpu().numpy())
        
        #relative_entropy_values=compute_node_relative_entropy_test(sample_embs_e,sample_embs_m,graph_data,test_mask,true_labels)
        print(f'best_test:{best_test}')
            
            
def main():
    from transformers import AutoTokenizer, AutoModel, AdamW
    from utils import args
    from lm_ft import ft_Model
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
  
    
    Lmft_Model=ft_Model(lm_model,hidden_dim=config['lm_dim'],labels_dim=config['num_classes']).to(device)#已经微调好的lm模型
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
        
    Lmft_Model=ft_Model(lm_model,hidden_dim=config['lm_dim'],labels_dim=config['num_classes']).to(device)
    load_ft_model_path=args.folder_name+args.dataset_name+'/'+'best_'+args.lm_name+'_model_'+args.dataset_name+'.pth'
    print(load_ft_model_path)
    Lmft_Model.load_state_dict(torch.load(load_ft_model_path,map_location=device))
    Lmft_Model_trans=Lmft_Model.lm_model
    trainer = Ablation2(config,tokenizer,Lmft_Model_trans)

    print(f'data_name:{args.dataset_name},batch_size:{args.batch_size},start train!!')
    print(f'e1_lr:{args.e1_lr},e2_lr:{args.e2_lr},m1_lr:{args.m1_lr},m2_lr:{args.m2_lr},epoch_e_early:{args.early_epoch_e},epoch_m_early:{args.early_epoch_m}')
    trainer.train(
        x_feature=Data_splited.x_feature,
        texts=Data_splited.texts,
        graph_data=Data_splited.edge_index,
        train_mask=Data_splited.train_mask,
        val_mask=Data_splited.val_mask,
        test_mask=Data_splited.test_mask,
        true_labels=Data_splited.true_labels,
        flag='mlp'
    )

if __name__=="__main__":
    main()      
            