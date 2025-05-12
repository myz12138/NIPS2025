import torch
from utils import args
from lm_ft import ft_Model
from my_em_train import EMTrainer
from transformers import AutoTokenizer, AutoModel, AdamW
from data_pro import read_data_products,read_data_arxiv23,read_data,read_data_wikics,read_data_photo,read_data_ogbn_arxiv
from my_em_models import device

#transfer node_id into mask
def indices_to_mask(indices, num_nodes):
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[indices] = True
    return mask
def main():
    # Initialize tokenizer, lm_model and config
    tokenizer = AutoTokenizer.from_pretrained(args.lm_file_path)
    lm_model = AutoModel.from_pretrained(args.lm_file_path)
    config = {
        'lm_dim':args.lm_dim,
        'e_hidden_size': args.e_hidden_size,
        'm_hidden_size': args.m_hidden_size,
        'num_classes':  args.num_classes,
        'batch_size':args.batch_size,
        'e1_lr': args.e1_lr,
        'e2_lr': args.e2_lr,
        'm1_lr': args.m1_lr,
        'm2_lr': args.m2_lr,
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
        'gnn_name':args.gnn_name,
        'dataset_name':args.dataset_name,
        'folder_name':args.folder_name
    }

    Lmft_Model=ft_Model(lm_model,hidden_dim=config['lm_dim'],labels_dim=config['num_classes']).to(device)#load fine-tuned LM
    if args.dataset_name in ['cora','citeseer']:
        Data_splited=read_data(args=args,device=device)
    elif args.dataset_name=='arxiv-2023':
        Data_splited=read_data_arxiv23(device=device)
    elif args.dataset_name=='ogbn_products(subset)':
        Data_splited=read_data_products(device=device)
    elif  args.dataset_name=='wiki-cs':
        Data_splited=read_data_wikics(device=device)
    elif  args.dataset_name=='photo':
        Data_splited=read_data_photo(device=device)
    elif args.dataset_name=='ogbn_arxiv':
        Data_splited=read_data_ogbn_arxiv(device=device)
    else:
        print('error')
        
    Lmft_Model=ft_Model(lm_model,hidden_dim=config['lm_dim'],labels_dim=config['num_classes']).to(device)
    load_ft_model_path=args.folder_name+args.dataset_name+'/'+'best_'+args.lm_name+'_model_'+args.dataset_name+'.pth'
    Lmft_Model.load_state_dict(torch.load(load_ft_model_path,map_location=device))
    Lmft_Model_trans=Lmft_Model.lm_model
    trainer = EMTrainer(config,tokenizer,Lmft_Model_trans)
    print(load_ft_model_path)
    print(f'e2_loss_Lweight:{args.e2_loss_Lweigh},m2_loss_Lweight:{args.m2_loss_Lweigh}')
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
    )

if __name__=="__main__":
    main()
