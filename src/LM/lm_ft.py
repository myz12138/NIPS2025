
import argparse  
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer, AutoModel, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from data_pro import LmftDataset,encode_texts,read_data,read_data_products,read_data_arxiv23,read_data_wikics,read_data_photo,read_data_ogbn_arxiv
import time
from torch_geometric.data import Data


class ft_Model(nn.Module):
    def __init__(self,lm_model,hidden_dim,labels_dim):
        super(ft_Model, self).__init__()

        self.lm_model = lm_model
        self.classifier = nn.Linear(hidden_dim, labels_dim)  
        
    def forward(self,ids,mask):
        outputs = self.lm_model(**{'input_ids':ids,"attention_mask":mask}).last_hidden_state
        outputs=torch.mean(outputs,dim=-2)
        logits = self.classifier(outputs)
        return outputs,logits

def train(args,model, train_loader, val_loader, epochs, learning_rate):
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    best_accuracy = 0.0
    best_model_state = None

    for epoch in range(epochs):
        t = time.time()
        model.train()
        total_loss = 0
        for train_step, train_batch in enumerate(train_loader):
            train_ids,train_mask,train_labels=train_batch[0].to(device),train_batch[1].to(device),train_batch[2].to(device)
            _,outputs = model(train_ids,train_mask)
            loss = criterion(outputs,train_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if train_step%30==0:
                print(f'train_step {train_step}, train_Loss: {loss:.4f},time: {time.time() - t}s')
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_accuracy = evaluate(model, val_loader,flag='val')
        print(f'Epoch {epoch}, Loss: {avg_loss:.4f}, Validation Accuracy: {val_accuracy:.4f},time: {time.time() - t}s')

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model_state = model.state_dict()
            torch.save(best_model_state, args.folder_name+args.dataset_name+'/'+'best_'+args.lm_name+'_model_'+args.dataset_name+'.pth')


def evaluate(model,val_loader,flag):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for val_step, val_batch in enumerate(val_loader):
            val_ids,val_mask,val_labels=val_batch[0].to(device),val_batch[1].to(device),val_batch[2].to(device)
            _,outputs = model(val_ids,val_mask)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(val_labels.cpu().numpy())
            if val_step%30==0:
                print(f'flag{flag},step{val_step}')
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy


def test(model, test_loader,args):
    model.load_state_dict(torch.load(args.folder_name+args.dataset_name+'/'+'best_'+args.lm_name+'_model_'+args.dataset_name+'.pth'))
    test_accuracy = evaluate(model, test_loader,flag='test')
    print(f'Test Accuracy: {test_accuracy:.4f}')


if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='parser example')

    parser.add_argument('--batch_size', default=64, type=int, help='batch size.')

    parser.add_argument('--cuda_number', default='0', type=str, help='cuda number.')

    parser.add_argument('--dataset_name', default='wikics', type=str, help='name of dataset.')

    parser.add_argument('--lm_file_path', default='./roberta-large', type=str, help='path of lm_model.')

    parser.add_argument('--lm_name', default='roberta', type=str, help='name of lm_model.')

    parser.add_argument('--folder_name', default='./TAG_data/', type=str, help='folder name for dataset.')

    parser.add_argument('--epoch', default=10, type=int, help='epoch of train for fine-tuning pertrained language models.')

    parser.add_argument('--lm_dim', default=1024, type=int, help='hidden-dim of LMs.')

    parser.add_argument('--label_dim', default=10, type=int, help='number of categories.')

    parser.add_argument('--learning_rate', default=1e-5, type=float, help='learning_rate in fine-tuning lm.')

    parser.add_argument('--train_ratio', default=0.6, type=float, help='train ratio in spilting the dataset.')

    parser.add_argument('--val_ratio', default=0.2, type=float, help='val ratio in spilting the dataset, the test ratio is equal to the val ratio.')

    parser.add_argument('--flag', default='lm_ft', type=str, help='whether restart spliting dataset or not.')

    args_ft = parser.parse_args()

    device = torch.device('cuda:'+args_ft.cuda_number if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(args_ft.lm_file_path)
    lm_model = AutoModel.from_pretrained(args_ft.lm_file_path)
    
    if args_ft.dataset_name in ['cora','citeseer']:
        Data_splited=read_data(args=args_ft,device=device)
    elif args_ft.dataset_name=='arxiv-2023':
        Data_splited=read_data_arxiv23(device=device)
    elif args_ft.dataset_name=='ogbn_products(subset)':
        Data_splited=read_data_products(device=device)
    elif args_ft.dataset_name=='wikics':
        Data_splited=read_data_wikics(device=device)
    elif  args_ft.dataset_name=='photo':
        Data_splited=read_data_photo(device=device)
    elif args_ft.dataset_name=='ogbn_arxiv':
        Data_splited=read_data_ogbn_arxiv(device=device)
    else:
        print('error')
    print(f'dataset:{args_ft.dataset_name},lm_name:{args_ft.lm_name}')
    all_ids,all_token_mask=encode_texts(Data_splited.texts,device,tokenizer,truncation_length=512)
    train_dataset =LmftDataset(all_ids[Data_splited.train_mask],all_token_mask[Data_splited.train_mask],Data_splited.true_labels[Data_splited.train_mask])
    val_dataset = LmftDataset(all_ids[Data_splited.val_mask],all_token_mask[Data_splited.val_mask],Data_splited.true_labels[Data_splited.val_mask])
    test_dataset = LmftDataset(all_ids[Data_splited.test_mask],all_token_mask[Data_splited.test_mask],Data_splited.true_labels[Data_splited.test_mask])

    train_loader = DataLoader(train_dataset, batch_size=args_ft.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=100,shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=100,shuffle=False)
    
    Lmft_Model=ft_Model(lm_model,hidden_dim=args_ft.lm_dim,labels_dim=args_ft.label_dim).to(device)

    train(args_ft,Lmft_Model, train_loader, val_loader, epochs=args_ft.epoch, learning_rate=args_ft.learning_rate)
    test(Lmft_Model, test_loader,args_ft)
