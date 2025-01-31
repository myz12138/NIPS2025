import numpy as np
import torch
import matplotlib.pyplot as plt

def kl_divergence(p, q, dim=1):
    """
    Compute KL divergence between two tensors along a specified dimension.
    """
    p = torch.nn.functional.softmax(p, dim=dim)
    q = torch.nn.functional.softmax(q, dim=dim)
    kl = torch.sum(p * (torch.log(p + 1e-10) - torch.log(q + 1e-10)), dim=dim)
    return kl

def get_kl_data(text_features_path, struct_features_path, test_mask_path):
    """
    Load data and compute KL divergence values and means.
    """
    
    text_features = np.load(text_features_path)
    struct_features = np.load(struct_features_path)
    test_mask = np.load(test_mask_path)
    text_features=text_features[test_mask]
    struct_features=struct_features[test_mask]

    text_features = torch.from_numpy(text_features)
    struct_features = torch.from_numpy(struct_features)

    text_text_similarity = torch.mm(text_features, text_features.T)
    struct_struct_similarity = torch.mm(struct_features, struct_features.T)
    text_struct_similarity = torch.mm(text_features, struct_features.T)

    text_text_vs_text_struct_kl = kl_divergence(
        text_text_similarity, text_struct_similarity, dim=1
    ).numpy()
    struct_struct_vs_text_struct_kl = kl_divergence(
        struct_struct_similarity, text_struct_similarity.T, dim=1
    ).numpy()

    mean_tt_vs_ts = np.mean(text_text_vs_text_struct_kl)
    mean_ss_vs_ts = np.mean(struct_struct_vs_text_struct_kl)
    
    return text_text_vs_text_struct_kl, struct_struct_vs_text_struct_kl, mean_tt_vs_ts, mean_ss_vs_ts
# Paths for the datasets
model_names_cite = ['ours','concat',  'GLEM' ]
folder_cite='./TAG_data/citeseer/'
data_name_cite='citeseer'
dataset_paths_cite = [
    {
        'text_features': folder_cite+'embs_e_'+data_name_cite+'_'+model_names_cite[0]+'.npy',
        'struct_features': folder_cite+'embs_m_'+data_name_cite+'_'+model_names_cite[0]+'.npy',
        'categories': folder_cite+'labels_'+data_name_cite+'.npy',
        'test_mask': folder_cite+'test_mask_'+data_name_cite+'.npy'
    },
    {
        'text_features': folder_cite+'embs_e_'+data_name_cite+'_'+model_names_cite[1]+'.npy',
        'struct_features': folder_cite+'embs_m_'+data_name_cite+'_'+model_names_cite[1]+'.npy',
        'categories': folder_cite+'labels_'+data_name_cite+'.npy',
        'test_mask': folder_cite+'test_mask_'+data_name_cite+'.npy'
    },
    
    {
        'text_features': folder_cite+'embs_e_'+data_name_cite+'_'+model_names_cite[2]+'.npy',
        'struct_features': folder_cite+'embs_m_'+data_name_cite+'_'+model_names_cite[2]+'.npy',
        'categories': folder_cite+'labels_'+data_name_cite+'.npy',
        'test_mask': folder_cite+'test_mask_'+data_name_cite+'.npy'
    }
    ]


model_names_cora = ['ours','concat',  'GLEM' ]
folder_cora='./TAG_data/cora/'
data_name_cora='cora'
dataset_paths_cora = [
    {
        'text_features': folder_cora+'embs_e_'+data_name_cora+'_'+model_names_cora[0]+'.npy',
        'struct_features': folder_cora+'embs_m_'+data_name_cora+'_'+model_names_cora[0]+'.npy',
        'categories': folder_cora+'labels_'+data_name_cora+'.npy',
        'test_mask': folder_cora+'test_mask_'+data_name_cora+'.npy'
    },
    {
        'text_features': folder_cora+'embs_e_'+data_name_cora+'_'+model_names_cora[1]+'.npy',
        'struct_features': folder_cora+'embs_m_'+data_name_cora+'_'+model_names_cora[1]+'.npy',
        'categories': folder_cora+'labels_'+data_name_cora+'.npy',
        'test_mask': folder_cora+'test_mask_'+data_name_cora+'.npy'
    },
    
    {
        'text_features': folder_cora+'embs_e_'+data_name_cora+'_'+model_names_cora[2]+'.npy',
        'struct_features': folder_cora+'embs_m_'+data_name_cora+'_'+model_names_cora[2]+'.npy',
        'categories': folder_cora+'labels_'+data_name_cora+'.npy',
        'test_mask': folder_cora+'test_mask_'+data_name_cora+'.npy'
    }
    ]

# Collect data for cora
ts_kl_cora = []
st_kl_cora = []
mean_ts_cora = []
mean_st_cora = []

for paths in dataset_paths_cora:
    text_text_vs_text_struct_kl, struct_struct_vs_text_struct_kl, mean_tt_vs_ts, mean_ss_vs_ts = get_kl_data(
        paths['text_features'], paths['struct_features'], paths['test_mask']
    )
    ts_kl_cora.append(text_text_vs_text_struct_kl)
    st_kl_cora.append(struct_struct_vs_text_struct_kl)
    mean_ts_cora.append(mean_tt_vs_ts)
    mean_st_cora.append(mean_ss_vs_ts)

# Collect data for citeseer
ts_kl_citeseer = []
st_kl_citeseer= []
mean_ts_citeseer= []
mean_st_citeseer= []

for paths in dataset_paths_cite:
    text_text_vs_text_struct_kl, struct_struct_vs_text_struct_kl, mean_tt_vs_ts, mean_ss_vs_ts = get_kl_data(
        paths['text_features'], paths['struct_features'], paths['test_mask']
    )
    ts_kl_citeseer .append(text_text_vs_text_struct_kl)
    st_kl_citeseer.append(struct_struct_vs_text_struct_kl)
    mean_ts_citeseer.append(mean_tt_vs_ts)
    mean_st_citeseer.append(mean_ss_vs_ts)


# Your existing functions and data loading code remain unchanged

# # Plotting
# fig, axes = plt.subplots(1, 4, figsize=(32, 8), dpi=600)

# # First plot: cora
# x = range(len(ts_kl_cora[0]))
# for i, kl_values in enumerate(ts_kl_cora):
#     axes[0].plot(x, kl_values, label=f'{model_names_cora[i]}', color=f'C{i}', linestyle='-', alpha=0.5, linewidth=1.5)
#     axes[0].axhline(y=mean_ts_cora[i], color=f'C{i}', linestyle='-', linewidth=3)

# axes[0].set_title("KL-TS-Cora", fontsize=40)  # Enlarged title font
# axes[0].set_xlabel("Test Samples", fontsize=40)  # Enlarged x-axis label font
# axes[0].set_ylabel("KL-Divergence", fontsize=40)  # Enlarged y-axis label font
# axes[0].tick_params(axis='both', which='major', labelsize=40)  # Enlarged tick labels

# x = range(len(st_kl_cora[0]))
# for i, kl_values in enumerate(st_kl_cora):
#     axes[1].plot(x, kl_values, label=f'{model_names_cora[i]}', color=f'C{i}', linestyle='-', alpha=0.5, linewidth=1.5)
#     axes[1].axhline(y=mean_st_cora[i], color=f'C{i}', linestyle='-', linewidth=3)

# axes[1].set_title("KL-ST-Cora", fontsize=40)
# axes[1].set_xlabel("Test Samples", fontsize=40)
# axes[1].set_ylabel("KL-Divergence", fontsize=40)
# axes[1].tick_params(axis='both', which='major', labelsize=40)

# # Second plot: citeseer
# x = range(len(ts_kl_citeseer[0]))
# for i, kl_values in enumerate(ts_kl_citeseer):
#     axes[2].plot(x, kl_values, label=f'{model_names_cite[i]}', color=f'C{i}', linestyle='-', alpha=0.5, linewidth=1.5)
#     axes[2].axhline(y=mean_ts_citeseer[i], color=f'C{i}', linestyle='-', linewidth=3)

# axes[2].set_title("KL-TS-Citeseer", fontsize=40)
# axes[2].set_xlabel("Test Samples", fontsize=40)
# axes[2].set_ylabel("KL-Divergence", fontsize=40)
# axes[2].tick_params(axis='both', which='major', labelsize=40)

# x = range(len(st_kl_citeseer[0]))
# for i, kl_values in enumerate(st_kl_citeseer):
#     axes[3].plot(x, kl_values, label=f'{model_names_cite[i]}', color=f'C{i}', linestyle='-', alpha=0.5, linewidth=1.5)
#     axes[3].axhline(y=mean_st_citeseer[i], color=f'C{i}', linestyle='-', linewidth=3)

# axes[3].set_title("KL-ST-Citeseer", fontsize=40)
# axes[3].set_xlabel("Test Samples", fontsize=40)
# axes[3].set_ylabel("KL-Divergence", fontsize=40)
# axes[3].tick_params(axis='both', which='major', labelsize=40)

# # Add a single legend below all subplots
# handles, labels = axes[0].get_legend_handles_labels()
# fig.legend(handles, labels, loc='lower center', fontsize=24, ncol=6, bbox_to_anchor=(0.5, -0.1))  # Adjusted legend

# # Adjust layout and save
# plt.tight_layout()
# plt.subplots_adjust(bottom=0.2)  # Leave space for the legend
# plt.savefig('./pic/alignment.png')
# plt.savefig('./pic/alignment.pdf')
# plt.show()

# # Plotting
fig, axes = plt.subplots(1, 4, figsize=(32, 8), dpi=600)

# First plot: cora
x = range(len(ts_kl_cora[0]))
for i, kl_values in enumerate(ts_kl_cora):
    # Plot individual KL divergence values with lighter colors and dashed lines
    axes[0].plot(x, kl_values, label=f'{model_names_cora[i]}', color=f'C{i}', linestyle='-', alpha=0.5, linewidth=1.5)
    # Plot mean KL divergence with darker colors and thicker solid lines
    axes[0].axhline(y=mean_ts_cora[i], color=f'C{i}', linestyle='-', linewidth=3, label=f'{model_names_cora[i]} (Mean)')

axes[0].set_title("KL-TS-cora", fontsize=20)
axes[0].set_xlabel("Test Samples", fontsize=16)
axes[0].set_ylabel("KL Divergence", fontsize=16)
axes[0].legend(loc='best')

x = range(len(st_kl_cora[0]))
for i, kl_values in enumerate(st_kl_cora):
    # Plot individual KL divergence values with lighter colors and dashed lines
    axes[1].plot(x, kl_values, label=f'{model_names_cora[i]}', color=f'C{i}', linestyle='-', alpha=0.5, linewidth=1.5)
    # Plot mean KL divergence with darker colors and thicker solid lines
    axes[1].axhline(y=mean_st_cora[i], color=f'C{i}', linestyle='-', linewidth=3, label=f'{model_names_cora[i]} (Mean)')

axes[1].set_title("KL-ST-cora", fontsize=20)
axes[1].set_xlabel("Test Samples", fontsize=16)
axes[1].set_ylabel("KL Divergence", fontsize=16)
axes[1].legend(loc='best')

# Second plot: citeseer
x = range(len(ts_kl_citeseer[0]))
for i, kl_values in enumerate(ts_kl_citeseer):
    # Plot individual KL divergence values with lighter colors and dashed lines
    axes[2].plot(x, kl_values, label=f'{model_names_cite[i]}', color=f'C{i}', linestyle='-', alpha=0.5, linewidth=1.5)
    # Plot mean KL divergence with darker colors and thicker solid lines
    axes[2].axhline(y=mean_ts_citeseer[i], color=f'C{i}', linestyle='-', linewidth=3, label=f'{model_names_cite[i]} (Mean)')

axes[2].set_title("KL-TS-citeseer", fontsize=20)
axes[2].set_xlabel("Test Samples", fontsize=16)
axes[2].set_ylabel("KL Divergence", fontsize=16)
axes[2].legend(loc='best')

x = range(len(st_kl_citeseer[0]))
for i, kl_values in enumerate(st_kl_citeseer):
    # Plot individual KL divergence values with lighter colors and dashed lines
    axes[3].plot(x, kl_values, label=f'{model_names_cite[i]}', color=f'C{i}', linestyle='-', alpha=0.5, linewidth=1.5)
    # Plot mean KL divergence with darker colors and thicker solid lines
    axes[3].axhline(y=mean_st_citeseer[i], color=f'C{i}', linestyle='-', linewidth=3, label=f'{model_names_cite[i]} (Mean)')

axes[3].set_title("KL-ST-citeseer", fontsize=20)
axes[3].set_xlabel("Samples", fontsize=16)
axes[3].set_ylabel("KL Divergence", fontsize=16)
axes[3].legend(loc='best')

# Adjust layout and save
plt.tight_layout()
plt.savefig('./alignment.png')
plt.savefig('./alignment.pdf')
plt.show()
########################################################################
# import matplotlib.pyplot as plt

# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from data_pro import read_data
# import torch.nn.functional as F
# def row_normalize(matrix):
#     row_sums = matrix.sum(dim=1, keepdim=True)  # Sum along rows
#     return matrix / (row_sums + 1e-8)  # Avoid division by zero


# def column_normalize(matrix):
#     col_sums = matrix.sum(dim=0, keepdim=True)  # Sum along columns
#     return matrix / (col_sums + 1e-8)  # Avoid division by zero

# def kl_divergence(p, q,dim):
# # Add a small constant (1e-8) to avoid log(0) or division by zero
#     p =F.softmax(p) + 1e-8
#     q = F.softmax(q)  + 1e-8
#     return (p * (p.log() - q.log())).mean(dim=dim)

# def get_kl_concat():
#     text_features =np.load('./TAG_data/cora/embs_e_cora_concat.npy') # 示例文本模态特征
#     struct_features =np.load('./TAG_data/cora/embs_m_cora_concat.npy')  # 示例结构模态特
#     test_mask=np.load('./TAG_data/cora/test_mask_cora.npy')
#     text_features=text_features[test_mask]
#     struct_features=struct_features[test_mask]
#     text_features=torch.from_numpy(text_features)
#     struct_features=torch.from_numpy(struct_features)
#     text_text_similarity = torch.mm(text_features, text_features.T)
#     struct_struct_similarity = torch.mm(struct_features, struct_features.T)
#     text_struct_similarity= torch.mm(text_features, struct_features.T)

#     text_text_vs_text_struct_kl =kl_divergence(
#         text_text_similarity, text_struct_similarity,dim=1
#     ).numpy()

#     # Compute relative entropy between struct_struct_col_normalized and text_struct_col_normalized
#     struct_struct_vs_text_struct_kl =kl_divergence(
#         struct_struct_similarity ,text_struct_similarity.T,dim=1
#     ).numpy()
#     mean_tt_vs_ts,mean_ss_vs_ts=np.mean(text_text_vs_text_struct_kl),np.mean(struct_struct_vs_text_struct_kl)
#     return text_text_vs_text_struct_kl,struct_struct_vs_text_struct_kl,mean_tt_vs_ts,mean_ss_vs_ts


# def get_kl_simTAG():
#     text_features =np.load('./TAG_data/cora/embs_e_cora_simTAG.npy') # 示例文本模态特征
#     struct_features =np.load('./TAG_data/cora/embs_m_cora_simTAG.npy')  # 示例结构模态特
#     test_mask=np.load('./TAG_data/cora/test_mask_cora.npy')
#     text_features=text_features[test_mask]
#     struct_features=struct_features[test_mask]
#     text_features=torch.from_numpy(text_features)
#     struct_features=torch.from_numpy(struct_features)
#     text_text_similarity = torch.mm(text_features, text_features.T)
#     struct_struct_similarity = torch.mm(struct_features, struct_features.T)
#     text_struct_similarity= torch.mm(text_features, struct_features.T)

#     text_text_vs_text_struct_kl =kl_divergence(
#         text_text_similarity, text_struct_similarity,dim=1
#     ).numpy()

#     # Compute relative entropy between struct_struct_col_normalized and text_struct_col_normalized
#     struct_struct_vs_text_struct_kl =kl_divergence(
#         struct_struct_similarity ,text_struct_similarity.T,dim=1
#     ).numpy()
#     mean_tt_vs_ts,mean_ss_vs_ts=np.mean(text_text_vs_text_struct_kl),np.mean(struct_struct_vs_text_struct_kl)
#     return text_text_vs_text_struct_kl,struct_struct_vs_text_struct_kl,mean_tt_vs_ts,mean_ss_vs_ts

# def get_kl_ConGraT():
#     text_features =np.load('./TAG_data/cora/embs_e_cora_ConGraT.npy') # 示例文本模态特征
#     struct_features =np.load('./TAG_data/cora/embs_m_cora_ConGraT.npy')  # 示例结构模态特
#     text_features=text_features#[test_mask]
#     struct_features=struct_features#[test_mask]
#     text_features=torch.from_numpy(text_features)
#     struct_features=torch.from_numpy(struct_features)
#     text_text_similarity = torch.mm(text_features, text_features.T)
#     struct_struct_similarity = torch.mm(struct_features, struct_features.T)
#     text_struct_similarity= torch.mm(text_features, struct_features.T)

#     text_text_vs_text_struct_kl =kl_divergence(
#         text_text_similarity, text_struct_similarity,dim=1
#     ).numpy()

#     # Compute relative entropy between struct_struct_col_normalized and text_struct_col_normalized
#     struct_struct_vs_text_struct_kl =kl_divergence(
#         struct_struct_similarity ,text_struct_similarity.T,dim=1
#     ).numpy()
#     mean_tt_vs_ts,mean_ss_vs_ts=np.mean(text_text_vs_text_struct_kl),np.mean(struct_struct_vs_text_struct_kl)
#     return text_text_vs_text_struct_kl,struct_struct_vs_text_struct_kl,mean_tt_vs_ts,mean_ss_vs_ts

# def get_kl_GLEM():
#     text_features =np.load('./TAG_data/cora/embs_e_cora_GLEM.npy') # 示例文本模态特征
#     struct_features =np.load('./TAG_data/cora/embs_m_cora_GLEM.npy')  # 示例结构模态特征
#     test_mask=np.load('./TAG_data/cora/test_mask_cora_GLEM.npy')
#     text_features=text_features[test_mask]
#     struct_features=struct_features[test_mask]
#     text_features=torch.from_numpy(text_features)
#     struct_features=torch.from_numpy(struct_features)
#     text_text_similarity = torch.mm(text_features, text_features.T)
#     struct_struct_similarity = torch.mm(struct_features, struct_features.T)
#     text_struct_similarity= torch.mm(text_features, struct_features.T)

#     text_text_vs_text_struct_kl =kl_divergence(
#         text_text_similarity, text_struct_similarity,dim=1
#     ).numpy()
#     struct_struct_vs_text_struct_kl =kl_divergence(
#         struct_struct_similarity ,text_struct_similarity.T,dim=1
#     ).numpy()
#     mean_tt_vs_ts,mean_ss_vs_ts=np.mean(text_text_vs_text_struct_kl),np.mean(struct_struct_vs_text_struct_kl)
#     return text_text_vs_text_struct_kl,struct_struct_vs_text_struct_kl,mean_tt_vs_ts,mean_ss_vs_ts


# def get_kl_ours():
#     text_features =np.load('./TAG_data/cora/embs_e_cora_ours.npy') # 示例文本模态特征
#     struct_features =np.load('./TAG_data/cora/embs_m_cora_ours.npy')  # 示例结构模态特征
#     test_mask=np.load('./TAG_data/cora/test_mask_cora.npy')
#     text_features=text_features[test_mask]
#     struct_features=struct_features[test_mask]
#     text_features=torch.from_numpy(text_features)
#     struct_features=torch.from_numpy(struct_features)
#     text_text_similarity = torch.mm(text_features, text_features.T)
#     struct_struct_similarity = torch.mm(struct_features, struct_features.T)
#     text_struct_similarity= torch.mm(text_features, struct_features.T)

#     text_text_vs_text_struct_kl =kl_divergence(
#         text_text_similarity, text_struct_similarity,dim=1
#     ).numpy()
#     struct_struct_vs_text_struct_kl =kl_divergence(
#         struct_struct_similarity ,text_struct_similarity.T,dim=1
#     ).numpy()
#     mean_tt_vs_ts,mean_ss_vs_ts=np.mean(text_text_vs_text_struct_kl),np.mean(struct_struct_vs_text_struct_kl)
#     return text_text_vs_text_struct_kl,struct_struct_vs_text_struct_kl,mean_tt_vs_ts,mean_ss_vs_ts

# concat_kl1,concat__kl2,concat__mean_1,concat_mean_2=get_kl_concat()
# ConGraT_kl1,ConGraT_kl2,ConGraT_mean_1,ConGraT_mean_2=get_kl_ConGraT()
# simTAG_kl1,simTAG_kl2,simTAG_mean_1,simTAG_mean_2=get_kl_simTAG()
# GLEM_kl1,GLEM_kl2,GLEM_mean_1,GLEM_mean_2=get_kl_GLEM()
# our_kl1,our_kl2,our_mean_1,our_mean_2=get_kl_ours()
# # Create x-axis values
# x = range(len(ConGraT_kl1))
# print(1e5*ConGraT_kl1)
# print(1e4*simTAG_kl1)
# # Create the plot
# fig, ax = plt.subplots(figsize=(10, 6),dpi=1200)
# ax.axhline(y=1e3*ConGraT_mean_1, color='blue', linestyle='--', linewidth=1.5, label='ConGraT_mean')
# ax.axhline(y=GLEM_mean_1, color='y', linestyle='--', linewidth=1.5, label='GLEM_mean')
# ax.axhline(y=our_mean_1, color='r', linestyle='--', linewidth=1.5, label='our_mean')
# ax.axhline(y=concat__mean_1, color='orange', linestyle='--', linewidth=1.5, label='concat_mean')
# ax.axhline(y=1e3*simTAG_mean_1, color='c', linestyle='--', linewidth=1.5, label='simTAG_mean')


# # 绘制折线图
# plt.plot(x, concat_kl1, label='Concat_kl')  # 第一组数据
# plt.plot(x, 1e3*ConGraT_kl1, label='ConGraT_kl')  # 第一组数据
# plt.plot(x,GLEM_kl1, label='GLEM_kl1')  # 第二组数据
# plt.plot(x,our_kl1, label='our_kl')  # 第三组数据
# plt.plot(x,1e3*simTAG_kl1, label='simTAG_kl')  # 第三组数据
# #plt.plot(x, y4, label='Group 4', marker='d')  # 第四组数据

# # 添加图例、标题和轴标签
# plt.title('Line Chart of Four Groups')
# plt.xlabel('X-axis Label')
# plt.ylabel('Y-axis Label')
# plt.legend()  # 显示图例


# # 显示图表
# plt.show()

# save_file='./pic/aligned_all.png'
# plt.savefig(save_file)