import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def task_plt_2x4(model_names, dataset_paths):
    """
    绘制 8 个 T-SNE 降维结果，2x4 排列。
    第一行使用文本特征，第二行使用结构特征。
    
    Args:
        model_names (list): 模型名称列表。
        dataset_paths (list): 每个数据集的路径字典列表，包含文本特征、结构特征、类别和测试掩码路径。
    """
    num_datasets = len(model_names)
    fig, axes = plt.subplots(2, num_datasets, figsize=(6 * num_datasets, 12), dpi=300)

    # 预定义全局颜色映射
    all_categories = set()
    for paths in dataset_paths:
        categories = np.load(paths['categories'])
        all_categories.update(categories)
    all_categories = sorted(all_categories)
    num_classes = len(all_categories)
    color_map = plt.cm.get_cmap('tab10', num_classes)
    class_to_color = {class_id: color_map(i) for i, class_id in enumerate(all_categories)}

    for i, (model_name, paths) in enumerate(zip(model_names, dataset_paths)):
        print(model_name)
        # 加载数据
        text_features = np.load(paths['text_features'])
        struct_features = np.load(paths['struct_features'])
        categories = np.load(paths['categories'])
        test_mask = np.load(paths['test_mask'])

        # 如果不是 ConGraT 模型，使用测试掩码过滤
        #if model_name=='ours' or  model_name=='msealign'or model_name=='myalign' :
        text_features = text_features[test_mask]
        struct_features = struct_features[test_mask]
        categories = categories[test_mask]

        # 使用 T-SNE 降维到 2D
        tsne = TSNE(n_components=2, random_state=42, early_exaggeration=10)
        reduced_text_features = tsne.fit_transform(text_features)
        reduced_struct_features = tsne.fit_transform(struct_features)

        # 第一行：文本特征
        ax_text = axes[0, i]
        for class_id in all_categories:
            mask = categories == class_id
            ax_text.scatter(
                reduced_text_features[mask, 0],
                reduced_text_features[mask, 1],
                label=f'Class {class_id}',
                alpha=0.7,
                color=class_to_color[class_id],
                marker='o'
            )
        ax_text.set_title(f"Text Features - {model_name}", fontsize=20)
        ax_text.set_xlabel("")
        ax_text.set_ylabel("")
        ax_text.legend(loc='best', fontsize=8)

        # 第二行：结构特征
        ax_struct = axes[1, i]
        for class_id in all_categories:
            mask = categories == class_id
            ax_struct.scatter(
                reduced_struct_features[mask, 0],
                reduced_struct_features[mask, 1],
                label=f'Class {class_id}',
                alpha=0.7,
                color=class_to_color[class_id],
                marker='o'
            )
        ax_struct.set_title(f"Struct Features - {model_name}", fontsize=20)
        ax_struct.set_xlabel("")
        ax_struct.set_ylabel("")
        ax_struct.legend(loc='best', fontsize=12)


    # 调整布局并保存
    plt.tight_layout()
    plt.savefig('./pic/task-'+data_name+'.png')
    plt.savefig('./pic/task-'+data_name+'.pdf')


# 示例调用
model_names = ['ours','ConGraT','myalign', 'msealign' ]#myalign:我们的方法，先对齐，再分类。msealign:用mse对齐，然后再分类
folder='./TAG_data/cora/'
data_name='cora'
dataset_paths = [
    {
        'text_features': folder+'embs_e_'+data_name+'_'+model_names[0]+'.npy',
        'struct_features': folder+'embs_m_'+data_name+'_'+model_names[0]+'.npy',
        'categories': folder+'labels_'+data_name+'.npy',
        'test_mask': folder+'test_mask_'+data_name+'.npy'
    },
    {
        'text_features': folder+'embs_e_'+data_name+'_'+model_names[1]+'.npy',
        'struct_features': folder+'embs_m_'+data_name+'_'+model_names[1]+'.npy',
        'categories': folder+'labels_'+data_name+'.npy',
        'test_mask': folder+'test_mask_'+data_name+'.npy'
    },
    
    {
        'text_features': folder+'embs_e_'+data_name+'_'+model_names[2]+'.npy',
        'struct_features': folder+'embs_m_'+data_name+'_'+model_names[2]+'.npy',
        'categories': folder+'labels_'+data_name+'.npy',
        'test_mask': folder+'test_mask_'+data_name+'.npy'
    },
    {
        'text_features': folder+'embs_e_'+data_name+'_'+model_names[3]+'.npy',
        'struct_features': folder+'embs_m_'+data_name+'_'+model_names[3]+'.npy',
        'categories': folder+'labels_'+data_name+'.npy',
        'test_mask': folder+'test_mask_'+data_name+'.npy'
    }
    ]
model_names = ['Ours','ConGraT','Ours-splited', 'ConGraT-MSE' ]
task_plt_2x4(model_names, dataset_paths)   

# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from data_pro import read_data
# import torch.nn.functional as F
# def task_plt_alltest(model_name):
#     text_features =np.load('./TAG_data/cora/embs_e_cora_'+model_name+'.npy') # 示例文本模态特征
#     struct_features =np.load('./TAG_data/cora/embs_m_cora_'+model_name+'.npy')  # 示例结构模态特征
#     categories=np.load('./TAG_data/cora/labels_cora.npy')
#     test_mask=np.load('./TAG_data/cora/test_mask_cora.npy')
#     if model_name!='ConGraT':
#         text_features=text_features[test_mask] 
#         struct_features=struct_features[test_mask] 
#     categories = categories[test_mask]  # 假设这是类别的 tensor

#     # 使用 T-SNE 降维到 2D
#     tsne = TSNE(n_components=2, random_state=42)
#     tsne_text = TSNE(n_components=2, random_state=42,early_exaggeration=10)
#     reduced_text_features = tsne_text.fit_transform(text_features)
#     reduced_struct_features =tsne_text.fit_transform(struct_features)

    
#     #reduced_initial_features =tsne_text.fit_transform(struct_features)
#     # 绘制 T-SNE 图
#     plt.figure(figsize=(8, 6),dpi=1200)

#     # 定义类别的颜色和标签
#     num_classes = len(np.unique(categories))
#     colors = plt.cm.get_cmap('tab10', num_classes)  # 使用 tab10 调色板

# # 按类别绘制点
# # for class_id in range(num_classes):
# #     mask = categories == class_id
# #     plt.scatter(
# #     reduced_text_features[mask, 0],
# #     reduced_text_features[mask, 1],
# #     label=f'Text Class {class_id}',
# #     alpha=0.6,
# #     color=colors(class_id),
# #     marker='o'  # Circle marker for text features
# # )
# # plt.title("T-SNE Visualization of Modal Alignment with Categories")
# # plt.xlabel("Dimension 1")
# # plt.savefig('T-SN-struct.png')

# # Plot structural features
#     for class_id in range(num_classes):
#         mask = categories == class_id
#         plt.scatter(
#             reduced_text_features[mask, 0],
#             reduced_text_features[mask, 1],
#             label=f'Struct Class {class_id}',
#             alpha=1,
#             color=colors(class_id),
#             marker='o'  # Cross marker for structural features
#         )
#     plt.title("T-SNE Visualization of Modal Alignment with Categories")
#     plt.xlabel("Dimension 1")
#     plt.savefig('./pic/task-'+model_name+'.png')

# if __name__=='__main__':
#     task_plt_alltest('ConGraT')
#     task_plt_alltest('GLEM')
#     task_plt_alltest('concat')
#     task_plt_alltest('ours')
