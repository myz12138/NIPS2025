import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from data_pro import read_data
import torch.nn.functional as F
def row_normalize(matrix):
    row_sums = matrix.sum(dim=1, keepdim=True)  # Sum along rows
    return matrix / (row_sums + 1e-8)  # Avoid division by zero


def column_normalize(matrix):
    col_sums = matrix.sum(dim=0, keepdim=True)  # Sum along columns
    return matrix / (col_sums + 1e-8)  # Avoid division by zero

def kl_divergence(p, q,dim):
# Add a small constant (1e-8) to avoid log(0) or division by zero
    p =F.softmax(p) + 1e-8
    q = F.softmax(q)  + 1e-8
    return (p * (p.log() - q.log())).mean(dim=dim)

text_features =np.load('./TAG_data/cora/embs_e_cora_GLEM.npy') # 示例文本模态特征
struct_features =np.load('./TAG_data/cora/embs_m_cora_GLEM.npy')  # 示例结构模态特征
categories_tensor=np.load('./TAG_data/cora/labels_cora.npy')
test_mask=np.load('./TAG_data/cora/test_mask_cora_GLEM.npy')
text_features=text_features[test_mask]
struct_features=struct_features[test_mask]
text_features=torch.from_numpy(text_features)
struct_features=torch.from_numpy(struct_features)
text_text_similarity = torch.mm(text_features, text_features.T)
struct_struct_similarity = torch.mm(struct_features, struct_features.T)
text_struct_similarity= torch.mm(text_features, struct_features.T)

text_text_vs_text_struct_kl = kl_divergence(
    text_text_similarity, text_struct_similarity,dim=1
)

# Compute relative entropy between struct_struct_col_normalized and text_struct_col_normalized
struct_struct_vs_text_struct_kl = kl_divergence(
    struct_struct_similarity ,text_struct_similarity.T,dim=1
)
# Normalize the similarity matrices
# text_text_row_normalized = row_normalize(text_text_similarity)
# text_text_col_normalized = column_normalize(text_text_similarity)

# text_struct_row_normalized = row_normalize(text_struct_similarity)
# text_struct_col_normalized = column_normalize(text_struct_similarity)

# struct_struct_row_normalized = row_normalize(struct_struct_similarity)
# struct_struct_col_normalized = column_normalize(struct_struct_similarity)


a=1
# # Step 1: Load stock prices from a file
# input_file = "relative_entropy_results.txt"
# with open(input_file, "r") as f:
#     stock_prices = [float(line.strip()) for line in f]

# Step 2: Convert the values to a PyTorch tensor
stock_prices_tensor =text_text_vs_text_struct_kl

second_stock_tensor = struct_struct_vs_text_struct_kl

stock_prices = stock_prices_tensor.numpy()
second_stock = second_stock_tensor.numpy()
mean_stock=np.mean( stock_prices)
mean_second=np.mean( second_stock)
# Create x-axis values
x = range(len(stock_prices))

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6),dpi=1200)
ax.axhline(y=mean_stock, color='blue', linestyle='--', linewidth=3, label='Mean-KL-T')
ax.axhline(y=-mean_second, color='red', linestyle='--', linewidth=3, label='Mean-KL-G')
# Plot the first dataset as an area chart above the x-axis
ax.fill_between(x, stock_prices, color='blue', alpha=0.5, label='KL-T')

# Plot the second dataset as an area chart below the x-axis
ax.fill_between(x, -second_stock, color='red', alpha=0.5, label='KL-G')

# Add labels, legend, and grid
ax.set_xlabel('ID of test node')
ax.set_ylabel('KL')
ax.legend(loc='upper left')
ax.grid(True)

y_ticks = np.arange(0, 0.07, 0.01) 
y_ticks2 = np.arange(-0.05, 0.0, 0.01) 
print(np.concatenate([y_ticks2, y_ticks]))
ax.set_yticks(np.concatenate([y_ticks2, y_ticks])) 
y_tick_labels = [f'{abs(x):.2f}' for x in ax.get_yticks()]
ax.set_yticklabels(y_tick_labels)
# Add a second y-axis for the bottom dataset
plt.title('Modal Alignment Analysis')
plt.show()
output_image_file = "./pic/modal_align_GLEM.pdf"
plt.savefig(output_image_file)
