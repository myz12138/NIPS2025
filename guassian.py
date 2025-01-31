import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 定义高斯分布的参数
mean = 0  # 均值
std_dev = 1  # 标准差

# 生成数据点
x = np.linspace(mean - 4 * std_dev, mean + 4 * std_dev, 500)
y = norm.pdf(x, mean, std_dev)

# 绘制高斯分布图像
plt.plot(x, y, color='blue')  # 绘制高斯分布曲线

# 添加均值处的垂直虚线，限制其范围不超过 x 轴
plt.vlines(mean, ymin=0, ymax=max(y), color='red', linestyle='--')

# 在虚线附近标注 "u"
plt.text(mean + 0.1, max(y) * 0.9, r'$\mu$', color='red', fontsize=12)

# 在图中标注方差
plt.text(mean + 2 * std_dev, max(y) * 0.5, r'$\sigma^2 = {}$'.format(std_dev**2), color='black', fontsize=12)

# 设置横坐标轴
plt.gca().spines['top'].set_visible(False)  # 隐藏顶部边框
plt.gca().spines['right'].set_visible(False)  # 隐藏右侧边框
plt.gca().spines['left'].set_visible(False)  # 隐藏左侧边框
plt.gca().spines['bottom'].set_position(('data', 0))  # 将横坐标轴移到 y=0 的位置

plt.xticks([])  # 移除 x 轴刻度
plt.yticks([])  # 不显示纵坐标刻度

# 移除图例
plt.title('Gaussian Distribution')
plt.xlabel('x')

# 保存为 PDF 文件
plt.savefig('./gaussian.pdf', format='pdf')
