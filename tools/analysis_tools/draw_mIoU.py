# import matplotlib.pyplot as plt
# import numpy as np
#
# # 模型名称和对应的mIoU值
# models = ['ours', 'mobile', 'ocrnet', 'point', 'swin', 'twinsvt', 'uper', 'segmenter']
# miou_values = [92.06, 89.03, 91.04, 89.49, 91.42, 91.53, 88.54, 86.54]
# health = [83.37, 78.72, 82.52, 79.36, 83.17, 82.81, 74.65, 77.51]
# illness = [91.59, 88.77, 91.15, 89.73, 91.51, 91.01, 87.07, 87.13]
# wheat_ear = [94.07, 89.56, 93.51, 92.53, 93.44, 93.16, 91.65, 85.95]
# sky = [94.9, 94.2, 94.26, 90.97, 94.01, 94.27, 94.25, 92.98]
# soil = [95.95, 93.89, 96.67, 96.79, 95.29, 95.27, 94.57, 89.12]
# # 创建条形图
# fig, ax = plt.subplots()
# # 为每个模型选择Viridis颜色映射中的颜色
# cmap = plt.get_cmap('viridis')
# colors = cmap(np.linspace(0, 1, len(models)))
#
# bars = ax.bar(models, miou_values, color=colors)
#
# # 添加数据标签
# for bar in bars:
#     yval = bar.get_height()
#     ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', ha='center', va='bottom')  # ha: horizontal alignment
#
# # 设置标题和标签
# ax.set_title('Comparison of mIoU for Different Semantic Segmentation Models')
# ax.set_xlabel('Models')
# ax.set_ylabel('mIoU (%)')
# ax.set_ylim([80, max(miou_values) + 1])  # 增加上限0.05保证顶部空间足够
# # 显示图表
# plt.savefig("mIoU-bar.png")
# plt.show()
import matplotlib.pyplot as plt

# 数据
miou_values = [92.06, 89.03, 91.04, 89.49, 91.42, 91.53, 88.54, 86.54]
health = [83.87, 78.72, 82.52, 79.36, 83.17, 82.81, 74.65, 77.51]
illness = [91.89, 88.77, 91.15, 89.73, 91.51, 91.01, 87.07, 87.13]
wheat_ear = [94.87, 89.56, 93.51, 92.53, 93.44, 93.16, 91.65, 85.95]
sky = [94.9, 94.2, 94.26, 90.97, 94.01, 94.27, 94.25, 92.98]
soil = [95.95, 93.89, 96.67, 96.79, 95.29, 95.27, 94.57, 89.12]
models = ['Ours', 'Mobile', 'Ocrnet', 'Point', 'Swin', 'Twinsvt', 'Uper', 'Segmenter']
xl = [1, 2, 3, 4, 5, 6, 7, 8]
# 绘图
fig, axs = plt.subplots(2, 3, figsize=(18, 10))

# 第一行
axs[0, 0].bar(xl, miou_values, color='white', edgecolor="black")
# axs[0, 0].set_title('miou_values', fontsize=20)
axs[0, 0].set_ylabel('mIoU(%)', fontsize=20)
axs[0, 0].set_ylim([min(miou_values)-1, max(miou_values) + 1])
axs[0, 0].set_xticks(xl, models, rotation=30)



axs[0, 1].bar(xl, health, color=(0.329, 0.675, 0.459), alpha=1)
# axs[0, 1].set_title('health', fontsize=20)
axs[0, 1].set_ylim([min(health)-1, max(health) + 1])
axs[0, 1].set_xticks(xl, models, rotation=30)


axs[0, 2].bar(xl, illness, color=(197/255, 86/255, 89/255), alpha=0.7)
# axs[0, 2].set_title('illness', fontsize=20)
axs[0, 2].set_ylim([min(illness)-1, max(illness) + 1])
axs[0, 2].set_xticks(xl, models, rotation=30)

# 第二行
axs[1, 0].bar(xl, wheat_ear, color=(203/255, 180/255, 123/255), alpha=0.7)
# axs[1, 0].set_title('wheat_ear', fontsize=20)
axs[1, 0].set_ylabel('mIoU(%)', fontsize=20)
axs[1, 0].set_ylim([min(wheat_ear)-1, max(wheat_ear) + 1])
axs[1, 0].set_xticks(xl, models, rotation=30)


axs[1, 1].bar(xl, sky, color=(117/255, 114/255, 181/255), alpha=0.7)
# axs[1, 1].set_title('sky', fontsize=20)
axs[1, 1].set_ylim([min(sky)-1, max(sky) + 1])
axs[1, 1].set_xticks(xl, models, rotation=30)


axs[1, 2].bar(xl, soil, color=(71/255, 120/255, 185/255), alpha=0.7)
# axs[1, 2].set_title('soil', fontsize=20)
axs[1, 2].set_ylim([min(soil)-1, max(soil) + 1])
axs[1, 2].set_xticks(xl, models, rotation=30)


for ax in axs.flat:
    ax.tick_params(axis='both', which='major', labelsize=20)

# 调整布局
plt.tight_layout()
plt.show()

