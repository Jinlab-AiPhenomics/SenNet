
import matplotlib.pyplot as plt
import numpy as np

# 模型名称
models = ['illness', 'health', 'wheatear', 'soil', 'sky']

# 第一组数据
data1 = [71.71, 85.74, 88.65, 94.05, 89.98]

# 第二组数据
data2 = [83.37, 91.59, 94.07, 94.90, 95.25]

# 设置柱状图的宽度
bar_width = 0.35

# 设置x轴位置
x = np.arange(len(models))

# 自定义颜色
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# 绘制柱状图
plt.bar(x - bar_width/2, data1, bar_width, label='base', color=[color + '55' for color in colors])
plt.bar(x + bar_width/2, data2, bar_width, label='ours', color=colors)  # 添加透明度以增加对比度

# 添加标签和标题
plt.xlabel('class')
plt.ylabel('Performance')
plt.title('Comparison of Performance after modification')

# 设置y轴范围从80开始
plt.ylim(70, 100)

plt.xticks(x, models)
plt.legend()

# 显示图表
plt.show()
