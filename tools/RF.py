import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# 读取Excel文件
file_path = 'sorted_2023corn-para-with-predictions.xlsx'
data = pd.read_excel(file_path)

# 假设数据有相应的特征和目标列
# 确保你的DataFrame有这些列
features = ['STS', 'Mincurve', 'Steepness', 'Maxcurve', 'ETS', 'TOS']  # 替换为你的实际特征列
target = 'Yield'  # 替换为你的目标列

# 提取特征和目标变量
X = data[features]
y = data[target]

# 数据预处理：去除缺失值
X = X.dropna()
y = y[X.index]

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 创建并训练随机森林模型
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# 使用交叉验证来评估模型表现
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print(f"Cross-validated MSE (negative): {cv_scores.mean()}")

# 训练模型
rf_model.fit(X_train, y_train)

# 预测产量
y_pred = rf_model.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R-squared (R²):", r2)
print("Mean Absolute Error (MAE):", mae)

# 显示部分预测结果
predictions = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})

# 分类：将预测值按照实际产量的25%, 50%, 75%分位数进行划分
q1 = y_test.quantile(0.25)
q2 = y_test.quantile(0.5)
q3 = y_test.quantile(0.75)

def classify_yield(value):
    if value <= q1:
        return 'Low'
    elif value <= q2:
        return 'Medium'
    else:
        return 'High'

# 应用分类
predictions['Category'] = predictions['Predicted'].apply(classify_yield)

# 保存分类结果到新的Excel文件
predictions.to_excel('predictions_with_categories.xlsx', index=False)

# 查看部分分类结果
print(predictions.head())
