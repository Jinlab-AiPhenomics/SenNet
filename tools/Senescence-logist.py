import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import integrate
from scipy.optimize import leastsq, curve_fit


# 定义Logistic函数
def func(x, b, c, d):
    return 1 / (1 + b * np.exp(c * (x + d)))


# 定义误差函数


# 计算误差指标
def calculate_errors(Y_true, Y_pred):
    ssr = np.sum(np.square(Y_true - Y_pred))  # 残差平方和 SSR
    mse = np.mean(np.square(Y_true - Y_pred))  # 均方误差 MSE
    ss_tot = np.sum(np.square(Y_true - np.mean(Y_true)))  # 总平方和
    r_squared = 1 - ssr / ss_tot  # R^2 决定系数
    return ssr, mse, r_squared


def logist_derivative(x, b, c, d):
    return -b * c * np.exp(c * (x + d)) / (1 + b * np.exp(c * (x + d))) ** 2


# Second derivative of the logistic function
def logist_second_derivative(x, b, c, d):
    return (b * c ** 2 * np.exp(c * (x + d)) * (b * np.exp(c * (x + d)) - 1)) / (1 + b * np.exp(c * (x + d))) ** 3


# Function to compute STS, Mincurve, Steepness, Maxcurve, ETS, and TOS
def compute_features(x, Sen_1=0.95, Sen_2=0.05):
    # Compute STS and ETS
    STS = (1 / c) * np.log((1 - Sen_1) / (b * Sen_1)) - d
    ETS = (1 / c) * np.log((1 - Sen_2) / (b * Sen_2)) - d

    # Compute Mincurve and Maxcurve (second derivative)
    Mincurve = logist_second_derivative(x, b, c, d)
    Maxcurve = logist_second_derivative(x, b, c, d)

    # Compute Steepness (first derivative)
    Steepness = logist_derivative(x, b, c, d)

    # TOS (ETS - STS)
    TOS = ETS - STS

    return STS, Mincurve, Steepness, Maxcurve, ETS, TOS


# Integration for AreaUnderCurve and AreaUnderSenRate
def integrate_area(x1, x2, b, c, d):
    # Area under the curve
    area_curve, _ = integrate.quad(lambda x: func(x, b, c, d), x1, x2)

    # Area under SenRate
    area_sen_rate, _ = integrate.quad(lambda x: (b * c * np.exp(c * (x + d))) / (1 + b * np.exp(c * (x + d))) ** 2, x1,
                                      x2)

    return area_curve, area_sen_rate




file_path = 'sorted_2023corn-para-with-predictions.xlsx'  # 替换成你的Excel文件路径
df = pd.read_excel(file_path)
column_indices = df.columns.tolist()
x_data = np.array(column_indices[1:8])

# 处理每一行数据
for index, row in df.iterrows():
    y_data = np.array(row.values[1:8])
    popt, pcov = curve_fit(func, x_data, y_data, p0=[1, -0.001, 0],
                           bounds=([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf]))
    b, c, d = popt

    # 计算特征
    x_fit = np.linspace(min(x_data) - 200, max(x_data) + 200, 500)
    y_fit = func(x_fit, b, c, d)

    STS, Mincurve, Steepness, Maxcurve, ETS, TOS = compute_features(x_data)
    area_curve, area_sen_rate = integrate_area(x_data[0], x_data[-1], b, c, d)
    # 保存特征到DataFrame
    df.at[index, 'STS'] = STS
    df.at[index, 'Mincurve'] = Mincurve
    df.at[index, 'Steepness'] = Steepness
    df.at[index, 'Maxcurve'] = Maxcurve
    df.at[index, 'ETS'] = ETS
    df.at[index, 'TOS'] = TOS
    df.at[index, 'area_curve'] = area_curve
    df.at[index, 'area_sen_rate'] = area_sen_rate

# 将结果保存到Excel文件
output_file = "2023corn.xlsx"
df.to_excel(output_file, index=False, float_format='%.4f')
