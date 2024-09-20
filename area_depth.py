import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# 读取CSV文件
file_path = 'area_depth.csv'  # 替换为实际的文件路径
data = pd.read_csv(file_path)

# 提取depth和area数据
y = data['depth'].values
x = data['max_area'].values

# 定义幂函数模型
def power_law(x, a, b):
    return a * np.power(x, b)

# 幂函数拟合（以area为自变量，depth为因变量）
params, covariance = curve_fit(power_law, x, y)
a, b = params

# 计算拟合值
y_fit = power_law(x, a, b)

# 计算R平方值
ss_res = np.sum((y - y_fit) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - (ss_res / ss_tot)

# 输出拟合关系式和精确度
print(f'拟合关系式: force = {a:.4f} * intensity^{b:.4f}')
print(f'拟合精确度R²: {r2:.4f}')

# 绘制拟合曲线
plt.scatter(x, y, label='Data')
plt.plot(x, y_fit, color='red', label='Power Law Fit')
plt.xlabel('intensity')
plt.ylabel('force')
plt.legend()
plt.title('Fit')
plt.show()
