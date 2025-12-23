#calibration
"""


#for quadratic in two-point contact
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load data
file_path = 'intensity_force_two.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)

# Extract x1, x2, and y data
x1 = data['intensity1'].values
x2 = data['intensity2'].values
y = data['force'].values

# Prepare polynomial features of degree 3
X = np.column_stack((x1, x2))
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

# Perform linear regression on polynomial features
model = LinearRegression()
model.fit(X_poly, y)

# Get regression coefficients and intercept
coef = model.coef_
intercept = model.intercept_

# Predict values
y_pred = model.predict(X_poly)

# Calculate R-squared and RMSE
r2 = r2_score(y, y_pred)
rmse = mean_squared_error(y, y_pred, squared=False)

# Output the fitted equation and accuracy
print(f'equation: y = {coef[0]:.18f} + {coef[1]:.18f} * x1 + {coef[2]:.18f} * x2 + {coef[3]:.18f} * x1^2 + {coef[4]:.18f} * x1*x2 + {coef[5]:.18f} * x2^2 + {coef[6]:.18f} * x1^3 + {coef[7]:.18f} * x1^2*x2 + {coef[8]:.18f} * x1*x2^2 + {coef[9]:.18f} * x2^3')
print(f'R²: {r2:.8f}')
print(f'RMSE: {rmse:.8f}')

# Plot 3D scatter and fitted plane
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot scatter points
ax.scatter(x1, x2, y, color=[55/255, 103/255, 160/255], marker='o', label='Data Point')

# Create a grid for plotting the fitted surface
x1_range = np.linspace(x1.min(), x1.max(), 100)
x2_range = np.linspace(x2.min(), x2.max(), 100)
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
X_grid = np.column_stack((x1_grid.ravel(), x2_grid.ravel()))
X_poly_grid = poly.transform(X_grid)
y_grid = model.predict(X_poly_grid).reshape(x1_grid.shape)

# Plot the fitted surface
ax.plot_surface(x1_grid, x2_grid, y_grid, alpha=0.5, color=[250/255, 98/255, 84/255], label='Relation fit')

# Set axis labels
ax.set_xlabel('Intensity1')
ax.set_ylabel('Intensity2')
ax.set_zlabel('Force')
ax.set_title('Force-Intensity Relation')
ax.set_xticks(np.linspace(x1.min(), x1.max(), 5))  # X-axis ticks
ax.set_yticks(np.linspace(x2.min(), x2.max(), 5))
ax.grid(False)

# Show the plot
plt.show()
"""

#two point exploration
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,root_mean_squared_error
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit

import matplotlib.font_manager as fm

font_path = r"C:\Users\lenovo\AppData\Local\Microsoft\Windows\Fonts\Helvetica.ttf"
font_prop = fm.FontProperties(fname=font_path)
font_name = font_prop.get_name()
plt.rcParams.update({
    'font.serif': font_name,
    'font.size': 15,
    'axes.titlesize': 15,
    'axes.labelsize': 15,
    'legend.fontsize': 15,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
})
file_path = 'intensity_force_single.csv'
data = pd.read_csv(file_path)

y = data['force'].values
x = data['intensity'].values

def linear_func(x, a, b):
    return a * x + b

def quadratic_func(x, a, b, c):
    return a * x**2 + b * x + c

def cubic_func(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

# linear fit
params_linear, _ = curve_fit(linear_func, x, y)
a_linear, b_linear = params_linear
y_fit_linear = linear_func(x, a_linear, b_linear)
r2_linear = r2_score(y, y_fit_linear)


# quadratic fit
params_quadratic, _ = curve_fit(quadratic_func, x, y)
a_quadratic, b_quadratic, c_quadratic = params_quadratic
y_fit_quadratic = quadratic_func(x, a_quadratic, b_quadratic, c_quadratic)
r2_quadratic = r2_score(y, y_fit_quadratic)

# cubic line fit
params_cubic, _ = curve_fit(cubic_func, x, y)
x_fit = np.linspace(min(x), max(x), 100)
a_cubic, b_cubic, c_cubic, d_cubic = params_cubic
y_fit_cubic = cubic_func(x_fit, a_cubic, b_cubic, c_cubic, d_cubic)
y_p=cubic_func(x, a_cubic, b_cubic, c_cubic, d_cubic)
r2_cubic = r2_score(y, y_p)
rmse = root_mean_squared_error(y, y_p)

# 输出拟合关系式和精确度
print(f'linear equation: force = {a_linear:.8f} * intensity + {b_linear:.8f}')
print(f'R²: {r2_linear:.6f}')
print(f'quadrac equation: force = {a_quadratic:.6f} * intensity^2 + {b_quadratic:.4f} * intensity + {c_quadratic:.4f}')
print(f'R²: {r2_quadratic:.6f}')
print(f'cubic equation: force = {a_cubic:.17f} * intensity^3 + {b_cubic:.15f} * intensity^2 + {c_cubic:.15f} * intensity + {d_cubic:.9f}')
print(f'R²: {r2_cubic:.6f}')
print(f'RMSE: {rmse:.6f}')

fig = plt.figure(figsize=(10,6.18))

plt.scatter(x, y, color=[0,114/255,192/255],label='Data Point')
#plt.plot(x, y_fit_linear, color='yellow', label='Linear Fit')
#plt.plot(x, y_fit_quadratic, color='green', label='Quadratic Fit')
plt.plot(x_fit, y_fit_cubic, color=[218/255,84/255,17/255], label='Cubic Fit',linewidth='3')
plt.xlabel('Intensity')
plt.ylabel('Force(N)')
plt.show()