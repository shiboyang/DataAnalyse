# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,-1].values

# 简单线性回归模型
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

# 多项式线性回归模型训练
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)

# 可视化图形 (散点图 & 简单线性回归模型 图)
plt.scatter(x,y,color='red')
plt.plot(x, lin_reg.predict(x))
plt.title("Truth of Bluff (Linear Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# 多项式图形
plt.scatter(x, y, color='red')
# 由于x轴中的每个元素之间step太长所以线段显得并不平滑
x_grid = np.arange(min(x), max(x), step=0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.plot(x_grid, lin_reg2.predict(poly_reg.fit_transform(x_grid)), color='blue')
plt.title("Polynomia Features")
plt.xlabel("Positive Level")
plt.ylabel("Salary")
plt.show()