# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# import dataset
dataset = pd.read_csv("Salary_Data.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values


# 分割数据集和训练集
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=1/3, random_state=0)

# 简单线性回归模型
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(x_train, y_train)

# 预测结果
y_pred = regression.predict(x_test)

# 可视化散点图和训练模型的线性图
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regression.predict(x_train), color='blue')
plt.xlabel('Years VS Experience (train set)')
plt.ylabel('Salary')
plt.show()

# 可视化测试集的图像
plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, regression.predict(x_train), color='blue')
plt.xlabel("Year VS Experience (test set)")
plt.ylabel("Salary")
plt.show()