# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 15:56:03 2019

@author: Administrator
"""

from matplotlib import pyplot as plt 
import numpy as np
import pandas as pd

dataset = pd.read_csv("50_Startups.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

# 数据预处理的热编码 将分类标签数据编码成虚拟变量
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# 编码
labelencoder = LabelEncoder()
x[:,3] = labelencoder.fit_transform(x[:,3])
# 转换成虚拟变量
onehotencoder = OneHotEncoder(categorical_features=[3])
x = onehotencoder.fit_transform(x).toarray()

# 避免虚拟变量陷阱 保证所有的特征之间不存在可计算的关联性
x = x[:,1:]

# 数据分割
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=0)

# 多元线性预测 创建模型
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
# 训练回归器
regressor.fit(x_train,y_train)

# 预测测试模型
y_dirc = regressor.predict(x_test)

# 创建最佳模型使用反向消除法（计算P_Value p值越大说明这个特征对结果的影响越小）
import statsmodels.formula.api as sm
x_train = np.append(np.ones(shape=(40,1)), values=x_train, axis=1)
x_opt = x_train[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=y_train, exog=x_opt).fit()
# 输出模型的测试结果
regressor_OLS.summary()

# remove x2
x_opt = x_train[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog=y_train, exog=x_opt).fit()
regressor_OLS.summary()

# remove x1
x_opt = x_train[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog=y_train, exog=x_opt).fit()
regressor_OLS.summary()

# remove administration
x_opt = x_train[:,[0,3,5]]
regressor_OLS = sm.OLS(endog=y_train, exog=x_opt).fit()
regressor_OLS.summary()
















