# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib as plt
import numpy as np

dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

# 处理NaN的值
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])

# 将标签编码 虚拟变量
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder().fit_transform(x[:,0])
x[:,0] = labelencoder_x
onehotencoder = OneHotEncoder(categorical_features=[0])
x = onehotencoder.fit_transform(x).toarray()
y = LabelEncoder().fit_transform(y)

# 测试集和训练集分割
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=0)

# 特征缩放
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

