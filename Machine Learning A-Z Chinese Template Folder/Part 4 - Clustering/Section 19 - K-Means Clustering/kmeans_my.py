# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 22:40:10 2019

@author: Administrator

本次数据分析的需求为：
    1、根据用户的年收入和商场对用的打分聚合用户的类别
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Mall_Customers.csv")
x = dataset.iloc[:,[3,4]].values

# 测试手肘法计算最合适的族群数量
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, max_iter=300, n_init=10, init='k-means++', random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
# 可视化手肘图
plt.plot(range(1,11), wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of Cluter")
plt.ylabel("WCSS")
plt.show()
# 根据手肘法则可得知使用5个分类小效果是最好的

# 使用5个点对数据进行建模
kmeans = KMeans(n_clusters=5, random_state=0)
y_kmeans = kmeans.fit_predict(x)
# 可视化据类图像
plt.scatter(x[y_kmeans==0, 0], x[y_kmeans==0, 1], s=100, c='red', 
            label='Cluster0')
plt.scatter(x[y_kmeans==1, 0], x[y_kmeans==1, 1], s=100, c='blue', 
            label='Cluster1')
plt.scatter(x[y_kmeans==2, 0], x[y_kmeans==2, 1], s=100, c='green', 
            label='Cluster2')
plt.scatter(x[y_kmeans==3, 0], x[y_kmeans==3, 1], s=100, c='cyan', 
            label='Cluster3')
plt.scatter(x[y_kmeans==4, 0], x[y_kmeans==4, 1], s=100, c='magenta', 
            label='Cluster4')
# 画中心点
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300,
            c='yellow', label='Centroids')

plt.title('Cluster of Clients')
plt.xlabel("Annual of lncomes")
plt.ylabel("Spending Score")
plt.legend()
plt.show()