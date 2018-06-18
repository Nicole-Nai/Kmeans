# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 14:02:12 2018

@author: Nicol
"""

import numpy as np


#x为数据集m*n的矩阵
def TrainTestSplit(x,test_size = 0.4):
    x_num = x.shape[0]
    train_index = range(x_num)
    test_index = []
    test_num = int(x_num*test_size)
    for i in range(test_num):
        ranindex = int(np.random.uniform(0,len(train_index)))
        test_index.append(train_index[ranindex])
        del train_index[ranindex]
    train = x.ix[train_index]
    test = x.ix[test_index]
    return train, test

#x为数据集m*n的矩阵,k为类别数
def KMeans(x,k=2):
    def rand_centroids(x,k):
        n = x.shape[1]
        centroids = np.zeors((k,n))
        for i in range (n):
            dmin = np.min(x[:,i])
            dmax = np.max(x[:,i])
            centroids[:,i] = dmin + (dmax-dmin)*np.random.rand(k)
            return centroids

    #将数据集以及k个中心点分别传输给k台机器
    #每台机器分别计算数据集里的每个点到中心点的距离
    #第一台机器
    centroids = rand_centroids(x,k)
    p = centroids[0,:]
    def distance(x,p):
        dist0 = []
        n = x.shape[1]
        q = np.zeros((1,n))
        for i in range(x.shape[0]):
            q = x[i,:]
            dis = np.sum((p-q)**2)
            dist0.append(np.sqrt(dis))
        return dist0
    #结果传输回主机，合并成m*k的矩阵y，比较每个点到中心点的距离，将数据归类到距离最近的中心点
    #assume that k = 2
    m = x.shape[0]
    label = np.zeros(m,dtype = np.int)

    def converged(old_centroids,centorids):
        set1 = set([tuple(c) for c in old_centroids])
        set2 = set([tuple(c) for c in centroids])
        return (set1 == set2)

    check = False
    while not check:
        old_centroids = np.copy(centroids)
        for i in range(m):
            min_dist = np.inf
            min_index = -1
            for j in range(k):
                dist0 = distance(x,centroids[0,:])
                dist1 = distance(x,centroids[1,:])
                re = np.hstack((dist0,dist1))
                y = re.T
                a = y[i:j]
                if a < min_dist:
                    min_dist = a
                    min_index = j
                    label[i] = j
        #计算新的k个中心点
        for i in range(k):
            centroids[i] = np.mean(x[label == i],axis = 0)
        #再重新将新的k个中心点传输到k台机器计算距离
        check = converged(old_centroids,centroids)
    return centroids,label





            