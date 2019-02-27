#!/usr/bin/env python
# coding: utf-8

# 本文利用逻辑斯蒂回归模型进行数据分析预测

# In[1]:


#使用支持向量机算法对Credit card fraud detection数据集进行分析
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import time
import math
import random
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


def loadData(fileName):
    '''
    加载breast cancer数据集
    :param fileName:要加载的数据集路径
    :return: list形式的数据集及标记
    '''
    print('数据读取')
    # 存放数据及标记的list
    dataArr = []; labelArr = []
    # 打开文件
    fr = open(fileName, 'r')
    # 将文件按行读取
    lines=fr.readlines()
    for line in lines[1:]:
        # 对每一行数据按切割福','进行切割，返回字段列表
        curLine = line.strip().split(',')
        if curLine[29]=='0':
            labelArr.append(0)
        else:
            labelArr.append(1)
        #存放标记
        dataArr.append([float(num) for num in curLine[1:29]])
    #返回data和label
    return dataArr, labelArr


# In[7]:


def logisticRegression(trainDataList, trainLabelList, iter = 200):
    '''
    逻辑斯蒂回归训练过程
    :param trainDataList:训练集
    :param trainLabelList: 标签集
    :param iter: 迭代次数
    :return: 习得的w
    '''
    #按照书本“6.1.2 二项逻辑斯蒂回归模型”中式6.5的规则，将w与b合在一起，
    #此时x也需要添加一维，数值为1
    #循环遍历每一个样本，并在其最后添加一个1
    for i in range(len(trainDataList)):
        trainDataList[i].append(1)
    #将数据集由列表转换为数组形式，主要是后期涉及到向量的运算，统一转换成数组形式比较方便
    trainDataList = np.array(trainDataList)
    #初始化w，维数为样本x维数+1，+1的那一位是b，初始为0
    w = np.zeros(trainDataList.shape[1])
    #设置步长
    h = 0.001
    #迭代iter次进行随机梯度下降
    for i in range(iter):
        #每次迭代冲遍历一次所有样本，进行随机梯度下降
        for j in range(trainDataList.shape[0]):
            #随机梯度上升部分
            wx = np.dot(w, trainDataList[j])
            yi = trainLabelList[j]
            xi = trainDataList[j]
            #梯度上升
            w +=  h * (xi * yi - (np.exp(wx) * xi) / ( 1 + np.exp(wx)))
    #返回学到的w
    return w


# In[8]:


def predict(w, x):
    '''
    预测标签
    :param w:训练过程中学到的w
    :param x: 要预测的样本
    :return: 预测结果
    '''
    #dot为两个向量的点积操作，计算得到w * x
    wx = np.dot(w, x)
    #计算标签为1的概率
    #该公式参考“6.1.2 二项逻辑斯蒂回归模型”中的式6.5
    P1 = np.exp(wx) / (1 + np.exp(wx))
    #如果为1的概率大于0.5，返回1
    if P1 >= 0.5:
        return 1
    #否则返回0
    return 0


# In[9]:


def test(testDataList, testLabelList, w):
    '''
    验证
    :param testDataList:测试集
    :param testLabelList: 测试集标签
    :param w: 训练过程中学到的w
    :return: 正确率
    '''
    #与训练过程一致，先将所有的样本添加一维，值为1，理由请查看训练函数
    for i in range(len(testDataList)):
        testDataList[i].append(1)
    #错误值计数
    errorCnt = 0
    #对于测试集中每一个测试样本进行验证
    for i in range(len(testDataList)):
        #如果标记与预测不一致，错误值加1
        if testLabelList[i] != predict(w, testDataList[i]):
            errorCnt += 1
    #返回准确率
    return 1 - errorCnt / len(testDataList)


# In[12]:


start = time.time()
# 获取训练集及标签
print('start read transSet')
trainDataList, trainLabelList = loadData('F://Machine Learning//ML//Data//Datasets//creditcardfraud//train_data.csv')
# 获取测试集及标签
print('start read testSet')
testDataList, testLabelList = loadData('F://Machine Learning//ML//Data//Datasets//creditcardfraud//test_data.csv')
#初始化SVM类
# 开始训练
print('start to train')
w = logisticRegression(trainDataList, trainLabelList)
# 开始测试
print('start to test')
accuracy = test(testDataList, testLabelList, w)
print('the accuracy is:%d'%(accuracy * 100), '%')
# 打印时间
print('time span:', time.time() - start)


# In[ ]:




