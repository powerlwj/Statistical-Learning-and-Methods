
# coding: utf-8

# 使用感知机对Breast Cancer Wisconsin (Diagnostic) Data Set数据进行分离超平面的计算
# 目标：Predict whether the cancer is benign or malignant
# 感知机相关方程和思路详见李航的《统计学习与方法》第二章
# 主要步骤如下：
# 1. 导入相关库，导入训练和测试数据
# 2. 数据清洗：将类别数据转化为数值数据，去除含有空值的行（必要步骤，否则会因为含有NAN值而在感知机训练过程中报错）
# 3. 写感知器训练函数与测试函数，原理见书本

# In[57]:


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import time
get_ipython().run_line_magic('matplotlib', 'inline')


# 特征信息:
# 1) ID number 2) Diagnosis (M = malignant, B = benign) 3-32)
# Ten real-valued features are computed for each cell nucleus:
# a) radius (mean of distances from center to points on the perimeter) b) texture (standard deviation of gray-scale values) c) perimeter d) area e) smoothness (local variation in radius lengths) f) compactness (perimeter^2 / area - 1.0) g) concavity (severity of concave portions of the contour) h) concave points (number of concave portions of the contour) i) symmetry j) fractal dimension ("coastline approximation" - 1)
# The mean, standard error and "worst" or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features. For instance, field 3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius.
# All feature values are recoded with four significant digits.
# Missing attribute values: none
# Class distribution: 357 benign, 212 malignant
# 训练集和测试集的分配：训练集470：测试集100

# In[58]:


train_data = pd.read_csv("F:\Machine Learning\ML\Data\Datasets\Breast-Cancer//train.csv ")
test_data = pd.read_csv("F:\Machine Learning\ML\Data\Datasets\Breast-Cancer//test.csv ")
train_data.dropna(axis=0, how='any')
test_data.dropna(axis=0, how='any')
train_data.head()


# In[59]:


#类别数据转化为数值数据，因为是感知机，所以转化为1和-1，而非0和1
diagnosis_mapping = {'M': -1,'B': 1}
train_data['diagnosis'] = train_data['diagnosis'].map(diagnosis_mapping)
test_data['diagnosis'] = test_data['diagnosis'].map(diagnosis_mapping)
test_data.head()


# 感知器训练
# 关键点：
# 1.数据中的第一列ID，虽为数值信息，但是并非特征，在训练的时候需要去掉；
# 2.为了运算方便，将数据转化为矩阵；

# In[71]:


def perceptron(data, label, num):
    print('感知机训练：')
    #将数据转换成矩阵形式（在机器学习中因为通常都是向量的运算，转换称矩阵形式方便运算）
    dataMat = np.mat(data)
    #将标签转换成矩阵，之后转置(.T为转置)。
    labelMat = np.mat(label).T
    #获取数据矩阵的大小，为m*n
    m, n = np.shape(dataMat)
    #创建初始权重w，初始值全为0。
    #样本长度保持一致
    w = np.zeros((1, np.shape(dataMat)[1]))
    #初始化偏置b为0
    b = 0
    #初始化步长，也就是梯度下降过程中的n，控制梯度下降速率
    h = 0.0001
    #进行num次迭代计算
    res_b=[]
    for k in range(num):
        #对于每一个样本进行随机梯度下降。
        for i in range(m):
            #获取当前样本的向量
            xi = dataMat[i]
            #获取当前样本所对应的标签
            yi = labelMat[i]
            #判断是否是误分类样本
            #误分类样本特诊为： -yi(w*xi+b)>=0
            if -1 * yi * (w * xi.T + b) >= 0:
                #对于误分类样本，进行梯度下降，更新w和b
                w = w + h *  yi * xi
                b = b + h * yi
                res_b.append(b)
        #打印训练进度
        print('Round %d:%d training' % (k, num))
    print('截距b的训练进程为:',res_b[::50])
    #返回训练完的w、b
    return w, b


# In[72]:


def test(data, label, w, b):
    print('测试：')
    #将数据集转换为矩阵形式方便运算
    dataMat = np.mat(data)
    labelMat = np.mat(label).T
    #获取测试数据集矩阵的大小
    m, n = np.shape(dataMat)
    #错误样本数计数
    errorCnt = 0
    #遍历所有测试样本
    for i in range(m):
        #获得单个样本向量
        xi = dataMat[i]
        #获得该样本标记
        yi = labelMat[i]
        #获得运算结果
        result = -1 * yi * (w * xi.T + b)
        #如果-yi(w*xi+b)>=0，说明该样本被误分类，错误样本数加一
        if result >= 0: errorCnt += 1
    #正确率 = 1 - （样本分类错误数 / 样本总数）
    accruRate = 1 - (errorCnt / m)
    #返回正确率
    return accruRate


# In[74]:


start = time.time()
w, b = perceptron(train_data[2:], train_data['diagnosis'], num = 50)
#进行测试，获得正确率
accruRate = test(test_data[2:], test_data['diagnosis'], w, b)
#获取当前时间，作为结束时间
end = time.time()
#显示正确率
print('准确率：:', accruRate)
#显示用时时长
print('耗时:', end - start)

