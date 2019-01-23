
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

# In[ ]:


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


# 结果：
# 感知机训练：
# Round 0:50 training
# Round 1:50 training
# Round 2:50 training
# Round 3:50 training
# Round 4:50 training
# Round 5:50 training
# Round 6:50 training
# Round 7:50 training
# Round 8:50 training
# Round 9:50 training
# Round 10:50 training
# Round 11:50 training
# Round 12:50 training
# Round 13:50 training
# Round 14:50 training
# Round 15:50 training
# Round 16:50 training
# Round 17:50 training
# Round 18:50 training
# Round 19:50 training
# Round 20:50 training
# Round 21:50 training
# Round 22:50 training
# Round 23:50 training
# Round 24:50 training
# Round 25:50 training
# Round 26:50 training
# Round 27:50 training
# Round 28:50 training
# Round 29:50 training
# Round 30:50 training
# Round 31:50 training
# Round 32:50 training
# Round 33:50 training
# Round 34:50 training
# Round 35:50 training
# Round 36:50 training
# Round 37:50 training
# Round 38:50 training
# Round 39:50 training
# Round 40:50 training
# Round 41:50 training
# Round 42:50 training
# Round 43:50 training
# Round 44:50 training
# Round 45:50 training
# Round 46:50 training
# Round 47:50 training
# Round 48:50 training
# Round 49:50 training
# 截距b的训练进程为: [matrix([[-0.0001]]), matrix([[0.0039]]), matrix([[0.0003]]), matrix([[1.e-04]]), matrix([[-0.0025]]), matrix([[-0.0025]]), matrix([[-0.0035]]), matrix([[-0.0059]]), matrix([[-0.0065]]), matrix([[-0.0097]]), matrix([[-0.0073]]), matrix([[-0.0119]]), matrix([[-0.0113]]), matrix([[-0.0149]]), matrix([[-0.0129]]), matrix([[-0.0159]]), matrix([[-0.0163]]), matrix([[-0.0187]]), matrix([[-0.0195]]), matrix([[-0.0197]]), matrix([[-0.0229]]), matrix([[-0.0231]]), matrix([[-0.0269]]), matrix([[-0.0241]]), matrix([[-0.0285]]), matrix([[-0.0279]]), matrix([[-0.0313]]), matrix([[-0.0299]]), matrix([[-0.0321]]), matrix([[-0.0335]]), matrix([[-0.0349]]), matrix([[-0.0367]]), matrix([[-0.0359]]), matrix([[-0.0395]]), matrix([[-0.0399]]), matrix([[-0.0437]]), matrix([[-0.0409]]), matrix([[-0.0447]]), matrix([[-0.0447]]), matrix([[-0.0475]]), matrix([[-0.0467]]), matrix([[-0.0485]]), matrix([[-0.0501]]), matrix([[-0.0513]]), matrix([[-0.0539]]), matrix([[-0.0521]]), matrix([[-0.0561]]), matrix([[-0.0563]]), matrix([[-0.0599]]), matrix([[-0.0575]]), matrix([[-0.0609]]), matrix([[-0.0611]]), matrix([[-0.0637]]), matrix([[-0.0639]]), matrix([[-0.0647]]), matrix([[-0.0673]]), matrix([[-0.0679]]), matrix([[-0.0711]]), matrix([[-0.0687]]), matrix([[-0.0731]]), matrix([[-0.0729]]), matrix([[-0.0765]]), matrix([[-0.0743]]), matrix([[-0.0773]]), matrix([[-0.0775]]), matrix([[-0.0801]]), matrix([[-0.0807]]), matrix([[-0.0811]]), matrix([[-0.0843]]), matrix([[-0.0845]]), matrix([[-0.0881]]), matrix([[-0.0855]]), matrix([[-0.0899]]), matrix([[-0.0891]]), matrix([[-0.0927]]), matrix([[-0.0911]]), matrix([[-0.0937]]), matrix([[-0.0945]]), matrix([[-0.0965]]), matrix([[-0.0977]]), matrix([[-0.0973]]), matrix([[-0.1007]]), matrix([[-0.1011]]), matrix([[-0.1051]]), matrix([[-0.1021]]), matrix([[-0.1061]]), matrix([[-0.1059]]), matrix([[-0.1089]]), matrix([[-0.1079]]), matrix([[-0.1099]]), matrix([[-0.1115]]), matrix([[-0.1127]]), matrix([[-0.1149]]), matrix([[-0.1139]]), matrix([[-0.1175]]), matrix([[-0.1179]]), matrix([[-0.1217]]), matrix([[-0.1187]]), matrix([[-0.1225]]), matrix([[-0.1227]]), matrix([[-0.1253]]), matrix([[-0.1247]]), matrix([[-0.1263]]), matrix([[-0.1281]]), matrix([[-0.1291]]), matrix([[-0.1319]]), matrix([[-0.1301]]), matrix([[-0.1341]]), matrix([[-0.1343]]), matrix([[-0.1379]]), matrix([[-0.1355]]), matrix([[-0.1389]]), matrix([[-0.1391]]), matrix([[-0.1417]]), matrix([[-0.1417]]), matrix([[-0.1425]]), matrix([[-0.1451]]), matrix([[-0.1457]]), matrix([[-0.1489]]), matrix([[-0.1465]]), matrix([[-0.1511]]), matrix([[-0.1505]]), matrix([[-0.1541]]), matrix([[-0.1521]]), matrix([[-0.1551]]), matrix([[-0.1555]]), matrix([[-0.1579]]), matrix([[-0.1587]]), matrix([[-0.1591]]), matrix([[-0.1621]]), matrix([[-0.1623]]), matrix([[-0.1659]]), matrix([[-0.1633]]), matrix([[-0.1677]]), matrix([[-0.1669]]), matrix([[-0.1705]]), matrix([[-0.1691]]), matrix([[-0.1715]]), matrix([[-0.1725]]), matrix([[-0.1743]]), matrix([[-0.1757]]), matrix([[-0.1753]]), matrix([[-0.1787]]), matrix([[-0.1789]]), matrix([[-0.1829]]), matrix([[-0.1799]]), matrix([[-0.1841]]), matrix([[-0.1837]]), matrix([[-0.1869]]), matrix([[-0.1857]]), matrix([[-0.1877]]), matrix([[-0.1893]]), matrix([[-0.1905]]), matrix([[-0.1925]]), matrix([[-0.1915]]), matrix([[-0.1951]]), matrix([[-0.1955]]), matrix([[-0.1993]]), matrix([[-0.1965]]), matrix([[-0.2003]]), matrix([[-0.2005]]), matrix([[-0.2031]]), matrix([[-0.2025]]), matrix([[-0.2043]]), matrix([[-0.2059]]), matrix([[-0.2071]]), matrix([[-0.2095]]), matrix([[-0.2079]]), matrix([[-0.2119]]), matrix([[-0.2121]]), matrix([[-0.2157]]), matrix([[-0.2131]]), matrix([[-0.2167]]), matrix([[-0.2169]]), matrix([[-0.2195]]), matrix([[-0.2195]]), matrix([[-0.2205]]), matrix([[-0.2227]]), matrix([[-0.2235]]), matrix([[-0.2265]]), matrix([[-0.2243]]), matrix([[-0.2285]]), matrix([[-0.2285]]), matrix([[-0.2321]]), matrix([[-0.2297]]), matrix([[-0.2329]]), matrix([[-0.2331]]), matrix([[-0.2357]]), matrix([[-0.2361]]), matrix([[-0.2367]]), matrix([[-0.2397]]), matrix([[-0.2399]]), matrix([[-0.2435]]), matrix([[-0.2409]]), matrix([[-0.2455]]), matrix([[-0.2447]]), matrix([[-0.2483]]), matrix([[-0.2465]]), matrix([[-0.2495]]), matrix([[-0.2497]]), matrix([[-0.2523]]), matrix([[-0.2529]]), matrix([[-0.2531]]), matrix([[-0.2563]]), matrix([[-0.2565]]), matrix([[-0.2603]]), matrix([[-0.2575]]), matrix([[-0.2619]]), matrix([[-0.2613]]), matrix([[-0.2647]]), matrix([[-0.2633]]), matrix([[-0.2657]]), matrix([[-0.2667]]), matrix([[-0.2685]]), matrix([[-0.2699]]), matrix([[-0.2695]]), matrix([[-0.2729]]), matrix([[-0.2731]])]
# 测试：
# 准确率：: 0.7755102040816326
# 耗时: 0.745969295501709
