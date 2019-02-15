
# coding: utf-8

# 使用K近邻算法对breast cancer数据进行评估，原则：近朱者赤近墨者黑

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import time
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


先对数据进行预览，主要目的是将diagnosis的类别数据转为数字


# In[2]:


data = pd.read_csv("F:\Machine Learning\ML\Data\Datasets\Breast-Cancer//train.csv ")
data.head()


# In[10]:


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
        if curLine[1]=='M':
            labelArr.append(-1)
        else:
            labelArr.append(1)
        #存放标记
        dataArr.append([float(num) for num in curLine[2:]])
    #返回data和label
    return dataArr, labelArr


# In[11]:


def calcDist(x1, x2):
    '''
    计算两个样本点向量之间的距离
    使用的是欧氏距离，即 样本点每个元素相减的平方，再求和，再开方
    欧式举例公式这里不方便写，可以百度或谷歌欧式距离（也称欧几里得距离）
    :param x1:向量1
    :param x2:向量2
    :return:向量之间的欧式距离
    '''
    return np.sqrt(np.sum(np.square(x1 - x2)))
    #曼哈顿距离计算公式
    # return np.sum(x1 - x2)


# In[12]:


def getClosest(trainDataMat, trainLabelMat, x, topK):
    '''
    预测样本x的标记。
    获取方式通过找到与样本x最近的topK个点，并查看它们的标签。
    查找里面占某类标签最多的那类标签
    :param trainDataMat:训练集数据集
    :param trainLabelMat:训练集标签集
    :param x:要预测的样本x
    :param topK:选择参考最邻近样本的数目（样本数目的选择关系到正确率）
    :return:预测的标记
    '''
    #建立一个存放向量x与每个训练集中样本距离的列表
    #列表的长度为训练集的长度，distList[i]表示x与训练集中第
    ## i个样本的距离
    distList = [0] * len(trainLabelMat)
    #遍历训练集中所有的样本点，计算与x的距离
    for i in range(len(trainDataMat)):
        #获取训练集中当前样本的向量
        x1 = trainDataMat[i]
        #计算向量x与训练集样本x的距离
        curDist = calcDist(x1, x)
        #将距离放入对应的列表位置中
        distList[i] = curDist
    #对距离列表进行排序
    #argsort：函数将数组的值从小到大排序后，并按照其相对应的索引值输出
    #例如：
    #   >>> x = np.array([3, 1, 2])
    #   >>> np.argsort(x)
    #   array([1, 2, 0])
    #返回的是列表中从小到大的元素索引值，对于我们这种需要查找最小距离的情况来说很合适
    #array返回的是整个索引值列表，我们通过[:topK]取列表中前topL个放入list中。
    topKList = np.argsort(np.array(distList))[:topK]        #升序排序
    #建立一个长度时的列表，用于选择数量最多的标记
    #3.2.4提到了分类决策使用的是投票表决，topK个标记每人有一票，在数组中每个标记代表的位置中投入
    #自己对应的地方，随后进行唱票选择最高票的标记
    labelList = [0] * 2
    #对topK个索引进行遍历
    for index in topKList:
        #trainLabelMat[index]：在训练集标签中寻找topK元素索引对应的标记
        #int(trainLabelMat[index])：将标记转换为int（实际上已经是int了，但是不int的话，报错）
        #labelList[int(trainLabelMat[index])]：找到标记在labelList中对应的位置
        #最后加1，表示投了一票
        labelList[int(trainLabelMat[index])] += 1
    #max(labelList)：找到选票箱中票数最多的票数值
    #labelList.index(max(labelList))：再根据最大值在列表中找到该值对应的索引，等同于预测的标记
    return labelList.index(max(labelList))


# In[13]:


def test(trainDataArr, trainLabelArr, testDataArr, testLabelArr, topK):
    '''
    测试正确率
    :param trainDataArr:训练集数据集
    :param trainLabelArr: 训练集标记
    :param testDataArr: 测试集数据集
    :param testLabelArr: 测试集标记
    :param topK: 选择多少个邻近点参考
    :return: 正确率
    '''
    print('start test')
    #将所有列表转换为矩阵形式，方便运算
    trainDataMat = np.mat(trainDataArr); trainLabelMat = np.mat(trainLabelArr).T
    testDataMat = np.mat(testDataArr); testLabelMat = np.mat(testLabelArr).T
    #错误值计数
    errorCnt = 0
    #遍历测试集，对每个测试集样本进行测试
    #和return也要相应的更换注释行
    for i in range(len(testDataMat)):
        print('test %d:%d'%(i, len(trainDataArr)))
        #读取测试集当前测试样本的向量
        x = testDataMat[i]
        #获取预测的标记
        y = getClosest(trainDataMat, trainLabelMat, x, topK)
        #如果预测标记与实际标记不符，错误值计数加1
        if y != testLabelMat[i]: errorCnt += 1
    #返回正确率
    return 1 - (errorCnt / len(testDataMat))


# In[17]:


start = time.time()
#获取训练集
trainData, trainLabel = loadData('F:\Machine Learning\ML\Data\Datasets\Breast-Cancer//train.csv ')
#获取测试集
testData, testLabel = loadData('F:\Machine Learning\ML\Data\Datasets\Breast-Cancer//test.csv ')
#计算测试集正确率
accur = test(trainData, trainLabel, testData, testLabel, 10)
#打印正确率
print('accur is:%d'%(accur * 100), '%')
end = time.time()
#显示花费时间
print('time span:', end - start)

