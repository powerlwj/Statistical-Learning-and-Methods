
# coding: utf-8

# In[6]:


#使用朴素贝叶斯算法对Voice Gender数据集进行分析，思路：对于给出的待分类项，求解在此项出现的条件下各个类别出现的概率，哪个最大，就认为此待分类项属于哪个类别。
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import time
get_ipython().run_line_magic('matplotlib', 'inline')


# 数据集介绍：
# 语音性别
# 通过语音和语音分析进行性别识别，该数据库的创建是为了根据语音和语音的声学特性识别男性或女性的声音。
# 特征如下：
# 
# meanfreq: mean frequency (in kHz)
# sd: standard deviation of frequency
# median: median frequency (in kHz)
# Q25: first quantile (in kHz)
# Q75: third quantile (in kHz)
# IQR: interquantile range (in kHz)
# skew: skewness (see note in specprop description)
# kurt: kurtosis (see note in specprop description)
# sp.ent: spectral entropy
# sfm: spectral flatness
# mode: mode frequency
# centroid: frequency centroid (see specprop)
# peakf: peak frequency (frequency with highest energy)
# meanfun: average of fundamental frequency measured across acoustic signal
# minfun: minimum fundamental frequency measured across acoustic signal
# maxfun: maximum fundamental frequency measured across acoustic signal
# meandom: average of dominant frequency measured across acoustic signal
# mindom: minimum of dominant frequency measured across acoustic signal
# maxdom: maximum of dominant frequency measured across acoustic signal
# dfrange: range of dominant frequency measured across acoustic signal
# modindx: modulation index. Calculated as the accumulated absolute difference between adjacent measurements of fundamental frequencies divided by the frequency range
# 
# label: male or female

# In[138]:


#数据预览
data = pd.read_csv("F://Machine Learning//ML\Data//Datasets//voicegender//train_data.csv")
data.head()


# In[139]:


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
        if curLine[21]=='female':
            labelArr.append(0)
        else:
            labelArr.append(1)
        #存放标记
        dataArr.append([float(num) for num in curLine[1:21]])
    #返回data和label
    return dataArr, labelArr


# In[140]:


trainDataArr, trainLabelArr = loadData('F://Machine Learning//ML//Data//Datasets//voicegender//train_data.csv')


# In[141]:


trainLabelArr


# In[146]:


def getAllProbability(trainDataArr, trainLabelArr):
    '''
    通过训练集计算先验概率分布和条件概率分布
    :param trainDataArr: 训练数据集
    :param trainLabelArr: 训练标记集
    :return: 先验概率分布和条件概率分布
    '''
    featureNum = 20
    #设置类别数目，共2个类别
    classNum = 2
    Py = np.zeros((classNum, 1))
    #对每个类别进行一次循环，分别计算它们的先验概率分布
    #计算公式为书中"4.2节 朴素贝叶斯法的参数估计 公式4.8"
    #Py=set()
    #for i in trainLabelArr:
        #res_set.add(i)
        #a=np.sum(np.mat(trainLabelArr) == i)
        #Py.add(((np.sum(np.mat(trainLabelArr) == i))) / (len(trainLabelArr)))
    #print(a)
    for i in range(classNum):
        Py[i] = ((np.sum(np.mat(trainLabelArr) == i)) + 1) / (len(trainLabelArr) + 2)
    #此外连乘项通过log以后，可以变成各项累加，简化了计算。
    #在似然函数中通常会使用log的方式进行处理
    #Py=list(Py)
    print('>>>>>>',Py)
    Py = np.log(Py)
    #计算条件概率 Px_y=P（X=x|Y = y）
    #计算条件概率分成了两个步骤，下方第一个大for循环用于累加，参考书中“4.2.3 贝叶斯估计 式4.10”，下方第一个大for循环内部是
    #用于计算式4.10的分子，至于分子的+1以及分母的计算在下方第二个大For内
    #初始化为全0矩阵，用于存放所有情况下的条件概率
    Px_y = np.zeros((classNum, featureNum, 130962))
    #对标记集进行遍历
    for i in range(len(trainLabelArr)):
        #获取当前循环所使用的标记
        label = trainLabelArr[i]
        #获取当前要处理的样本
        x = trainDataArr[i]
        #对该样本的每一维特诊进行遍历
        for j in range(featureNum):
            #在矩阵中对应位置加1
            #这里还没有计算条件概率，先把所有数累加，全加完以后，在后续步骤中再求对应的条件概率
            label=int(label)
            j=int(j)
            x[j]=int(x[j])
            Px_y[label][j][x[j]] += 1
    #第二个大for，计算式4.10的分母，以及分子和分母之间的除法
    #循环每一个标记（共10个）
    for label in range(classNum):
        #循环每一个标记对应的每一个特征
        for j in range(featureNum):
            #获取y=label，第j个特诊为0的个数
            label=int(label)
            j=int(j)
            Px_y0 = Px_y[label][j][0]
            #获取y=label，第j个特诊为1的个数
            Px_y1 = Px_y[label][j][1]
            #对式4.10的分子和分母进行相除，再除之前依据贝叶斯估计，分母需要加上2（为每个特征可取值个数）
            #分别计算对于y= label，x第j个特征为0和1的条件概率分布
            Px_y[label][j][0] = np.log((Px_y0 + 1) / (Px_y0 + Px_y1 + 2))
            Px_y[label][j][1] = np.log((Px_y1 + 1) / (Px_y0 + Px_y1 + 2))
    #返回先验概率分布和条件概率分布
    return Py, Px_y


# In[143]:


def NaiveBayes(Py, Px_y, x):
    '''
    通过朴素贝叶斯进行概率估计
    :param Py: 先验概率分布
    :param Px_y: 条件概率分布
    :param x: 要估计的样本x
    :return: 返回所有label的估计概率
    '''
    #设置特征数目
    featrueNum = 20
    #设置类别数目
    classNum = 2
    #建立存放所有标记的估计概率数组
    P = [0] * classNum
    #对于每一个类别，单独估计其概率
    for i in range(classNum):
        #初始化sum为0，sum为求和项。
        #在训练过程中对概率进行了log处理，所以这里原先应当是连乘所有概率，最后比较哪个概率最大
        #但是当使用log处理时，连乘变成了累加，所以使用sum
        sum = 0
        #获取每一个条件概率值，进行累加
        for j in range(featrueNum):
            j=int(j)
            x[j]=int(x[j])
            sum += Px_y[i][j][x[j]]
        #最后再和先验概率相加（也就是式4.7中的先验概率乘以后头那些东西，乘法因为log全变成了加法）
        P[i] = sum + Py[i]
    #max(P)：找到概率最大值
    #P.index(max(P))：找到该概率最大值对应的所有（索引值和标签值相等）
    return P.index(max(P))


# In[144]:


def test(Py, Px_y, testDataArr, testLabelArr):
    '''
    对测试集进行测试
    :param Py: 先验概率分布
    :param Px_y: 条件概率分布
    :param testDataArr: 测试集数据
    :param testLabelArr: 测试集标记
    :return: 准确率
    '''
    #错误值计数
    errorCnt = 0
    #循环遍历测试集中的每一个样本
    for i in range(len(testDataArr)):
        #获取预测值
        presict = NaiveBayes(Py, Px_y, testDataArr[i])
        #与答案进行比较
        if presict != testLabelArr[i]:
            #若错误  错误值计数加1
            errorCnt += 1
    #返回准确率
    return 1 - (errorCnt / len(testDataArr))


# In[147]:


start = time.time()
# 获取训练集
print('start read transSet')
trainDataArr, trainLabelArr = loadData('F://Machine Learning//ML//Data//Datasets//voicegender//train_data.csv')
# 获取测试集
print('start read testSet')
testDataArr, testLabelArr = loadData('F://Machine Learning//ML//Data//Datasets//voicegender//test_data.csv')
#开始训练，学习先验概率分布和条件概率分布
print('start to train')
Py, Px_y = getAllProbability(trainDataArr, trainLabelArr)
#使用习得的先验概率分布和条件概率分布对测试集进行测试
print('start to test')
accuracy = test(Py, Px_y, testDataArr, testLabelArr)
#打印准确率
print('the accuracy is:', accuracy)
#打印时间
print('time span:', time.time() -start)

