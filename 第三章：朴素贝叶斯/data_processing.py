
# coding: utf-8

# In[ ]:


由于在后续矩阵计算中需要尽可能使整数，而本文中的数据集绝大部分都是小数，直接取整之后不能用，因此考虑先对数据集进行预处理，
此处我们的操作是：对所有数据都乘以100，然后取整，以下是数据处理部分：


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import time
get_ipython().run_line_magic('matplotlib', 'inline')


# In[19]:


train = pd.read_csv("F://Machine Learning//ML\Data//Datasets//voicegender//train.csv")
test=pd.read_csv("F://Machine Learning//ML\Data//Datasets//voicegender//test.csv")


# In[20]:


train2=train[:]*100
test2=test[:]*100


# In[21]:


train2['label']=train['label']
test2['label']=test['label']


# In[22]:


train2.head()


# In[23]:


train2.describe()


# In[25]:


train2.to_csv("F://Machine Learning//ML\Data//Datasets//voicegender//train_data.csv")
test2.to_csv("F://Machine Learning//ML\Data//Datasets//voicegender//test_data.csv")

