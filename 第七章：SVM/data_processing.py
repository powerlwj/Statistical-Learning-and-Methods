
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import time
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data=pd.read_csv('F://Machine Learning//ML//Data//Datasets//creditcardfraud//train.csv')
data.head()


# In[3]:


count_classes = pd.value_counts(data['Class'], sort = True).sort_index()
count_classes.plot(kind = 'bar')
plt.title("Fraud class histogram")
plt.xlabel("Class")
plt.ylabel("Frequency")


# In[5]:


from sklearn.preprocessing import StandardScaler

data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
data = data.drop(['Time','Amount'],axis=1)
data.head()


# In[6]:


X = data.ix[:, data.columns != 'Class']
y = data.ix[:, data.columns == 'Class']


# In[9]:


y.head()


# 这是一个明显的例子，使用典型的准确度分数来评估我们的分类算法。例如，如果我们只使用多数类为所有记录分配值，我们仍然会有很高的准确性，但是，20万个数据中，非0的数据，也就是欺诈交期数据只有不足500个，这样的话，即使准确率很高，但是其实是有误导性的！

# In[10]:


# Number of data points in the minority class
number_records_fraud = len(data[data.Class == 1])
fraud_indices = np.array(data[data.Class == 1].index)

# Picking the indices of the normal classes
normal_indices = data[data.Class == 0].index

# Out of the indices we picked, randomly select "x" number (number_records_fraud)
random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace = False)
random_normal_indices = np.array(random_normal_indices)

# Appending the 2 indices
under_sample_indices = np.concatenate([fraud_indices,random_normal_indices])

# Under sample dataset
under_sample_data = data.iloc[under_sample_indices,:]

X_undersample = under_sample_data.ix[:, under_sample_data.columns != 'Class']
y_undersample = under_sample_data.ix[:, under_sample_data.columns == 'Class']

# Showing ratio
print("Percentage of normal transactions: ", len(under_sample_data[under_sample_data.Class == 0])/len(under_sample_data))
print("Percentage of fraud transactions: ", len(under_sample_data[under_sample_data.Class == 1])/len(under_sample_data))
print("Total number of transactions in resampled data: ", len(under_sample_data))


# In[13]:


under_sample_data.describe()


# In[14]:


train_data=under_sample_data[:700]
test_data=under_sample_data[700:]


# In[15]:


train_data.to_csv('F://Machine Learning//ML//Data//Datasets//creditcardfraud//train_data.csv')
test_data.to_csv('F://Machine Learning//ML//Data//Datasets//creditcardfraud//test_data.csv')

