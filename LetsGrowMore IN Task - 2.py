#!/usr/bin/env python
# coding: utf-8

# # LetsGrow More: Data Science Internship
# ## Task 2 : (Intermediate Level Task) Prediction Using Decision Tree Algorithm.
# ## Intern name : Koli Dhanashree Suresh

# ### Step 1 : Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


data = pd.read_csv("Iris.csv")
data


# In[3]:


data.head()


# In[4]:


data.tail()


# ### Step 2 : Exploratory Data Analysis

# In[5]:


data.shape


# In[6]:


data.info()


# In[7]:


data.describe()


# In[8]:


data.isnull().sum()


# In[9]:


data.isnull().values.any()


# In[10]:


# unique value in each columns
for i in data.columns:
    print(i, "\t\t",len(data[i].unique()))


# In[11]:


list_columns=data.columns
list_columns


# In[12]:


features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
x= data.loc[:, features].values
print(x)


# In[13]:


y=data.Species
print(y)


# In[14]:


data['Species'].unique()


# In[15]:


data['Species'].value_counts()


# ### Step 3 : Data Visualization

# In[16]:


sns.pairplot(data)


# In[17]:


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


# In[18]:


sns.violinplot(y='Species', x='SepalLengthCm', data=data, inner='quartile')
sns.violinplot(y='Species', x='SepalWidthCm', data=data, inner='quartile')
sns.violinplot(y='Species', x='PetalLengthCm', data=data, inner='quartile')
sns.violinplot(y='Species', x='PetalWidthCm', data=data, inner='quartile')
plt.show()


# In[19]:


fig,(ax1,ax2) = plt.subplots(ncols=2, figsize=(16,5))
sns.scatterplot(x='SepalLengthCm', y='PetalLengthCm', data=data, hue='Species',ax=ax1, s=300, marker='o')
sns.scatterplot(x='SepalWidthCm', y='PetalWidthCm', data=data, hue='Species',ax=ax2, s=300, marker='o')


# In[20]:


plt.figure(figsize=(10,5))
sns.heatmap(data.describe(),annot=True,fmt='.2f',cmap='rainbow')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("visualize average,number0f,min,max,std,Queartile",fontsize=16)


# In[21]:


# Relationship between the data
data.corr()


# In[22]:


plt.figure(figsize=(10,5))
sns.heatmap(data.corr(),annot=True,fmt='.2f',cmap='rainbow')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("visualize average,number0f,min,max,std,Queartile",fontsize=18)


# ### Step 4 : Fitting Decision Tree Classifier

# In[26]:


features = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
x = data.loc[:, features].values  #defining the feature matrix
y = data.Species
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=0)


# In[27]:


dtree = DecisionTreeClassifier()
dtree.fit(x_train, y_train)


# In[28]:


from sklearn import tree
feature_name = ['sepal length)cm','sepal width(cm)','petal length(cm)','petal width(cm)']
class_name = data.Species.unique()
plt.figure(figsize=(15,10))
tree.plot_tree(dtree, filled = True, feature_names=feature_name, class_names=class_name)


# In[31]:


from sklearn.metrics import confusion_matrix,accuracy_score
y_pred = dtree.predict(x_test)
y_pred
score=accuracy_score(y_test,y_pred)
print("Accuracy of Model:", score)


# In[38]:


from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
def report(model):
    preds=model.predict(x_test)
    print(classification_report(preds,y_test))
    plot_confusion_matrix(model,x_test,y_test,cmap='nipy_spectral',colorbar=True)
print('Decision Tree Classifier')
report(dtree)
print(f'Accuracy: {round(score*100,2)}%')
confusion_matrix(y_test, y_pred)
dtree.predict([[5, 3.6, 1.4, 0.2]])
dtree.predict([[9, 3.1, 5, 1.5]])
dtree.predict([[4.1, 3.0, 5.1, 1.8]])


# ## Thank You...
