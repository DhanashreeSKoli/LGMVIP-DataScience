#!/usr/bin/env python
# coding: utf-8

# # LetsGrowMore : Data Science Internship
# ## Task 1 : (Beginner Level Task) Iris flowers classification ML project
# ## Intern Name : Koli Dhanashree Suresh

# ### Step 1 : Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC ,LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,  classification_report,confusion_matrix


# ### Step 2 : Importing Dataset

# In[3]:


data = pd.read_csv("Iris.csv")
data


# ### Step 3 : Exploratory Data Analysis

# In[4]:


#for first 5 rows
data.head()


# In[5]:


#for last 5 rows
data.tail()


# In[6]:


#Dimension of dataset
data.shape


# In[7]:


#Discriptive statistics
data.describe()


# In[8]:


#Information os dataset
data.info()


# ### In this Iris dataset have one target column which is "Species", it is categorical. It has 4 variables of iris and none ofthe cells have null values.

# ### Step 4 : Data Visualization

# In[11]:


sns.pairplot(data,hue="Species")


# In[13]:


sns.kdeplot(data=data, x="SepalLengthCm", hue="Species", multiple="stack")
plt.show()
sns.kdeplot(data=data, x="PetalLengthCm", hue="Species", multiple="stack")
plt.show()
sns.kdeplot(data=data, x="SepalWidthCm", hue="Species", multiple="stack")
plt.show()
sns.kdeplot(data=data, x="PetalWidthCm", hue="Species", multiple="stack")
plt.show()


# In[14]:


fig,(ax1,ax2)=plt.subplots(ncols=2,figsize=(16,5))
sns.scatterplot(x='SepalLengthCm',y='PetalLengthCm',data=data,hue='Species',ax=ax1)
sns.scatterplot(x='SepalWidthCm',y='PetalWidthCm',data=data,hue='Species',ax=ax2)


# In[15]:


sns.violinplot(y='Species', x='SepalLengthCm', data=data, inner='quartile')
plt.show()
sns.violinplot(y='Species', x='SepalWidthCm', data=data, inner='quartile')
plt.show()
sns.violinplot(y='Species', x='PetalLengthCm', data=data, inner='quartile')
plt.show()
sns.violinplot(y='Species', x='PetalWidthCm', data=data, inner='quartile')
plt.show()


# In[16]:


data.corr()


# #### Using heatmap to know the correlation among the variable

# In[23]:


plt.figure(figsize=(10,5))
sns.heatmap(data.corr(), annot=True,cmap='Blues_r')
plt.show()


# #### From above heatmap, it is seen that petal length and petal width are highly correlated

# In[27]:


plt.figure(figsize=(7,5))
sns.boxplot(x='Species', y='SepalLengthCm', data=data)
plt.show()


# In[32]:


plt.figure(figsize=(8,5))
plt.subplot(2,2,1)
sns.boxplot(x=data.Species, y=data.SepalLengthCm)
plt.subplot(2,2,2)
sns.boxplot(x=data.Species, y=data.SepalWidthCm)
plt.subplot(2,2,3)
sns.boxplot(x=data.Species, y=data.PetalLengthCm)
plt.subplot(2,2,4)
sns.boxplot(x=data.Species, y=data.PetalWidthCm)


# In[29]:


#Removing the 'Id' column
data.drop('Id', axis=1, inplace=True)
data.head()
data.groupby('Species').mean().plot.bar()
plt.show()


# In[30]:


plt.figure(figsize=(8,8))
plt.subplot(2,2,1)
sns.distplot(data['PetalLengthCm'])
plt.subplot(2,2,2)
sns.distplot(data['PetalWidthCm'])
plt.subplot(2,2,3)
sns.distplot(data['SepalLengthCm'])
plt.subplot(2,2,4)
sns.distplot(data['SepalWidthCm'])


# #### From above boxplot in 'Iris-virginica' one outlier is detected.

# In[34]:


sns.pairplot(data.corr())
plt.show()


# ### Step 5 : Model Building for Classification

#  ### 1.Linear Regression

# In[35]:


x = data[['SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = data['SepalLengthCm']


# In[38]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.65, random_state=101)
model = LinearRegression()
model.fit(x_train, y_train)
y_pred=model.predict(x_test)


# In[39]:


from sklearn.metrics import mean_squared_error, mean_absolute_error
print('mean_squared_error : ', mean_squared_error(y_test, y_pred))
print('mean_absolute_error : ', mean_absolute_error(y_test, y_pred))


# In[40]:


print('coefficient of determination: ', model.score(x_train,y_train))
print('intercept:', model.intercept_)
print('slope:', model.coef_)


# In[42]:


model.predict([[5.1, 2.5, 1.1]])


# In[43]:


model.predict([[7.5, 3.0, 1.8]])


# In[44]:


model.predict([[4.6, 3.5, 0.1]])


# ### 2. Linear SVC

# In[46]:


X = data.drop(columns=['Species'])
Y = data['Species']
x_train , x_test , y_train , y_test = train_test_split(X , Y , test_size = 0.2)


# In[47]:


l_svc=LinearSVC()


# In[48]:


l_svc.fit(x_train,y_train)


# In[49]:


y_pred=l_svc.predict(x_test)
score=accuracy_score(y_test,y_pred)


# In[50]:


score


# In[51]:


def report(model):
    preds=model.predict(x_test)
    print(classification_report(preds,y_test))


# In[52]:


print('Linear SVC')
report(l_svc)
print(f'Accuracy: {round(score*100,2)}%')


# ## Thank You...
