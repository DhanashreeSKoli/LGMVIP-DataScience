#!/usr/bin/env python
# coding: utf-8

# # LetsGrowMore : Data Science Internship
# ## Task 1 : (Intermediate  Level Task) Exploratory Data Analysis of Terrorism Dataset.
# ## Intern name : Koli Dhanashree Suresh

# ### Step 1 : Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# ### Step 2 : Loading Dataset

# In[2]:


data=pd.read_csv('globalterrorismdb_0718dist.csv',encoding='latin1',low_memory=False)
data.head()


# ### Step 3 : Exploratory Data Analysis

# In[3]:


data.shape


# In[4]:


data=data[["iyear","imonth","iday","country_txt","region_txt","provstate","city",
           "latitude","longitude","location","summary","attacktype1_txt","targtype1_txt",
           "gname","motive","weaptype1_txt","nkill","nwound","addnotes"]]
data.head()


# In[5]:


data.rename(columns={"iyear":"Year","imonth":"Month","iday":"Day","country_txt":"Country",
                    "region_txt":"Region","provstate":"Province/State","city":"City",
                    "latitude":"Latitude","longitude":"Longitude","location":"Location",
                    "summary":"Summary","attacktype1_txt":"Attack Type","targtype1_txt":"Target Type",
                    "gname":"Group Name","motive":"Motive","weaptype1_txt":"Weapon Type",
                    "nkill":"Killed","nwound":"Wounded","addnotes":"Add Notes"},inplace=True)
data.head()


# In[6]:


data.info()


# In[7]:


data.isnull().sum()


# In[8]:


data["Killed"] = data["Killed"].fillna(0)
data["Wounded"] = data["Wounded"].fillna(0)
data["Casualty"] = data["Killed"]+data["Wounded"]


# In[9]:


data.describe()


# #### The data consists information from Year 1970 to 2017.
# #### The maximum peoples killed in attack were 1570.

# ### Step 4 : Data Visualization

# #### Regionwise Analysis

# In[10]:


data.Region.value_counts()


# In[11]:


plt.figure(figsize=(22,10))
plt.bar(x=data.Region.value_counts().index, height=data.Region.value_counts().values)
plt.title('Regionwise terrorist attack')
plt.xlabel('Region')
plt.ylabel('No. of terrorist attack')


# In[12]:


plt.subplots(figsize=(10,6))
sns.countplot('Year',data=data,palette='spring_r',edgecolor=sns.color_palette('dark',10))
plt.xticks(rotation=90)
plt.title('Number Of Terrorist Activities Each Year')
plt.show()


# #### here The graph shows the number of Attacks in each year from 1970 to 1992 and here we observed that the number of global terrorism attacks are continuously increasing from 1988. The Graph shows 1992 is the most unlucky year because the most number of atacks took place in the year 1992.

# In[13]:


data.Year.plot(kind = 'hist', color = 'pink', bins=range(1970, 2018), figsize = (12,5),grid=True)
plt.xticks(range(1970, 2018), rotation=90, fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("Years", fontsize=15)
plt.ylabel("Number of Attacks", fontsize=15)
plt.title("Number of Attacks By Year", fontsize=16)
plt.show()


# In[14]:


killed=data["Region"].value_counts(dropna=False).sort_index().to_frame().reset_index().rename(columns={"index":"Region","Region":"Killed"}).set_index("Region")
killed.head()


# In[15]:


killed.plot(kind="bar",color="purple",figsize=(12,6),fontsize=15)
plt.title('Terrorist attack',fontsize=12)
plt.xlabel('Region',fontsize=12)
plt.ylabel('No of peoples killed',fontsize=12)
plt.show()


# In[16]:


attacks=data["Year"].value_counts(dropna=False).sort_index().to_frame().reset_index().rename(columns={"index":"Year","Year":"Attack"}).set_index("Year")
attacks.head()


# In[17]:


attacks.plot(kind='bar', color='green', figsize=(15,6), fontsize=15)
plt.title('Terrorost attack', fontsize=12)
plt.xlabel('Year', fontsize=12)
plt.ylabel('No. of terrorist attack', fontsize=12)


# In[18]:


country =data['Country'].value_counts().head(10)
country


# In[19]:


country.plot(kind="bar", color='grey', figsize=(12,5), fontsize=15)
plt.title('Terrorost attack', fontsize=12)
plt.xlabel('Country', fontsize=12)
plt.ylabel('No. of terrorist attack', fontsize=12)
plt.show()


# In[20]:


count = data[["Country","Casualty"]].groupby("Country").sum().sort_values(by="Casualty", ascending=False)
count.head(10)


# In[21]:


count[:10].plot(kind="bar",color="blue",figsize=(15,6),fontsize=15)
plt.title('Terrorost attack', fontsize=12)
plt.xlabel('Country', fontsize=12)
plt.ylabel('No. of Casualties', fontsize=12)
plt.show()


# In[22]:


countk=data[["Country","Killed"]].groupby("Country").sum().sort_values(by="Killed",ascending=False)
countk.head(10)


# In[23]:


countw=data[["Country","Wounded"]].groupby("Country").sum().sort_values(by="Wounded",ascending=False)
countw.head(10)


# In[24]:


fig=plt.figure()
ax0=fig.add_subplot(1,2,1)
ax1=fig.add_subplot(1,2,2)

#Killed
countk[:10].plot(kind="bar",color="pink",figsize=(15,6),ax=ax0)
ax0.set_title("People Killed in each country")
ax0.set_xlabel("Country")
ax0.set_ylabel("No. of people killed")

#Wounded
countw[:10].plot(kind="bar",color="pink",figsize=(15,6),ax=ax1)
ax1.set_title("People wounded in each country")
ax1.set_xlabel("Country")
ax1.set_ylabel("No. of people wounded")

plt.show()


# In[25]:


at=data["Attack Type"].value_counts()
at


# In[31]:


at.plot(kind="bar", color="blue", figsize=(15,6))
plt.title("Types of attacks", fontsize=12)
plt.xlabel("Attack Types", fontsize=12)
plt.xticks(fontsize=13)
plt.ylabel("No. of Attacks", fontsize=12)
plt.show()


# In[32]:


tt = data["Target Type"].value_counts()
tt


# In[33]:


tt.plot(kind="bar",color="blue",figsize=(15,6))
plt.title("Types of target", fontsize=12)
plt.xlabel("Target Types", fontsize=12)
plt.xticks(fontsize=13)
plt.ylabel("No. of Targets", fontsize=12)
plt.show()


# ## Thank You...

# In[ ]:




