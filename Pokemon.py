#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#IST 652 
#Bradley Choi
#Jen Yeu Lo


# In[4]:


#Pands for mangaging datasets
import pandas as pd


# In[5]:


#Matplotlib for additioanl customisation
from matplotlib import pyplot as plt


# In[6]:


#Seaborn for plotting and styling
import seaborn as sns


# In[10]:


# Read dataset
df = pd.read_csv('Pokemon.csv', index_col=0, encoding = "ISO-8859-1")


# In[12]:


print(df)


# In[13]:


#Boxplot with the Pokemon's combat statistics
sns.boxplot(data=df)


# In[17]:


#Dropping the unnecessary variables 
new_df = df.drop(['Total', 'Sp. Atk', 'Sp. Def', 'Stage', 'Legendary'], axis=1)


# In[18]:


#A cleaned up boxplot
sns.boxplot(data=new_df)


# In[ ]:


#The second plot is about the histrograms demonstrating distribution of HP (Health Points)


# In[19]:


sns.distplot(df.HP)


# In[ ]:


#The third plot is scatter plot looking at Health Points and Speed


# In[39]:


# Scatterplot arguments
sns.lmplot(x='HP', y='Speed', data=df,
           fit_reg=False, # No regression line
           hue='Legendary')   # Color by LEGENDARY stage


# In[42]:


# Scatterplot arguments
sns.lmplot(x='Defense', y='Attack', data=df,
           fit_reg=False, # No regression line
           hue='Legendary')   # Color by LEGENDARY stage


# In[38]:


#Barplot
#Distribution of the categorical data 
#Looking at the different kinds of Pokemon 
#i.e. (Water, Fire, Electric, Fire, Ice, Rock, Ground, Poison, and so forth)

#Barplot
sns.countplot(x='Type 1', data=df)
plt.xticks(rotation=90)

