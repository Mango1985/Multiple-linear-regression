#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.regressionplots import influence_plot
import statsmodels.formula.api as smf
import numpy as np


# In[3]:


import warnings
warnings .filterwarnings('ignore')


# In[4]:


spf=pd.read_csv("50_Startups.csv")
spf


# In[5]:


spf.head()


# In[6]:


spf.info()


# In[7]:


spf.isna().sum()


# In[8]:


spf.corr()


# In[9]:


sns.set_style(style='darkgrid')
sns.pairplot(spf)


# In[10]:


import statsmodels.formula.api as smf 


# In[17]:


spf.rename(columns={'R&D Spend': 'rnd'}, inplace=True)
spf.rename(columns={'Administration': 'adm'}, inplace=True)
spf.rename(columns={'Marketing Spend': 'mks'}, inplace=True)
spf.rename(columns={'State': 'ste'}, inplace=True)
spf.rename(columns={'Profit': 'pr'}, inplace=True)
spf.head()


# In[19]:


model = smf.ols('pr~adm+rnd+mks',data=spf).fit()


# In[21]:


model.params


# In[23]:


print(model.tvalues, '\n', model.pvalues)


# In[24]:


(model.rsquared,model.rsquared_adj)

