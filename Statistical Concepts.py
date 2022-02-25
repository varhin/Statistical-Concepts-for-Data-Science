#!/usr/bin/env python
# coding: utf-8

# In[2]:


# measures of central tendency
#finding mean

values = [1, 2, 3, 4, 5, 6, 7, 8, 9]
mean = sum(values) / len(values)

mean


# In[7]:


import statistics
mean = statistics.mean([1, 2, 3, 4, 5, 6, 7, 8, 9])

mean


# In[9]:


#using numpy
import numpy as np

a = np.array([[1,2,3],[3,4,5],[4,5,6]]) 

np.mean(a)


# In[10]:


#finding median
import numpy as np

a = np.array([[1,2,3],[3,4,5],[4,5,6]]) 

np.median(a)


# In[11]:


import statistics

median = statistics.median([1, 2, 3, 4, 5, 6, 7, 8, 9])

median


# In[14]:


#finding mode
import statistics

mode = statistics.multimode([1, 2, 2, 3, 4, 5, 8, 6, 7, 8, 9])

mode


# In[15]:


#standard deviation
import statistics
std = statistics.stdev([1, 2, 2, 3, 4, 5, 8, 6, 7, 8, 9])

std


# In[16]:


#variance
import statistics
var = statistics.variance([1, 2, 2, 3, 4, 5, 8, 6, 7, 8, 9])
var


# In[18]:


#normal distribution
import numpy
import matplotlib.pyplot as plt

x = numpy.random.normal(5.0, 1.0, 100000)

plt.hist(x, 100)
plt.show()


# In[21]:


#central limit data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def apply_CLT(sample_data,sample_size,total_samples):
  sample_mean = []
  for a in range(total_samples): 
    sample = np.random.choice(sample_data, size = sample_size)
    mean = np.mean(sample)
    sample_mean.append(mean)     
     
  return sample_mean

sample_data = np.random.normal(size=100)


sns.distplot( apply_CLT(sample_data,30,1000) )


# In[ ]:


#Poisson Distribution
from scipy.stats import poisson

poisson.rvs(mu=3, size=10)
array([2, 2, 2, 0, 7, 2, 1, 2, 5, 5])

pmf = poisson.pmf(k=5, mu=3)

cdf = poisson.cdf(k=4, mu=7)

cdf_greater =  1-poisson.cdf(k=20, mu=15)


# In[ ]:


#P-Value
from scipy import stats
rvs = stats.norm.rvs(loc = 5, scale = 10, size = (50,2))

stats.ttest_1samp(rvs,5.0)


# In[26]:


#Skew
from scipy.stats import skew
x =[53, 78, 64, 98, 97, 61, 67, 65, 83, 65]

skew(x, bias = False)

