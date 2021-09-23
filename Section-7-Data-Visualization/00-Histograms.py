# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
# Standard
import numpy as np
import pandas as pd
from numpy.random import randn


# %%
# Stats
from scipy import stats


# %%
# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


# %%
get_ipython().run_line_magic('matplotlib', 'inline')


# %%
url = 'https://en.wikipedia.org/wiki/Histogram'


# %%
dataset1 = randn(100)


# %%
plt.hist(dataset1)


# %%
dataset2 = randn(80)

plt.hist(dataset2, color='indianred')


# %%
plt.hist(dataset1,density=True,color='indianred',alpha=0.5,bins=20)
plt.hist(dataset2,density=True,alpha=0.5,bins=20)


# %%
data1 = randn(1000)
data2 = randn(1000)


# %%
sns.jointplot(data1, data2)


# %%
sns.jointplot(data1,data2,kind='hex')


