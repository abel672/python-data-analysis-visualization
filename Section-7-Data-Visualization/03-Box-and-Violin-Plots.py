# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
# Standard
import numpy as np
import pandas as pd
from numpy.random import randn

# Stats
from scipy import stats

# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# %%
url = 'https://en.wikipedia.org/wiki/Box_plot'
tutorial = 'https://www.youtube.com/watch?v=Vo-bfTqEFQk&ab_channel=KimberlyFessel'

# %%
data1 = randn(100)
data2 = randn(100)


# %%
sns.boxplot(data=[data1, data2])


# %%
sns.boxplot(data=[data1, data2], whis=np.inf)


# %%
sns.boxplot(data=data1, orient="h")


# %%
# Normal Dist
data1 = stats.norm(0,5).rvs(100)

# Two gamma dist. Concatenated together
data2 = np.concatenate([stats.gamma(5).rvs(50)-1,
                        -1*stats.gamma(5).rvs(50)])

# Box plot both data1 and data2
sns.boxplot(data=[data1, data2], whis=np.inf)


# %%
sns.violinplot(data=[data1, data2])


# %%
sns.violinplot(data=data2, bw=0.01)


# %%
sns.violinplot(data=data1, inner='stick')


