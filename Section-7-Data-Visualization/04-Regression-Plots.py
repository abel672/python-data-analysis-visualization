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
tips = sns.load_dataset('tips')


# %%
tips.head()


# %%
sns.lmplot(x='total_bill', y='tip', data=tips)


# %%
sns.lmplot(x='total_bill',y='tip',data=tips,
            scatter_kws={'marker':'o','color':'indianred'},
            line_kws={'linewidth':1,'color':'blue'})


# %%
sns.lmplot(x='total_bill',y='tip',data=tips,order=4,
            scatter_kws={'marker':'o','color':'indianred'},
            line_kws={'linewidth':1,'color':'blue'})


# %%
sns.lmplot(x='total_bill',y='tip',data=tips,fit_reg=False)


# %%
tips.head()


# %%
tips['tip_percent']=100*(tips['tip'] / tips['total_bill'])


# %%
tips.head()


# %%
sns.lmplot(x='size',y='tip_percent',data=tips)


# %%
url = 'https://en.wikipedia.org/wiki/Jitter'


# %%
sns.lmplot(x='size',y='tip_percent',data=tips,x_jitter=.1) # adding a jitter


# %%
sns.lmplot(x='size',y='tip_percent',data=tips,x_estimator=np.mean)


# %%
sns.lmplot(x='total_bill',y='tip_percent',data=tips,hue='sex',markers=['x','o']) # hue by column 'sex'


# %%
sns.lmplot(x='total_bill',y='tip_percent',data=tips,hue='day') # hue by column 'day'


# %%
url = 'https://en.wikipedia.org/wiki/Local_regression'

sns.lmplot(x='total_bill',y='tip_percent',data=tips,lowess=True,line_kws={'color':'black'})


# %%
sns.regplot(x='total_bill',y='tip_percent',data=tips) # regression plot


# %%
# creating two subplots
fig, (axis1, axis2) = plt.subplots(1,2,sharey=True)

# configuring them
sns.regplot(x='total_bill',y='tip_percent',data=tips,ax=axis1)
sns.violinplot(y=tips['tip_percent'],x=tips['size'],color='red',ax=axis2)


