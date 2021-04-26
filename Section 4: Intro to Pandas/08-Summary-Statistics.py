# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import numpy as np

from pandas import Series, DataFrame

import pandas as pd


# %%
arr = np.array([[1,2, np.nan], [np.nan, 3, 4]])


# %%
dframe1 = DataFrame(arr, index=['A', 'B'], columns=['One', 'Two', 'Three'])

dframe1


# %%
dframe1.sum() # sum column


# %%
dframe1.sum(axis=1) # sum row


# %%
dframe1.min()


# %%
dframe1


# %%
dframe1.idxmin()


# %%
dframe1.cumsum() # sum columns progresively


# %%
dframe1.describe()


# %%
from IPython.display import YouTubeVideo


# %%
YouTubeVideo('xGbpuFNR1ME')


# %%
YouTubeVideo('4EXNedimDMs')


# %%
import pandas_datareader.data as pdweb
import datetime


# %%
prices = pdweb.get_data_yahoo(['CVX', 'XOM', 'BP'], start = datetime.datetime(2010,1,1),
                                end=datetime.datetime(2013, 1, 1))['Adj Close']

prices.head()


# %%
volume = pdweb.get_data_yahoo(['CVX', 'XOM', 'BP'], start=datetime.datetime(2010,1,1),
                                end=datetime.datetime(2013,1,1))['Volume']


# %%
volume.head()


# %%
rets = prices.pct_change()


# %%
# Correlation of the stocks
corr = rets.corr


# %%
get_ipython().run_line_magic('matplotlib', 'inline')
prices.plot()


# %%
import seaborn as sns
# from seaborn.regression import corrplot,symmatplot
import matplotlib.pyplot as plt


# %%
sns.heatmap(rets, annot=False)


# %%
ser1 = Series(['w', 'w', 'x', 'y', 'z', 'w', 'x', 'y', 'x', 'a'])

ser1


# %%
ser1.unique()


# %%
ser1.value_counts()


# %%


