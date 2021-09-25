# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import pandas as pd
from pandas import Series, DataFrame
import numpy as np

# %% [markdown]
# 

# %%
import matplotlib.pyplot
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# %%
from pandas_datareader import DataReader


# %%
from datetime import datetime


# %%
from __future__ import division


# %%
tech_list = ['AAPL','GOOG','MSFT','AMZN']


# %%
end = datetime.now()

start = datetime(end.year-1, end.month, end.day)


# %%
for stock in tech_list:
    globals()[stock] = DataReader(stock,data_source='yahoo',start=start,end=end)


# %%
AAPL.head()


# %%
AAPL.info()


# %%
AAPL['Adj Close'].plot(legend=True,figsize=(10,4))


# %%
AAPL['Volume'].plot(legend=True,figsize=(10,4))


