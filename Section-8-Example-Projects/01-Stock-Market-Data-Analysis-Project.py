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


# %%
# Moving Averages
ma_day = [10,20,50]

for ma in ma_day:
    column_name = "MA for %s days" %(str(ma))
    AAPL[column_name] = AAPL['Adj Close'].rolling(ma).mean()


# %%
AAPL[['Adj Close','MA for 10 days','MA for 20 days','MA for 50 days']].plot(subplots=False,figsize=(10,4))


# %%
AAPL['Daily Return'] = AAPL['Adj Close'].pct_change()
AAPL['Daily Return'].plot(figsize=(10,4),legend=True,linestyle='--',marker='o')


# %%
sns.displot(AAPL['Daily Return'].dropna(),bins=100,color='purple')


# %%
AAPL['Daily Return'].hist(bins=100)


# %%
closing_df = DataReader(tech_list,'yahoo',start,end)['Adj Close']


# %%
closing_df.head()


# %%
tech_rets = closing_df.pct_change()


# %%
tech_rets.head()


# %%
sns.jointplot(x='GOOG',y='GOOG',data=tech_rets,kind='scatter',color='seagreen')


# %%
sns.jointplot(x='GOOG',y='MSFT',data=tech_rets,kind='scatter')


# %%
url = 'https://en.wikipedia.org/wiki/Pearson_correlation_coefficient'


