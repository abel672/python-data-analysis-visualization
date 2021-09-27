# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
from __future__ import division

# For data
import pandas as pd
from pandas import Series, DataFrame
import numpy as np

# For visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# %%
import requests

from io import StringIO


# %%
url = 'https://elections.huffingtonpost.com/pollster/2012-general-election-romney-vs-obama.csv'

source = requests.get(url, verify=False).text

poll_data = StringIO(source)


# %%
poll_df = pd.read_csv(poll_data)


# %%
poll_df.info()


# %%
poll_df.head()


# %%
sns.catplot(x='Affiliation',data=poll_df,kind='count')


# %%
sns.catplot(x='Affiliation',data=poll_df,hue='Population',kind='count')


# %%
poll_df.head()


# %%
avg = pd.DataFrame(poll_df.mean())

avg.drop('Number of Observations',axis=0,inplace=True)
# std.drop('Question Text',axis=0,inplace=True)
# std.drop('Question Iteration',axis=0,inplace=True)


# %%
avg.head()


# %%
std = pd.DataFrame(poll_df.std())
std.drop('Number of Observations',axis=0,inplace=True)


# %%
std.head()


# %%
avg.plot(yerr=std,kind='bar',legend=False) # plot avg and std together in a plot


# %%
poll_avg = pd.concat([avg,std],axis=1)


# %%
poll_avg.columns = ['Average','STD']


# %%
poll_avg
