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


# %%
# Quick time series analysis
poll_df.head()


# %%
poll_df.plot(x='End Date',y=['Obama','Romney','Undecided'],linestyle='',marker='o')


# %%
# difference vs time
from datetime import datetime


# %%
poll_df['Difference'] = (poll_df.Obama - poll_df.Romney) / 100

poll_df.head()


# %%
poll_df = poll_df.groupby(['Start Date'],as_index=False).mean() # To keep the original indexes

poll_df.head()


# %%
poll_df.plot(x='Start Date',y='Difference',figsize=(12,4),marker='o',linestyle='-',color='purple')


# %%
# Exercise: Look for the particular dates when Romney won .min() is the hint for this.
lower_rate = poll_df['Difference'].min()
poll_df.loc[poll_df['Difference'] == lower_rate]


# %%
# Markdown debates in October 2012
row_in = 0
xlimit = []

for date in poll_df['Start Date']:
    if date[0:7] == '2012-10':
        xlimit.append(row_in)
        row_in += 1
    else:
        row_in += 1

xlimit_min = min(xlimit)
xlimit_max = max(xlimit)

print(xlimit_min)
print(xlimit_max)


# %%
poll_df.plot(x='Start Date',y='Difference',figsize=(12,4),marker='o',linestyle='-',color='purple',xlim=(xlimit_min, xlimit_max))

# Oct 3rd
plt.axvline(x=xlimit_min+2,linewidth=4,color='grey')

# Oct 11th
plt.axvline(x=xlimit_min+10,linewidth=4,color='grey')

# Oct 22nd
plt.axvline(x=xlimit_min+21,linewidth=4,color='grey')


