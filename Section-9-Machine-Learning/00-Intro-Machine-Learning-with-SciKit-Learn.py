# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
# 1) Setup the data
import numpy as np
import pandas as pd
from pandas import Series, DataFrame


# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# %%
from sklearn.datasets import load_boston


# %%
boston = load_boston()


# %%
print (boston.DESCR)


# %%
plt.hist(boston.target,bins=50)

plt.xlabel('Prices in $1000s')
plt.ylabel('Number of houses')


# %%
plt.scatter(boston.data[:,5],boston.target) # column 5 of the data set array: RM

plt.ylabel('Price in $1000s')
plt.xlabel('Number of rooms')
# larger number of rooms = larger house size = larger prize
# therefore we see a correlation between the number of rooms and the price of the house


# %%
boston_df = DataFrame(boston.data)

boston_df.columns = boston.feature_names

boston_df.head()


# %%
boston_df['Price'] = boston.target


# %%
boston_df.head()


# %%
sns.lmplot(x='RM',y='Price',data=boston_df)


