# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import numpy as np
import pandas as pd
from pandas import Series, DataFrame


# %%
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/'


# %%
dframe_wine = pd.read_csv('winequality-red.csv', sep=';')


# %%
dframe_wine.head()


# %%
dframe_wine['alcohol'].mean()


# %%
dframe_wine


# %%
def max_to_min(arr):
    return arr.max() - arr.min()


# %%
wino = dframe_wine.groupby('quality')


# %%
wino.describe()


# %%
wino.agg(max_to_min)


# %%
wino.agg('mean') # .mean() in a different way


# %%
dframe_wine.head()


# %%
dframe_wine['qual/alc ratio'] = dframe_wine['quality'] / dframe_wine['alcohol']


# %%
dframe_wine.head()


# %%
dframe_wine.pivot_table(index=['quality'])


# %%
get_ipython().run_line_magic('matplotlib', 'inline')


# %%
dframe_wine.plot(kind='scatter', x='quality', y='alcohol')


