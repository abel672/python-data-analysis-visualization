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
dataset = randn(100)


# %%
sns.distplot(dataset, bins=25)


# %%
sns.distplot(dataset, bins=25, rug=True, hist=False)


# %%
sns.distplot(dataset, bins=25,
            kde_kws={'color':'indianred','label':'KDE PLOT'},
            hist_kws={'color':'blue','label':'HIST'})


# %%
from pandas import Series

ser1 = Series(dataset, name='My_data')

ser1


# %%
sns.distplot(ser1, bins=25)


