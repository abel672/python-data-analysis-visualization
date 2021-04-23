# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
from pandas import Series, DataFrame

import pandas as pd


# %%
my_ser = Series([1,2,3,4], index=['A', 'B', 'C', 'D'])

my_ser


# %%
my_index = my_ser.index


# %%
my_index


# %%
my_index[2]


# %%
my_index[2:]


# %%
my_index[0] = 'Z' # index are immutable


# %%



