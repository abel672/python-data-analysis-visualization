# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
from pandas import Series, DataFrame
import pandas as pd


# %%
ser1 = Series(np.arange(3), index=['a','b','c'])

ser1


# %%
ser1.drop('b')


# %%
dframe1 = DataFrame(np.arange(9).reshape((3,3)), index=['SF','LA','NY'], columns=['pop', 'size', 'year'])

dframe1


# %%
# droping row
dframe1.drop('LA') # returning modified frame


# %%
dframe1


# %%
dframe2 = dframe1.drop('LA')


# %%
dframe2


# %%
dframe1.drop('year', axis=1) # droping column


# %%



