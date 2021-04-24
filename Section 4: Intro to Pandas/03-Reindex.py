# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
from pandas import Series, DataFrame
import pandas as pd

from numpy.random import randn


# %%
ser1 = Series([1,2,3,4], index = ['A', 'B', 'C', 'D'])

ser1


# %%
ser2 = ser1.reindex(['A', 'B', 'C', 'D', 'E', 'F'])

ser2


# %%
# reindex with default value
ser2.reindex(['A','B','C','D','E','F','G'], fill_value=0)


# %%
ser3 = Series(['USA','Mexico','Canada'], index=[0,5,10])

ser3


# %%
ranger = range(15)


# %%
ranger


# %%
ser3.reindex(ranger, method='ffill')


# %%
dframe = DataFrame(randn(25).reshape((5,5)), index=['A', 'B', 'C', 'D', 'E'],
                    columns=['col1','col2','col3','col4','col5'])

dframe


# %%
dframe2 = dframe.reindex(['A', 'B', 'C', 'D', 'E', 'F']) # reindex and adding a new index 'F'


# %%
dframe2


# %%
new_columns = ['col1', 'col2', 'col3', 'col4', 'col5']


# %%
dframe2.reindex(columns=new_columns)


# %%
dframe


# %%
# dframe.ix[['A', 'B', 'C', 'D', 'E', 'F'], new_columns] # deprecated
dframe.loc[['A', 'B', 'C', 'D', 'E'], new_columns] # new version


# %%



