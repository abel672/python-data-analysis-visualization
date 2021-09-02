# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
from pandas import Series, DataFrame
import pandas as pd

from numpy.random import randn


# %%
ser = Series(randn(6), index=[[1,1,1,2,2,2],['a','b','c','a','b','c']])


# %%
ser


# %%
ser.index


# %%
ser[1]


# %%
ser[2]


# %%
ser[:,'a']


# %%
dframe = ser.unstack()


# %%
dframe


# %%
dframe2 = DataFrame(np.arange(16).reshape(4,4),index=[['a','a','b','b'],[1,2,1,2]],
                    columns=[['NY','NY','LA','SF'],['cold','hot','hot','cold']])

dframe2


# %%
dframe2.index.names = ['INDEX_1','INDEX_2']

dframe2.columns.names = ['Cities','Temp']

dframe2


# %%
dframe2.swaplevel('Cities', 'Temp', axis=1)


# %%
dframe2.sort_index(0)


# %%
dframe2.sum(level='Temp', axis=1)


# %%



