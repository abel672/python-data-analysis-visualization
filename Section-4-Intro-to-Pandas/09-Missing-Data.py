# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
from pandas import Series, DataFrame
import pandas as pd


# %%
data = Series(['one', 'two', np.nan, 'four'])


# %%
data


# %%
data.isnull()


# %%
data.dropna() # drop null values


# %%
dframe = DataFrame([[1,2,3],[np.nan,5,6],[7,np.nan,9],[np.nan,np.nan,np.nan]])

dframe


# %%
clean_dframe = dframe.dropna()


# %%
clean_dframe


# %%
dframe.dropna(how='all') # drop rows that are fully null


# %%
dframe.dropna(axis=1) # drop all


# %%
npn = np.nan

dframe2 = DataFrame([[1,2,3,npn],[2,npn,5,6],[npn,7,npn,9],[1,npn,npn,npn]])


# %%
dframe2


# %%
dframe2.dropna(thresh=2)


# %%
dframe2.fillna(1) # fill null with value


# %%
dframe2.fillna({0:0,1:1,2:2,3:3})


# %%
dframe2.fillna(0,inplace=True) # make the change permanent


# %%
dframe2


# %%



