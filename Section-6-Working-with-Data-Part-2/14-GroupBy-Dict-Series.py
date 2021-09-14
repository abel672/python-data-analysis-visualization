# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
from pandas import Series, DataFrame


# %%
animals = DataFrame(np.arange(16).reshape(4,4),
                    columns=['W','X','Y','Z'],
                    index=['Dog','Cat','Bird','Mouse'])


# %%
animals


# %%
animals.loc[1:2,['W','Y']] = np.nan


# %%
animals


# %%
behavior_map = {'W':'good','X':'bad','Y':'good','Z':'bad'}


# %%
animal_col = animals.groupby(behavior_map, axis=1)

animal_col.sum()


# %%
behav_series = Series(behavior_map)

behav_series


# %%
animals.groupby(behav_series, axis=1).count()


# %%
animals


# %%
animals.groupby(len).sum()


# %%
keys = ['A','B','A','B']


# %%
animals.groupby([len,keys]).max()


# %%
hier_col = pd.MultiIndex.from_arrays([['NY','NY','NY','SF','SF'], [1,2,3,1,2]], names=['City', 'sub_value'])


# %%
dframe_hr = DataFrame(np.arange(25).reshape(5,5), columns=hier_col)


# %%
dframe_hr = dframe_hr * 100


# %%
dframe_hr


# %%



