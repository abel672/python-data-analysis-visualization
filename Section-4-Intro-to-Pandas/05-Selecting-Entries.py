# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd

from pandas import Series, DataFrame


# %%
ser1 = Series(np.arange(3), index=['A','B','C'])

ser1 = 2*ser1

ser1


# %%
ser1['B']


# %%
ser1[1]


# %%
ser1[0:3]


# %%
ser1[['A','B']]


# %%
# condition
ser1[ser1>3]


# %%
ser1[ser1>3] = 10

ser1


# %%
dframe = DataFrame(np.arange(25).reshape((5,5)), index=['NYC','LA','SF','DC','Chi'],
                    columns=['A','B','C','D','E'])

dframe


# %%
dframe['B']


# %%
dframe[['B', 'E']]


# %%
dframe[dframe['C']>8]


# %%
dframe


# %%
dframe > 10


# %%
dframe.loc['LA']


# %%
dframe.iloc[1]


# %%



