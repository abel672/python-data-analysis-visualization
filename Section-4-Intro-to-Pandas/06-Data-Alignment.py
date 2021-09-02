# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
from pandas import Series, DataFrame


# %%
ser1 = Series([0,1,2], index=['A','B','C'])
ser1


# %%
ser2 = Series([3,4,5,6], index=['A','B','C','D'])
ser2


# %%
ser1 + ser2


# %%
dframe1 = DataFrame(np.arange(4).reshape((2,2)), columns=list('AB'), index=['NYC', 'LA'])
dframe1


# %%
dframe2 = DataFrame(np.arange(9).reshape((3,3)), columns=list('ADC'), index=['NYC', 'SF', 'LA'])

dframe2


# %%
dframe1 + dframe2 # it will sum just the ones who matches in colums and row indexes


# %%
# adding frames together and controlling null values
dframe1.add(dframe2, fill_value=0)


# %%
ser3 = dframe2.iloc[0]


# %%
ser3


# %%
dframe2-ser3


# %%



