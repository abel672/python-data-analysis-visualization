# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
from pandas import Series, DataFrame


# %%
arr1 = np.arange(9).reshape(3,3)

arr1


# %%
np.concatenate([arr1, arr1], axis=1)


# %%
np.concatenate([arr1, arr1], axis=0)


# %%
ser1 = Series([0,1,2], index=['T','U','V'])

ser2 = Series([3,4],index=['X','Y'])


# %%
ser1


# %%
ser2


# %%
pd.concat([ser1, ser2])


# %%
pd.concat([ser1,ser2],axis=1)


# %%
pd.concat([ser1,ser2],keys=['cat1','cat2'])


# %%
dframe1 = DataFrame(np.random.randn(4,3),columns=['X','Y','Z'])

dframe2 = DataFrame(np.random.randn(3,3),columns=['Y','Q','X'])


# %%
dframe1


# %%
dframe2


# %%
pd.concat([dframe1,dframe2])


# %%
pd.concat([dframe1, dframe2], ignore_index=True)


# %%
url = 'https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html'


