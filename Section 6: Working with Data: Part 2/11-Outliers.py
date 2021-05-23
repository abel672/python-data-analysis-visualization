# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
from pandas import Series, DataFrame


# %%
np.random.seed(12345)


# %%
dframe = DataFrame(np.random.randn(1000,4))


# %%
dframe.head()


# %%
dframe.tail()


# %%
dframe.describe()


# %%
col = dframe[0]


# %%
col.head()


# %%
col[np.abs(col)>3]


# %%
dframe[(np.abs(dframe)>3).any(1)]


# %%
dframe[np.abs(dframe)>3] = np.sign(dframe)*3


# %%
dframe.describe()


# %%



