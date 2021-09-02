# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
from pandas import Series, DataFrame


# %%
dframe = DataFrame({'key1':['A']*2 + ['B']*3,
                    'key2':[2,2,2,3,3]})


# %%
dframe


# %%
dframe.duplicated()


# %%
dframe.drop_duplicates()


# %%
dframe.drop_duplicates(['key1'])


# %%
dframe.drop_duplicates(['key1'],keep='first')


# %%



