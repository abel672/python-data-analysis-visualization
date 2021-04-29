# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np 
import pandas as pd 
from pandas import Series, DataFrame


# %%
ser1 = Series([2, np.nan, 4,np.nan,6,np.nan],
            index=['Q','R','S','T','U','V'])

ser1


# %%
ser2 = Series(np.arange(len(ser1)),dtype=np.float64,
                index=['Q','R','S','T','U','V'])
ser2


# %%


# %% [markdown]
# 

# %%



