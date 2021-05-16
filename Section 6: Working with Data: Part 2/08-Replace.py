# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
from pandas import Series, DataFrame


# %%
ser1 = Series([1,2,3,4,1,2,3,4])
ser1


# %%
ser1.replace(1,np.nan)


# %%
ser1.replace([1,4],[100,400])


# %%
ser1.replace({4:np.nan})


# %%



