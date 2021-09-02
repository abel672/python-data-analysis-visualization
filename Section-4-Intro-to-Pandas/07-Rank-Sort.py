# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
from pandas import Series, DataFrame
import pandas as pd


# %%
ser1 = Series(range(3), index=['C', 'A', 'B'])

ser1


# %%
ser1.sort_index()


# %%
ser1.sort_values()


# %%
from numpy.random import randn


# %%
ser2 = Series(randn(10))

ser2


# %%
ser2.sort_values() # sort orders by ranking value


# %%
ser2.rank() # every value as a rank by default, sort() uses it to order


# %%
ser3 = Series(randn(10))

ser3


# %%
ser3.rank()


# %%
ser3


# %%



