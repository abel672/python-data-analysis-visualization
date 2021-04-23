# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np

import pandas as pd

from pandas import Series, DataFrame


# %%
obj = Series([3,6,9,12])

# index and value
obj


# %%
obj.values


# %%
obj.index


# %%
ww2_cas = Series([8700000, 4300000, 3000000, 2100000, 400000], index=['USSR', 'Germany', 'China', 'Japan', 'USA'])

ww2_cas


# %%
ww2_cas['USA']


# %%
# Check which countries has cas greater than 4 million
ww2_cas[ww2_cas > 4000000]


# %%
'USSR' in ww2_cas


# %%
# conver to a dictionary
ww2_dict = ww2_cas.to_dict()

ww2_dict


# %%
# from dictionary to series again
ww2_series = Series(ww2_dict)

ww2_series


# %%
countries = ['China', 'Germany', 'Japan', 'USA', 'USSR', 'Argentina']


# %%
# Passing new indexes
obj2 = Series(ww2_dict, index=countries)

obj2


# %%
pd.isnull(obj2)


# %%
pd.notnull(obj2)


# %%
ww2_series


# %%
obj2


# %%
# adding two series together
ww2_series + obj2


# %%
# Adding a name to a serie
obj2.name = "World War 2 Casualties"

obj2


# %%
obj2.name = 'Countries'


# %%
obj2


# %%



