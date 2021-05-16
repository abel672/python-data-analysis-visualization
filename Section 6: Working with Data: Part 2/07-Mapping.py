# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
from pandas import Series, DataFrame


# %%
dframe = DataFrame({'city':['Alma','Brian Head','Fox Park'],
                    'altitude':[3158,3000,2762]})

dframe


# %%
state_map = {'Alma':'Colorado', 'Brian Head':'Utah', 'Fox Park':'Wyoming'}


# %%
dframe['state'] = dframe['city'].map(state_map)


# %%
dframe


# %%



