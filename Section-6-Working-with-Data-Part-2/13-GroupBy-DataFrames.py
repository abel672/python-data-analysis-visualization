# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
from pandas import DataFrame, Series


# %%
dframe = DataFrame({'k1':['X','X','Y','Y','Z'],
                    'k2':['alpha','beta','alpha','beta','alpha'],
                    'dataset1':np.random.randn(5),
                    'dataset2':np.random.randn(5)})

dframe


# %%
group1 = dframe['dataset1'].groupby(dframe['k1'])


# %%
group1


# %%
group1.mean()


# %%
cities = np.array(['NY','LA','LA','NY','NY'])

month = np.array(['JAN','FEB','JAN','FEB','JAN'])


# %%
dframe['dataset1'].groupby([cities,month]).mean()


# %%
dframe


# %%
dframe.groupby('k1').mean()


# %%
dframe.groupby(['k1','k2']).mean()


# %%
dframe.groupby(['k1']).size()


# %%
dframe


# %%
for name, group in dframe.groupby('k1'):
    print(f"This is the {name} group")
    print(group)
    print('\n')


# %%
for (k1,k2), group in dframe.groupby(['k1','k2']):
    print(f"Key1 = {k1} Key2 {k2}")
    print(group)
    print('\n')


# %%
group_dict = dict(list(dframe.groupby('k1')))


# %%
group_dict['X']


# %%
group_dict_axis1 = dict(list(dframe.groupby(dframe.dtypes, axis=1)))


# %%
group_dict_axis1


# %%
dataset2_group = dframe.groupby(['k1','k2'])[['dataset2']]

dataset2_group.mean()


# %%



