# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
from pandas import Series,DataFrame


# %%
dframe = pd.read_csv('lec25.csv')


# %%
dframe


# %%
dframe = pd.read_csv('lec25.csv', header = None)


# %%
dframe


# %%
dframe = pd.read_table('lec25.csv', sep=',', header=None)


# %%
dframe


# %%
pd.read_csv('lec25.csv', header=None, nrows=2)


# %%
# writing dframe into a csv file
dframe.to_csv('mytextdata_out.csv')


# %%
import sys


# %%
dframe.to_csv(sys.stdout)


# %%
dframe.to_csv(sys.stdout,sep='_')


# %%
dframe.to_csv(sys.stdout, sep='?')


# %%
dframe.to_csv(sys.stdout, columns=[0,1,2])


# %%
url = 'https://docs.python.org/3/library/csv.html'


