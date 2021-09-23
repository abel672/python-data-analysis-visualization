# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd


# %%
from io import StringIO


# %%
data="""Sample Animal Intelligence
1 Dog Smart
2 Dog Smart
3 Cat Dumb
4 Cat Dumb
5 Dog Dumb
6 Cat Smart"""


# %%
dframe = pd.read_table(StringIO(data),sep='\s+')


# %%
dframe


# %%
pd.crosstab(dframe.Animal, dframe.Intelligence, margins=True) # get Animal, sort by intelligence


# %%



