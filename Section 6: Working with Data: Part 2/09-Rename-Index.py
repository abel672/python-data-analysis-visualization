# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
from pandas import Series, DataFrame


# %%
dframe = DataFrame(np.arange(12).reshape(3,4),
                    index = ['NY','LA','SF'],
                    columns=['A','B','C','D'])

dframe


# %%
dframe.index.map(str.lower)


# %%
dframe.index = dframe.index.map(str.lower)


# %%
dframe


# %%
dframe.rename(index=str.title, columns=str.lower)


# %%
dframe.rename(index={'ny':'NEW YORK'},
                columns={'A':'ALPHA'})


# %%
dframe


# %%
dframe.rename(index={'ny':'NEW YORK'},inplace=True)


# %%
dframe


# %%



