# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
from pandas import Series, DataFrame


# %%
dframe_wine = pd.read_csv('winequality-red.csv', sep=';')

dframe_wine.head()


# %%
def ranker(df):
    df['alc_content_rank'] = np.arange(len(df)) + 1
    return df


# %%
dframe_wine.sort_values('alcohol', ascending = False, inplace=True)


# %%
dframe_wine = dframe_wine.groupby('quality').apply(ranker) # apply ranker function (slit and combine by groupby() and apply())


# %%
dframe_wine.head()


# %%
num_of_qual = dframe_wine['quality'].value_counts()


# %%
num_of_qual


# %%
dframe_wine[dframe_wine.alc_content_rank == 1].head(len(num_of_qual))


