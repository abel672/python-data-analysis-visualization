# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
from pandas import Series, DataFrame


# %%
dframe1 = DataFrame({'key':['X','Z','Y','Z','X','X'], 'data_set_1':np.arange(6)})

dframe1


# %%
dframe2 = DataFrame({'key':['Q', 'Y', 'Z'], 'data_set_2':[1,2,3]})

dframe2


# %%
pd.merge(dframe1, dframe2)


# %%
pd.merge(dframe1, dframe2, on='key')


# %%
pd.merge(dframe1, dframe2, on='key', how='left')


# %%
dframe2


# %%
pd.merge(dframe1, dframe2, on='key', how='right')


# %%
pd.merge(dframe1, dframe2, on='key', how='outer')


# %%
dframe3 = DataFrame({'key':['X','X','X','Y','Z','Z'], 'data_set_3': range(6)})


# %%
dframe4 = DataFrame({'key':['Y','Y','X','X','Z'], 'data_set_4': range(5)})


# %%
dframe3


# %%
dframe4


# %%
pd.merge(dframe3, dframe4)


# %%
df_left = DataFrame({'key1':['SF','SF','LA'],
                        'key2':['one','two','one'],
                        'right_data':[10,20,30]})


# %%
df_right = DataFrame({'key1':['SF','SF','LA','LA'],
                        'key2':['one','one','one','two'],
                        'right_data':[40,50,60,70]})


# %%
df_left


# %%
df_right


# %%
pd.merge(df_left,df_right,on=['key1','key2'],how='outer')


# %%
pd.merge(df_left,df_right,on='key1')


# %%
pd.merge(df_left, df_right, on='key1', suffixes=('_lefty', '_righty'))


# %%
url = 'https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html'


