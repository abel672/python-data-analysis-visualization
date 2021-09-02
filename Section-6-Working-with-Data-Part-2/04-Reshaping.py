# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
from pandas import Series, DataFrame


# %%
dframe1 = DataFrame(np.arange(8).reshape(2,4),
                    index=pd.Index(['LA','SF'],name='city'),
                    columns=pd.Index(['A','B','C','D'],name='letter'))

dframe1


# %%
dframe_st = dframe1.stack()

dframe_st


# %%
dframe_st.unstack()


# %%
dframe_st.unstack('letter')


# %%
dframe_st.unstack('city')


# %%
ser1 = Series([0,1,2],index=['Q','X','Y'])

ser2 = Series([4,5,6],index=['X','Y','Z'])


# %%
dframe = pd.concat([ser1,ser2],keys=['Alpha','Beta'])
dframe


# %%
dframe.unstack()


# %%
dframe.unstack().stack()


# %%
dframe = dframe.unstack()


# %%
dframe

# %% [markdown]
# 

# %%
dframe.stack(dropna=False)


# %%



