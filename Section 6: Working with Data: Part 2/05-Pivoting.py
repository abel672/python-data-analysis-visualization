# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
from pandas import Series, DataFrame


# %%
# import pandas testing utility
import pandas.util.testing as tm; tm.N = 3

# Create a unpivoted function
def unpivot(frame):
    N, K = frame.shape

    data = {'value': frame.values.ravel('F'),
            'variable': np.asarray(frame.columns).repeat(N),
            'date': np.tile(np.asarray(frame.index), K)}
    
    # Return the DataFrame
    return DataFrame(data, columns=['date', 'variable', 'value'])

# Set the DataFrame we will be using
dframe = unpivot(tm.makeTimeDataFrame())


# %%
dframe


# %%
dframe_piv = dframe.pivot('date','variable','value')


# %%
dframe_piv


# %%



