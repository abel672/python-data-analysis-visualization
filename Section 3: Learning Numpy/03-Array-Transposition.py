# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np


# %%
arr = np.arange(50).reshape((10,5))
arr


# %%
# Transpos array (turn rows into columns)
arr.T


# %%
np.dot(arr.T, arr)


# %%
arr3d = np.arange(50).reshape((5,5,2))
arr3d


# %%
# transpose method to also transpos an multi dimensional array
arr3d.transpose((1,0,2))


# %%
# if you want to get very specific, you can swap axis
arr = np.array([[1,2,3]])
arr


# %%
arr.swapaxes(0,1)


# %%



