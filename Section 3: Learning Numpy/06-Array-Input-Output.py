# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np


# %%
arr = np.arange(5)


# %%
arr


# %%
np.save('myarray', arr)


# %%
arr = np.arange(10)
arr


# %%
np.load('myarray.npy')


# %%
arr1 = np.load('myarray.npy')
arr1


# %%
arr2 = arr


# %%
arr2


# %%
# saving multiple arrays
np.savez('ziparray.npz', x=arr1, y=arr2)


# %%
archive_array = np.load('ziparray.npz')


# %%
archive_array['x']


# %%
archive_array['y']


# %%
arr = np.array([[1,2,3],[4,5,6]])
arr


# %%
np.savetxt('mytextarray.txt', arr, delimiter=',')


# %%
arr = np.loadtxt('mytextarray.txt', delimiter=',')
arr


# %%



