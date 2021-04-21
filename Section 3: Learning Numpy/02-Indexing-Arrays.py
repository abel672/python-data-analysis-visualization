# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np


# %%
arr = np.arange(0,11)


# %%
arr


# %%
arr[8]


# %%
# print a range in your array
arr[0:5]


# %%
# assign value to a range in your array
arr[0:5] = 100


# %%
arr


# %%
arr = np.arange(0,11)


# %%
arr


# %%
slice_of_arr = arr[0:6]

slice_of_arr


# %%
# overried entire array
slice_of_arr[:] = 99


# %%
slice_of_arr


# %%
# copying an array
arr_copy = arr.copy()


# %%
arr_copy


# %%
# create 2d arrat
arr_2d = np.array(([5,10,15],[20,25,30],[35,40,45]))

arr_2d


# %%
arr_2d[1]


# %%
arr_2d[0]


# %%
arr_2d


# %%
# take a matrix range
arr_2d[:2, 1:]


# %%
arr2d = np.zeros((10,10))


# %%
arr2d


# %%
arr_length = arr2d.shape[1]


# %%
arr_length


# %%
for i in range(arr_length):
    arr2d[i]=i


# %%
arr2d


# %%
# getting rows
arr2d[[2,4,6,8]]


# %%
arr2d[[3,2,6,5]]


# %%



