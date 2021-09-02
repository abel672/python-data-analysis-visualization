# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np


# %%
my_list1 = [1,2,3,4]


# %%
# creating an array
my_array1 = np.array(my_list1)


# %%
my_array1


# %%
my_list2 = [11,22,33,44]


# %%
my_lists = [my_list1, my_list2]


# %%
# creating matrix
my_array2 = np.array(my_lists)


# %%
my_array2


# %%
# (rows, cols)
my_array2.shape


# %%
# array type
my_array2.dtype


# %%
# create new array
np.zeros(5)


# %%
my_zeros_array = np.zeros(5)


# %%
my_zeros_array.dtype


# %%
# create new matrix
np.ones([5,5])


# %%
# create an empty array
np.empty(5)


# %%
# identity matrix
np.eye(5)


# %%
# create a full array
np.arange(5)


# %%
# from 5 to 50, jumping 2 every time
np.arange(5,50,2)


# %%



