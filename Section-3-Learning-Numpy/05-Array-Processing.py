# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import numpy as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# %%
points = np.arange(-5,5,0.01)


# %%
dx,dy = np.meshgrid(points,points)


# %%
dx


# %%
z = (np.sin(dx) + np.sin(dy))

z


# %%
plt.imshow(z)


# %%
plt.imshow(z)
plt.colorbar()

plt.title('Plot for sin(x)+sin(y)')


# %%
# numpy where

A = np.array([1,2,3,4]) # true cases

B = np.array([100, 200, 300, 400]) # false cases


# %%
condition = np.array([True, True, False, False]) # condition


# %%
# where function's formula
answer = [(A_val if cond else B_val) for A_val, B_val, cond in zip(A,B, condition)]


# %%
answer


# %%
# where function of numpy
answer2 = np.where(condition, A, B) # same than above


# %%
answer2


# %%
from numpy.random import randn


# %%
arr = randn(5,5)
arr


# %%
# clean up data with where function
np.where(arr<0, 0, arr)


# %%
# sum everything of a matrix
arr = np.array([[1,2,3], [4,5,6], [7,8,9]])
arr


# %%
arr.sum()


# %%
# sum the columns
arr.sum(0)


# %%
# average
arr.mean()


# %%
# standard deviation
arr.std()


# %%
# variant
arr.var()


# %%
bool_arr = np.array([True, False, True])


# %%
bool_arr.any() # checking some true value


# %%
bool_arr.all() # all values are true?


# %%
# Sort
arr = randn(5)
arr


# %%
arr.sort()
arr


# %%
# remove repeated values
countries = np.array(['France', 'Germany', 'USA', 'Russia', 'USA', 'Mexico', 'Germany'])


# %%
np.unique(countries)


# %%
# check if values are in array
np.in1d(['France', 'USA', 'Sweden'], countries)


# %%



