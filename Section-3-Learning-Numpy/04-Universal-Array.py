# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np


# %%
arr = np.arange(11)
arr


# %%
# square root of each number
np.sqrt(arr)


# %%
# exponent value of each number
np.exp(arr)


# %%
A = np.random.randn(10)

A


# %%
B = np.random.randn(10)
B


# %%
# Binary Functions
np.add(A,B) # sum of each index together


# %%
np.maximum(A,B) # maximum of each index


# %%
# Open Webside
website = 'https://docs.cupy.dev/en/stable/reference/ufunc.html'
import webbrowser
webbrowser.open(website)


# %%



