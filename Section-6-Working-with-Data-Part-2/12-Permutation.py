# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
from pandas import Series, DataFrame


# %%
dframe = DataFrame(np.arange(16).reshape(4,4))


# %%
blender = np.random.permutation(4)


# %%
blender


# %%
dframe


# %%
dframe.take(blender)


# %%
box = np.array([1,2,3])


# %%
shaker = np.random.randint(0,len(box),size=10)


# %%
shaker


# %%
hand_grabs = box.take(shaker)

hand_grabs


# %%



