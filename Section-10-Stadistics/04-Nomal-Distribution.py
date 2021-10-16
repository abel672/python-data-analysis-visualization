# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# [Notebook](https://github.com/jmportilla/Statistics-Notes/blob/master/Normal%20Distribution.ipynb)
# 
# [Normal Distribution](https://www.youtube.com/watch?v=rzFX5NWojp0&ab_channel=StatQuestwithJoshStarmer)

# %%
from IPython.display import Image
Image(url='http://upload.wikimedia.org/wikipedia/commons/thumb/2/25/The_Normal_Distribution.svg/725px-The_Normal_Distribution.svg.png')


# %%
# Normal distribution with scipy
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from scipy import stats

mean = 0

std = 1

X = np.arange(-4,4,0.01)

Y = stats.norm.pdf(X, mean, std)

plt.plot(X,Y)


# %%
# now let's do it with numpy
mu, sigma = 0,0.1

norm_set = np.random.normal(mu, sigma, 1000)


# %%
import seaborn as sns

plt.hist(norm_set, bins=50)


