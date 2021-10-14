# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# [Notebook](https://github.com/jmportilla/Statistics-Notes/blob/master/Discrete%20Uniform%20Distributions.ipynb)

# %%
import numpy as np 
from numpy.random import randn
import pandas as pd 
from scipy import stats

import matplotlib as mpl 
import matplotlib.pyplot as plt 
import seaborn as sns 

from __future__ import division
get_ipython().run_line_magic('matplotlib', 'inline')


# %%
# Roll dice
roll_options = [1,2,3,4,5,6]

# Total probability space
tprob = 1

# Each roll has some odds of appearing
prob_roll = tprob / len(roll_options)

# Plot using seaborn rugplot
uni_plot = sns.rugplot(roll_options, height=prob_roll, c='indianred')

uni_plot.set_title('Probability Mass Function for Dice Roll')


# %%
# Create Discrete Uniform Distribution with Scipy
from scipy.stats import randint

low, high = 1,7

# Get mean and variance
mean,var = randint.stats(low, high)

print('The mean is %2.1f' %mean)


# %%
# Now we can make a bar plot
plt.bar(roll_options, randint.pmf(roll_options, low, high))


# %%



