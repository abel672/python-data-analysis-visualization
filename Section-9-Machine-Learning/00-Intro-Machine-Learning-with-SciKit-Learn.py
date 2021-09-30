# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
# 1) Setup the data
import numpy as np
import pandas as pd
from pandas import Series, DataFrame


# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# %%
from sklearn.datasets import load_boston


# %%
boston = load_boston()


# %%
print (boston.DESCR)


# %%
plt.hist(boston.target,bins=50)

plt.xlabel('Prices in $1000s')
plt.ylabel('Number of houses')


# %%
plt.scatter(boston.data[:,5],boston.target) # column 5 of the data set array: RM

plt.ylabel('Price in $1000s')
plt.xlabel('Number of rooms')
# larger number of rooms = larger house size = larger prize
# therefore we see a correlation between the number of rooms and the price of the house


# %%
boston_df = DataFrame(boston.data)

boston_df.columns = boston.feature_names

boston_df.head()


# %%
boston_df['Price'] = boston.target


# %%
boston_df.head()


# %%
# Linear Regression with Seaborn
sns.lmplot(x='RM',y='Price',data=boston_df)

# %% [markdown]
# 
# ## Step 3: The mathematics behind the Least Squares Method.¶
# In this particular lecture we'll use the least squares method as the way to estimate the coefficients. Here's a quick breakdown of how this method works mathematically:
# 
# Take a quick look at the plot we created above using seaborn. Now consider each point, and know that they each have a coordinate in the form (X,Y). Now draw an imaginary line between each point and our current "best-fit" line. We'll call the distanace between each point and our current best-fit line, D. To get a quick image of what we're currently trying to visualize, take a look at the picture below:

# %%
# Mathematics behind the Least Squares Method.
tutorial = 'https://www.youtube.com/watch?v=Qa2APhWjQPc'


# %%
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/thumb/b/b0/Linear_least_squares_example2.svg/220px-Linear_least_squares_example2.svg.png'
Image(url)

# %% [markdown]
# 
# ## Step 4: Using Numpy for a Univariate Linear Regression
# 
# Numpy has a built in Least Square Method in its linear algebra library. We'll use this first for our Univariate regression and then move on to scikit learn for out Multi variate regression.
# 
# We will start by setting up the X and Y arrays for numpy to take in. An important note for the X array: Numpy expects a two-dimensional array, the first dimension is the different example values, and the second dimension is the attribute number. In this case we have our value as the mean number of rooms per house, and this is a single attribute so the second dimension of the array is just 1. So we'll need to create a (506,1) shape array. There are a few ways to do this, but an easy way to do this is by using numpy's built-in vertical stack tool, vstack.

# %%
# Liner regression with Pandas (all calculations)
X = boston_df.RM # values
X.shape


# %%
# v makes X two-dimensional
X = np.vstack(boston_df.RM) # attributes


# %%
X.shape # (values, attributes)


# %%
Y = boston_df.Price # Setup Y


# %%
X


# %%
# [X 1]
X = np.array( [ [value, 1] for value in X] )


# %%
X


# %%
# https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html

m, b = np.linalg.lstsq(X,Y)[0] # creating m and b variables for the formula


# %%
plt.plot(boston_df.RM,boston_df.Price,'o')

x = boston_df.RM

plt.plot(x, m*x + b,'r',label='Best Fit Line')


