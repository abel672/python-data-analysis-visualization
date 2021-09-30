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
# Liner Regression with Seaborn
sns.lmplot(x='RM',y='Price',data=boston_df)

# %% [markdown]
# 
# ## Step 3: The mathematics behind the Least Squares Method.¶
# In this particular lecture we'll use the least squares method as the way to estimate the coefficients. Here's a quick breakdown of how this method works mathematically:
# 
# Take a quick look at the plot we created above using seaborn. Now consider each point, and know that they each have a coordinate in the form (X,Y). Now draw an imaginary line between each point and our current "best-fit" line. We'll call the distanace between each point and our current best-fit line, D. To get a quick image of what we're currently trying to visualize, take a look at the picture below:

# %%
# Mathematics behind the Least Squares Method. tutorial: https://www.youtube.com/watch?v=Qa2APhWjQPc


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
# X = X.dtype('float64')


# %%
# [X 1]
X = np.array( [ [value, float(1)] for value in X], dtype='float64' )
# X = np.vstack([X, np.ones(len(X))]).T


# %%
X


# %%
# https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html
# Least Square Method solution (R^2). tutorial: https://www.youtube.com/watch?v=Qa2APhWjQPc (same one from the beggining of Step 3)

m, b = np.linalg.lstsq(X,Y,rcond=None)[0] # creating m and b variables for the formula


# %%
plt.plot(boston_df.RM,boston_df.Price,'o')

x = boston_df.RM

plt.plot(x, m*x + b,'r',label='Best Fit Line')

# %% [markdown]
# 
# ## Step 5: Getting the error
# 
# Great! We've just completed a single variable regression using the least squares method with Python! Let's see if we can find the error in our fitted line. Checking out the documentation here, we see that the resulting array has the total squared error. For each element, it checks the the difference between the line and the true value (our original D value), squares it, and returns the sum of all these. This was the summed D^2 value we discussed earlier.
# 
# It's probably easier to understand the root mean squared error, which is similar to the standard deviation. In this case, to find the root mean square error we divide by the number of elements and then take the square root. There is also an issue of bias and an unbiased regression, but we'll delve into those topics later.
# 
# For now let's see how we can get the root mean squared error of the line we just fitted.

# %%
result = np.linalg.lstsq(X,Y,rcond=None)

error_total = result[1]

rmse = np.sqrt(error_total / len(X))

print("The room mean square error was %.2f" %rmse)

# 68,95,99.7 rule tutorial: https://www.youtube.com/watch?v=mtbJbDwqWLE&ab_channel=SimpleLearningPro

# %% [markdown]
# ## Step 6: Using scikit learn to implement a multivariate regression
# 
# Now, we'll keep moving along with using scikit learn to do a multi variable regression. This will be a similar apporach to the above example, but sci kit learn will be able to take into account more than just a single data variable effecting the target!
# 
# We'll start by importing the linear regression library from the sklearn module.
# 
# The sklearn.linear_model.LinearRegression class is an estimator. Estimators predict a value based on the observed data. In scikit-learn, all estimators implement the fit() and predict() methods. The former method is used to learn the parameters of a model, and the latter method is used to predict the value of a response variable for an explanatory variable using the learned parameters. It is easy to experiment with different models using scikit-learn because all estimators implement the fit and predict methods.

# %%
# Import for Linear Regression
import sklearn
from sklearn.linear_model import LinearRegression


# %%
# R-Square (Cofficient of determination) tutorial: https://www.youtube.com/watch?v=2AQKmw14mHM&ab_channel=StatQuestwithJoshStarmer
lreg = LinearRegression()


# %%
X_multi = boston_df.drop('Price', 1)

Y_target = boston_df.Price


# %%
# Implement Linear Regression
lreg.fit(X_multi, Y_target)


# %%
print('The estimated intercept coefficient is %.2f' % lreg.intercept_)

print('The numerb for coefficients used was %d' % len(lreg.coef_))


# %%
coeff_df = DataFrame(boston_df.columns)
coeff_df.columns = ['Features']

coeff_df['Coefficient Estimate'] = Series(lreg.coef_)

coeff_df


