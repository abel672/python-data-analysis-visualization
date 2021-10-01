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

# %% [markdown]
# 
# ## Step 7: Using Training and Validation¶
# 
# In a dataset a training set is implemented to build up a model, while a validation set is used to validate the model built. Data points in the training set are excluded from the validation set. The correct way to pick out samples from your dataset to be part either the training or validation (also called test) set is randomly.
# 
# Fortunately, scikit learn has a built in function specifically for this called train_test_split.
# 
# The parameters passed are your X and Y, then optionally test_size parameter, representing the proportion of the dataset to include in the test split. As well a train_size parameter. ou can learn more about these parameters [here](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)

# %%
import sklearn.model_selection


# %%
# By splitting our data in train and test set, we can test how good our regression model is
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X,boston_df.Price)


# %%
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

# %% [markdown]
# ## Step 8: Predicting Prices
# 
# Now that we have our training and testing sets, let's go ahead and try to use them to predict house prices. We'll use our training set for the prediction and then use our testing set for validation.

# %%
lreg = LinearRegression()

lreg.fit(X_train,Y_train) # Fitting training data to this Linear Regression


# %%
# Predictions data sets
pred_train = lreg.predict(X_train)
pred_test = lreg.predict(X_test)


# %%
print("Fit a model X_train, and calculate the Mean Square Error (MSE) with Y_train: %.2f " % np.mean((Y_train - pred_train)**2))

print("Fit a model X_train, and calculate the Mean Square Error (MSE) with X_test and Y_test: %.2f " % np.mean((Y_test - pred_test)**2))

# %% [markdown]
# 
# ## Step 9 : Residual Plots
# 
# In regression analysis, the difference between the observed value of the dependent variable (y) and the predicted value (ŷ) is called the residual (e). Each data point has one residual, so that:
# 
# $$Residual = Observed\:value - Predicted\:value $$
# 
# 
# You can think of these residuals in the same way as the D value we discussed earlier, in this case however, there were multiple data points considered.
# 
# A residual plot is a graph that shows the residuals on the vertical axis and the independent variable on the horizontal axis. If the points in a residual plot are randomly dispersed around the horizontal axis, a linear regression model is appropriate for the data; otherwise, a non-linear model is more appropriate.
# 
# Residual plots are a good way to visualize the errors in your data. If you have done a good job then your data should be randomly scattered around line zero. If there is some strucutre or pattern, that means your model is not capturing some thing. There could be an interaction between 2 variables that you're not considering, or may be you are measuring time dependent data. If this is the case go back to your model and check your data set closely.
# 
# So now let's go ahead and create the residual plot. For more info on the residual plots check out this great [link](http://blog.minitab.com/blog/adventures-in-statistics/why-you-need-to-check-your-residual-plots-for-regression-analysis).

# %%
train = plt.scatter(pred_train,(pred_train - Y_train),c='b',alpha=0.5) # pred_train - Y_train = Residual

test = plt.scatter(pred_test,(pred_test - Y_test),c='r',alpha=0.5) # pred_test - Y_test = Residual

plt.hlines(y=0,xmin=-10,xmax=50,colors='black')

plt.legend((train, test),('Training','Test'),loc='lower left')

plt.title('Residual Plots')

# %% [markdown]
# That's it for this lesson. Linear regression is a very broad topic, theres a ton of great information in the sci kit learn documentation, and I encourage you to check it out here: [http://scikit-learn.org/stable/modules/linear_model.html#linear-model](http://scikit-learn.org/stable/modules/linear_model.html#linear-model)

