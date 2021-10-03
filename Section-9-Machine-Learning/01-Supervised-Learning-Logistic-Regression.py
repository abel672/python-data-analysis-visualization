# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
# for Data
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

# Math
import math

# Plot imports
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')

# Machine Learning Imports
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# For evaluating our ML results
from sklearn import metrics

# Dataset Import
import statsmodels.api as sm

# %% [markdown]
# ## Part 1: Basic Mathematical Overview
# 
# First, let's take a look at the [Logistic Function](https://en.wikipedia.org/wiki/Logistic_function). The logistic function can take an input from negative to positive infinity and it has always has an output between 0 and 1. The logistic function is defined as:$$ \sigma (t)= \frac{1}{1+e^{-t}}$$
# 
# Here a [tutorial](https://www.youtube.com/watch?v=TPqr8t919YM&ab_channel=PowerH)

# %%
# Logistic function
def logistic(t):
    return 1.0 / (1 + math.exp((-1.0)*t) )

# Set t from -6 to 6 ( 500 elements, linearly spaced)
t = np.linspace(-6,6,500)

# Set up y values (using list comprehesion) explanation: https://www.w3schools.com/python/python_lists_comprehension.asp
y = np.array([logistic(ele) for ele in t])

# Plot
plt.plot(t,y)
plt.title(' Logistic Function ')

# %% [markdown]
# If we remember back to the [Linear Regression](https://en.wikipedia.org/wiki/Linear_regression) Lectures, we could describe a Linear Regression Function model as:$$ y_i = \beta _1 x_{i1} + ... + \beta _i x_{ip}$$
# 
# Which was basically an expanded linear equation (y=mx+b) for various x data features. In the case of the above equation, we presume a data set of 'n' number of units, so that the data set would have the form:$$ [ y_i, x_{i1},...,x_{ip}]^{n}_{i=1}$$
# 
# For our logistic function, if we view t as a linear function with a variable x we could express t as:$$ t = \beta _0 + \beta _1 x $$
# 
# Here, we've basically just substituted a linear function (form similar to y=mx+b) for t. We could then rewrite our logistic function equation as:$$ F(x)= \frac{1}{1+e^{-(\beta _0 + \beta _1 x)}}$$
# 
# Now we can interpret F(x) as the probability that the dependent variable is a "success" case, this is a similar style of thinking as in the Binomial Distribution, in which we had successes and failures. So the formula for F(x) that we have here states that the probability of the dependent variable equaling a "success" case is equal to the value of the logistic function of the linear regression expression (the linear equation we used to replace t ).
# 
# Inputting the linear regression expression into the logistic function allows us to have a linear regression expression value that can vary from positive to negative infinity, but after the transformation due to the logistic expression we will have an output of F(x) that ranges from 0 to 1.
# 
# We can now perform a binary classification based on where F(x) lies, either from 0 to 0.5, or 0.5 to 1.
# %% [markdown]
# ## Part 2: Extra Math Resources
# 
# This is a very basic overview of binary classification using Logistic Regression, if you're still interested in a deeper dive into the mathematics, check out these sources:
# 
# 1.) [Andrew Ng's class notes](http://cs229.stanford.edu/notes2020spring/cs229-notes1.pdf) on Logistic Regression (Note: Scroll down)
# 
# 2.) [CMU notes Note](http://www.stat.cmu.edu/~cshalizi/uADA/12/lectures/ch12.pdf): Advanced math notation.
# 
# 3.) [Wikipedia](https://en.wikipedia.org/wiki/Logistic_regression) has a very extensive look at logistic regression.
# 
# 4.) Logistic Regression [entire tutorial](https://www.youtube.com/watch?v=yIYKR4sgzI8&list=PLblh5JKOoLUKxzEP5HA2d-Li7IJkHfXSe&ab_channel=StatQuestwithJoshStarmer)
# 
# Scroll down to the bottom for more resources similar to this lecture!
# %% [markdown]
# ## Part 3: Dataset Analysis
# 
# Let us go ahead and take a look at the [dataset](https://www.statsmodels.org/stable/datasets/generated/fair.html)

# %%
df = sm.datasets.fair.load_pandas().data


# %%
df.head()


# %%
def affair_check(x):
    if x != 0:
        return 1
    else:
        return 0


# %%
df['Had_Affair'] = df['affairs'].apply(affair_check)


# %%
df


# %%
df.groupby('Had_Affair').mean()


# %%
# It's always a good idea to visualize your data before performing some analysis on it
sns.catplot(x='age',data=df,hue='Had_Affair',palette='coolwarm',kind='count')


# %%
sns.catplot(x='yrs_married',data=df,hue='Had_Affair',palette='coolwarm',kind='count')


# %%
sns.catplot(x='children',data=df,hue='Had_Affair',palette='coolwarm',kind='count')


# %%
sns.catplot(x='educ',data=df,hue='Had_Affair',palette='coolwarm',kind='count')


# %%
sns.catplot(x='religious',data=df,hue='Had_Affair',palette='coolwarm',kind='count')

# %% [markdown]
# ## Part 5: Data Preparation
# 
# If we look at the data, we'll notice that two columns are unlike the others. Occupation and Husband's Occupation. These columns are in a format know as Categorical Variables. Basically they are in set quantity/category, so that 1.0 and 2.0 are separate variables, not values along a spectrum that goes from 1-2 (e.g. There is no 1.5 for the occupation column). Pandas has a built-in method of getting [dummy variables](https://www.youtube.com/watch?v=bnjPzHQ04Ac&ab_channel=DATAtab) and creating new columns from them.

# %%
occ_dummies = pd.get_dummies(df['occupation'])


# %%
husb_occ_dummies = pd.get_dummies(df['occupation_husb'])


# %%
occ_dummies.head()


# %%
occ_dummies.columns = ['occ1','occ2','occ3','occ4','occ5','occ6']


# %%
husb_occ_dummies.columns = ['hocc1','hocc2','hocc3','hocc4','hocc5','hocc6']


# %%
X = df.drop(['occupation','occupation_husb','Had_Affair'],axis=1)


# %%
dummies = pd.concat([occ_dummies,husb_occ_dummies],axis=1)


# %%
X.head()


# %%
X = pd.concat([X,dummies],axis=1)


# %%
X.head()


# %%
Y = df.Had_Affair

Y.tail()

# %% [markdown]
# Part 6: Multicollinearity Consideration.
# 
# Now we need to get rid of a few columns. We will be dropping the occ1 and hocc1 columns to avoid [multicollinearity](https://en.wikipedia.org/wiki/Multicollinearity#Remedies_for_multicollinearity). Multicollinearity occurs due to the dummy variables) we created. This is because the [dummy variables](https://en.wikipedia.org/wiki/Dummy_variable_%28statistics%29) are highly correlated, our model begins to get distorted because one of the dummy variables can be linearly predicted from the others. We take care of this problem by dropping one of the dummy variables from each set, we do this at the cost of losing a data set point.
# 
# The other column we will drop is the affairs column. This is because it is basically a repeat of what will be our Y target, instead of 0 and 1 it just has 0 or a number, so we'll need to drop it for our target to make sense.

# %%
# Droping one column of each dummy variable set to avoid multicollinearity
X = X.drop('occ1',axis=1)
X = X.drop('hocc1',axis=1)

# Drop affairs column so Y target make sense
X = X.drop('affairs',axis=1)


# %%
X.head()


# %%
Y.head()

# %% [markdown]
# In order to use the Y with SciKit Learn, we need to set it as a 1-D array. This means we need to "flatten" the array. Numpy has a built in method for this called [ravel](https://numpy.org/doc/stable/reference/generated/numpy.ravel.html). Let's use it!

# %%
Y = np.ravel(Y)

Y


# %%
log_model = LogisticRegression(solver='lbfgs', max_iter=1000)

log_model.fit(X,Y)

log_model.score(X,Y) # 73% accuracy

# %% [markdown]
# Looks like we got a 73% accuracy rating. Let's go ahead and compare this to the original Y data. We can do this by simply taking the mean of the Y data, since it is in the format 1 or 0, we can use the mean to calulate the percentage of women who reported having affairs. This is known as checking the [null error rate](https://www.youtube.com/watch?v=a_l991xUAOU&ab_channel=365DataScience).
# 
# 

# %%
Y.mean() # Percentage of women that had affairs


# %%
coeff_df = DataFrame(zip(X.columns,np.transpose(log_model.coef_)))


# %%
coeff_df # 0 =  Increment of the value, 1 = increment of the affair accordingly (negative or positive)

# %% [markdown]
# 
# Looking at the coefficients we can see that a positive coeffecient corresponds to increasing the [likelihood](https://www.youtube.com/watch?v=XepXtl9YKwc&ab_channel=StatQuestwithJoshStarmer) of having an affair while a negative coefficient means it corresponds to a decreased likelihood of having an affair as the actual data value point increases.
# 
# As you might expect, an increased marriage rating corresponded to a decrease in the likelihood of having an affair. Increased religiousness also seems to correspond to a decrease in the likelihood of having an affair.
# 
# Since all the dummy variables (the wife and husband occupations) are positive that means the lowest likelihood of having an affair corresponds to the baseline occupation we dropped (1-Student).
# %% [markdown]
# ## Part 8: Testing and Training Data Sets¶
# 
# Just like we did in the Linear Regression Lecture, we should be splitting our data into training and testing data sets. We'll follow a very similar procedure to the Linear Regression Lecture by using SciKit Learn's built-in train_test_split method.

# %%
X_train,X_test,Y_train,Y_test = train_test_split(X,Y)


# %%
log_model2 = LogisticRegression(solver='lbfgs', max_iter=1000)

log_model2.fit(X_train, Y_train)


# %%
class_predict = log_model2.predict(X_test)


# %%
# Compare the predicted classes to the actual test classes
print(metrics.accuracy_score(Y_test,class_predict))

# %% [markdown]
# Now we have a 73.35% accuracy score, which is basically the same as our previous accuracy score, 72.58%.
# %% [markdown]
# ## Part 9: Conclusion and more Resources
# 
# So what could we do to try to further improve our Logistic Regression model? We could try some [regularization techniques](https://en.wikipedia.org/wiki/Regularization_%28mathematics%29#Regularization_in_statistics_and_machine_learning) or using a non-linear model.
# 
# I'll leave the Logistic Regression topic here for you to explore more possibilites on your own. Here are several more resources and tutorials with other data sets to explore:
# 
# 1.) Here's another great post on how to do logistic regression analysis using Statsmodels from yhat!
# 
# 2.) The SciKit learn Documentation includes several [examples](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) at the bottom of the page.
# 
# 3.) DataRobot has a great [overview](https://www.datarobot.com/blog/classification-with-scikit-learn/) of Logistic Regression
# 
# 4.) Fantastic resource from [aimotion.blogspot](http://aimotion.blogspot.com/2011/11/machine-learning-with-python-logistic.html) on the Logistic Regression and the Mathematics of how it relates to the cost function and gradient!

