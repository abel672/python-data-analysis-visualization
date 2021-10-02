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


# %%



