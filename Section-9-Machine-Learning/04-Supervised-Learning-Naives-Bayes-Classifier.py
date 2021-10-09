# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# 
# ## Part 1: Note on Notation and Math Terms
# 
# There are a few more advanced notations and amthematical terms used during the explanation of naive Bayes Classification. You should be familiar with the following:
# 
# Product of Sequence
# 
# The product of a sequence of terms can be written with the product symbol, which derives from the capital letter Π (Pi) in the Greek alphabet. The meaning of this notation is given by:$$\prod_{i=1}^4 i = 1\cdot 2\cdot 3\cdot 4,  $$that is$$\prod_{i=1}^4 i = 24. $$
# 
# Arg Max
# 
# In mathematics, the argument of the maximum (abbreviated arg max or argmax) is the set of points of the given argument for which the given function attains its maximum value. In contrast to global maximums, which refer to a function's largest outputs, the arg max refers to the inputs which create those maximum outputs.
# 
# The arg max is defined by
# 
# $$\operatorname*{arg\,max}_x  f(x) := \{x \mid \forall y : f(y) \le f(x)\}$$
# In other words, it is the set of points x for which f(x) attains its largest value. This set may be empty, have one element, or have multiple elements. For example, if f(x) is 1−|x|, then it attains its maximum value of 1 at x = 0 and only there, so
# 
# $$\operatorname*{arg\,max}_x (1-|x|) = \{0\}$$
# %% [markdown]
# ## Part 2: Bayes' Theorem
# 
# First, for a quick introduction to Bayes' Theorem, check out the [Bayes' Theorem Lecture](https://github.com/jmportilla/Statistics-Notes/blob/master/Bayes'%20Theorem.ipynb) in the statistics appendix portion of this course, in order ot fully understand Naive Bayes, you'll need a complete understanding of the Bayes' Theorem. Also this [article](https://www.countbayesie.com/blog/2015/2/18/bayes-theorem-with-lego) and this [tutorial](https://www.youtube.com/watch?v=O2L2Uv9pdDA&ab_channel=StatQuestwithJoshStarmer)
# %% [markdown]
# ## Part 4: Naive Bayes Classifier Mathematics Overview
# 
# Naive Bayes methods are a set of supervised learning algorithms based on applying Bayes’ theorem with the “naive” assumption of independence between every pair of features. Given a class variable y and a dependent feature vector x1 through xn, Bayes’ theorem states the following relationship:
# 
# $$P(y \mid x_1, \dots, x_n) = \frac{P(y) P(x_1, \dots x_n \mid y)}
#                                  {P(x_1, \dots, x_n)}$$
# Using the naive independence assumption that$$P(x_i | y, x_1, \dots, x_{i-1}, x_{i+1}, \dots, x_n) = P(x_i | y)$$
# 
# for all i, this relationship is simplified to:
# 
# $$P(y \mid x_1, \dots, x_n) = \frac{P(y) \prod_{i=1}^{n} P(x_i \mid y)}
#                                  {P(x_1, \dots, x_n)}$$
# We now have a relationship between the target and the features using Bayes Theorem along with a Naive Assumption that all features are independent.
# %% [markdown]
# ## Part 7: Gaussian Naive Bayes with SciKit Learn
# 
# We'll start by importing the usual.

# %%
import pandas as pd
from pandas import Series, DataFrame

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB


# %%
# load the iris dataset
iris = datasets.load_iris()

X = iris.data

Y = iris.target

print(iris.DESCR)


# %%
model = GaussianNB() # Fit a Naive Bayes model to the data


# %%
# Split the data into Training and Testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)


# %%
# Fit the training model
model.fit(X_train, Y_train)


# %%
# Predicted outcomes
predicted = model.predict(X_test)

# Actual Expected Outcomes
expected = Y_test


# %%
# See the metrics for performance
print(metrics.accuracy_score(expected, predicted))

# %% [markdown]
# ## Part 8: More Resources¶
# 
# There are plenty more resources and types of Naive Bayes Classifiers, For more resources on Naive Bayes, check out the following links:
# 
# 1.) [SciKit Learn Documentation](https://scikit-learn.org/stable/modules/naive_bayes.html)
# 
# 2.) [Naive Bayes with NLTK](http://slendermeans.org/ml4h-ch3.html)
# 
# 3.) Andrew Ng's Class Notes
# 
# 4.) [Andrew Ng's Video Lecture on Naive Bayes](https://www.youtube.com/watch?v=z5UQyCESW64&ab_channel=WangZhiyang)
# 
# 5.) [UC Berkeley Lecture by Pieter Abbeel](https://www.youtube.com/watch?v=DNvwfNEiKvw&ab_channel=CS188Spring2013)

