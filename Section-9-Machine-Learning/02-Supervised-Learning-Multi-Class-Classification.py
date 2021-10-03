# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# 
# ## Step 1: Introduction to the Iris Data Set¶
# 
# For this series of lectures, we will be using the famous Iris flower data set.
# 
# The Iris flower data set or Fisher's Iris data set is a multivariate data set introduced by Sir Ronald Fisher in the 1936 as an example of discriminant analysis.
# 
# The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor), so 150 total samples. Four features were measured from each sample: the length and the width of the sepals and petals, in centimetres.
# 
# Notebook: https://github.com/jmportilla/Udemy---Machine-Learning/blob/master/Multi-Class%20Classification.ipynb

# %%
# Data Imports
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

# Plot imports
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

get_ipython().run_line_magic('matplotlib', 'inline')


# %%
from sklearn import linear_model
from sklearn.datasets import load_iris


# %%
iris = load_iris()


# %%
X = iris.data

Y = iris.target


# %%
print(iris.DESCR)


# %%
iris_data = DataFrame(X, columns=['Sepal Length','Sepal Width','Petal Length','Petal Width'])


# %%
iris_target = DataFrame(Y, columns=['Species'])


# %%
def flower(num):
    if num == 0:
        return 'Setosa'
    elif num == 1:
        return 'Versicolour'
    else:
        return 'Virginica'


# %%
iris_target['Species'] = iris_target['Species'].apply(flower)


# %%
iris_target.head()


# %%
iris = pd.concat([iris_data,iris_target],axis=1) # merging both dataframes together


# %%
iris.head()


# %%
sns.pairplot(iris, hue='Species',height=2)


# %%
sns.catplot(x='Petal Length',data=iris,hue='Species',kind='count',height=10)


