# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import pandas as pd
from pandas import Series, DataFrame


# %%
titanic_df = pd.read_csv('train.csv')


# %%
titanic_df.head()


# %%
titanic_df.info()

# %% [markdown]
# ### All good data analysis projects begin with trying to answer questions. Now that we know what column category data we have let's think of some questions or insights we would like to obtain from the data. So here's a list of questions we'll try to answer using our new data analysis skills!
# 
# First some basic questions:
# 
# 1.) Who were the passengers on the Titanic? (Ages,Gender,Class,..etc)
# 
# 2.) What deck were the passengers on and how does that relate to their class?
# 
# 3.) Where did the passengers come from?
# 
# 4.) Who was alone and who was with family?
# 
# Then we'll dig deeper, with a broader question:
# 
# 5.) What factors helped someone survive the sinking?
# 
# So let's start with the first question: Who were the passengers on the titanic?

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# %%
sns.catplot(x='Sex',kind="count", data=titanic_df) # woman and men


# %%
sns.catplot(x='Sex', data=titanic_df, kind='count', hue='Pclass') # women and men by Pclass column


# %%
# better way to show it
sns.catplot(x='Pclass', data=titanic_df, kind='count', hue='Sex') # women and men by classes


