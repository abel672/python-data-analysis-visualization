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


# %%
def male_female_child(passenger):
    age, sex = passenger

    if age < 16:
        return 'child'
    else:
        return sex


# %%
# Adding a 'person' column
titanic_df['person'] = titanic_df[['Age','Sex']].apply(male_female_child, axis=1)


# %%
titanic_df[0:10]


# %%
sns.catplot(x='Pclass',data=titanic_df, kind='count',hue='person') # plot with the 'person' column


# %%
titanic_df['Age'].hist(bins=70) # age histogram


# %%
titanic_df['Age'].mean()


# %%
titanic_df['person'].value_counts()


# %%
# Creating facade grid
fig = sns.FacetGrid(titanic_df,hue='person',aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)

oldest = titanic_df['Age'].max()

fig.set(xlim=(0,oldest))

fig.add_legend()


# %%
fig = sns.FacetGrid(titanic_df,hue='Pclass',aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)

oldest = titanic_df['Age'].max()

fig.set(xlim=(0,oldest))

fig.add_legend()


# %%
titanic_df.head()


# %%
# check in which decks the passengers were
deck = titanic_df['Cabin'].dropna() # drop null values


# %%
deck.head()


# %%
levels = []

for level in deck:
    levels.append(level[0])

cabin_df = DataFrame(levels)

cabin_df.columns = ['Cabin']

sns.catplot(x='Cabin',data=cabin_df,palette='winter_d', kind='count')


# %%
cabin_df = cabin_df[cabin_df.Cabin != 'T']
sns.catplot(x='Cabin', kind='count', data=cabin_df, palette='summer')


# %%
titanic_df.head()


# %%
# where did the passengers came from?
sns.catplot(x='Embarked',data=titanic_df,hue='Pclass',order=['C','Q','S'],kind='count')


