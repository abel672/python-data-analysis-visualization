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


# %%
# Who was alone and who was with family?
titanic_df.head()


# %%
titanic_df['Alone'] = titanic_df.SibSp + titanic_df.Parch


# %%
titanic_df['Alone']


# %%
titanic_df['Alone'].loc[titanic_df['Alone'] > 0] = 'With Family'

titanic_df['Alone'].loc[titanic_df['Alone'] == 0] = 'Alone'


# %%
titanic_df.head()


# %%
sns.catplot('Alone', data=titanic_df, palette='Blues', kind='count')


# %%
# What factors helped someone survive the sinking?
titanic_df['Survivor'] = titanic_df.Survived.map({0:'no',1:'yes'})

sns.catplot('Survivor',data=titanic_df,palette='Set1',kind='count')


# %%
sns.catplot(x='Pclass',y='Survived',data=titanic_df,kind='point') # catalog point plot considering class


# %%
sns.catplot(x='Pclass',y='Survived',hue='person',data=titanic_df,kind='point') # catalog point plot considering class and gender


# %%
sns.lmplot(x='Age',y='Survived',data=titanic_df)


# %%
sns.lmplot(x='Age',y='Survived',hue='Pclass',data=titanic_df,palette='winter')


# %%
generations = [10,20,40,60,80]

sns.lmplot(x='Age',y='Survived',hue='Pclass',data=titanic_df,palette='winter',x_bins=generations)


# %%
sns.lmplot(x='Age',y='Survived',hue='Sex',data=titanic_df,palette='winter',x_bins=generations)

# %% [markdown]
# 
# 1.) Did the deck have an effect on the passengers survival rate? Did this answer match up with your intuition?
# 
# 2.) Did having a family member increase the odds of suriving the crash?

# %%
titanic_df.head()


# %%
# Add a 'deck' column into our current data frame
def getDeckLetter(cabin):
    """This gets the cabin from titanic_df['Cabin'] object"""
    if type(cabin) == float:
        return np.nan
    elif cabin[0] == 'T':
        return np.nan
    else:
        return cabin[0]

# using apply
titanic_df['Deck'] = titanic_df['Cabin'].apply(getDeckLetter)


# %%
titanic_df.head()


# %%
titanic_deck = titanic_df.dropna(subset=['Deck']) # Getting cleaned decks from titanic data frame
titanic_deck.head()


# %%
sns.lmplot(x='Age',y='Survived',hue='Deck',data=titanic_deck,palette='winter',
            hue_order=['A','B','C','D','E','F','G'], x_bins=generations).set(ylim=[-0.4,1.4])


# %%
sns.catplot(x='Deck',y='Survived',data=titanic_deck,palette='winter',order=['A','B','C','D','E','F','G'],kind='point')


# %%
sns.catplot(x='Survivor',hue='Deck',data=titanic_deck,palette='winter',hue_order=['A','B','C','D','E','F','G'],kind='count')


# %%
sns.catplot(x='Survivor',data=titanic_deck,palette='winter',kind='count')


# %%
# Let's see who has deck information
def getDeckInfoPresent(cabin):
    """This gets the cabin from titanic_df['Cabin'] object"""
    if type(cabin) == float:
        return 'No'
    elif cabin[0] == 'T':
        return 'No'
    else:
        return 'Yes'

# using apply
titanic_df['DeckInformation'] = titanic_df['Cabin'].apply(getDeckInfoPresent)


# %%
sns.catplot(x='Survivor',hue='DeckInformation',data=titanic_df,palette='Set1',kind='count',
            order=['yes','no'],hue_order=['Yes','No'])


# %%
sns.catplot(x='Pclass',hue='DeckInformation',data=titanic_df,palette='Set1',kind='count',
            order=[1,2,3],hue_order=['Yes','No'])


# %%
sns.catplot(x='Pclass',hue='Deck',data=titanic_deck,palette='Set1',kind='count',
            order=[1,2,3],hue_order=['A','B','C','D','E','F','G'])


# %%
# Most of the data for the deck is in the first class only. Let's filter it to see it more in detail
titanic_deck = titanic_deck[titanic_deck['Pclass']==1]

sns.catplot(x='Pclass',hue='Deck',data=titanic_deck,palette='Set1',kind='count')


# %%
sns.catplot(x='Survivor',hue='Deck',data=titanic_deck,palette='winter',
            hue_order=['A','B','C','D','E'],kind='count')


# %%
# Create a deck survival DataFrame
decks = ['A','B','C','D','E']

adecks = {}
deck_survival = {}
total_on_deck = {}
rel_deck_survival = {}

for deck in decks:
    adecks[deck] = deck
    deck_survival[deck] = titanic_deck['Deck'].loc[titanic_deck['Deck'] == deck].loc[titanic_deck['Survived'] == 1].count()
    total_on_deck[deck] = titanic_deck['Deck'].loc[titanic_deck['Deck'] == deck].count()
    rel_deck_survival[deck] = deck_survival[deck] / total_on_deck[deck]

deck_survival_df = DataFrame({'Deck':adecks, 'Deck Survival':deck_survival, 'Total on Deck':total_on_deck, 'Percentage':rel_deck_survival})
deck_survival_df.head()


# %%
sns.barplot(x='Deck',y='Percentage',data=deck_survival_df,palette='winter')


# %%
sns.catplot(x='Sex',hue='Deck',data=titanic_deck,palette='winter',order=['male','female'],
            hue_order=['A','B','C','D','E'],kind='count')


# %%
#2.) Did having a family member increase the odds of suriving the crash?
titanic_df.head()


# %%
sns.catplot(x='Alone',hue='Survivor',data=titanic_df,palette='winter',hue_order=['yes','no'],kind='count')


# %%
titanic_df[titanic_df['Survived']==1].count()


