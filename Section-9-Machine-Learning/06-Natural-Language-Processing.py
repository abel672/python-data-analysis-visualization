# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# [Notebook](https://github.com/jmportilla/Udemy---Machine-Learning/blob/master/NLP%20(Natural%20Language%20Processing).ipynb)

# %%
import nltk


# %%
messages = [line.rstrip() for line in open('sms_spam_collection/SMSSpamCollection')]


# %%
print(len(messages))


# %%
for num, message in enumerate(messages[:10]):
    print(num, message)
    print('\n')


# %%
import pandas


# %%
messages = pandas.read_csv('sms_spam_collection/SMSSpamCollection',sep='\t',names=['labels','message'])


# %%
messages.head()


# %%
messages.describe()


# %%
messages.info()


# %%
messages.groupby('labels').describe()


# %%
# Feature engineering the 'length' column (Adding it as a new column)
messages['length'] = messages['message'].apply(len)
messages.head()


# %%
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# %%
messages['length'].plot(bins=50,kind='hist') # we see here that the x-axis continue almost until 1000, there must be some very long message in this frame. Let's look for it


# %%
messages['length'].describe() # first let's use describe to find out the biggest message (max column)


# %%
messages[messages['length'] == 910]['message'].iloc[0] # we have seen that the max length is 910, so we can use that param to get the message itself


# %%
messages.hist(column='length',by='labels',bins=50,figsize=(10,4))


# %%
# Exercise: Use an hist to determine if the length of a message is related with the fact that the message is a spam or not


