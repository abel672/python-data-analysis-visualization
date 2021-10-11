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
# x=span, y=ham, hue=length
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# In this graph we can see that there is a correlation between the length of the message, and the fact that the message is spam.
# When longer the message, the probability to be an span email increases exponentially.
sns.catplot(x='length',data=messages, hue='labels',kind='count', height=9.27, aspect=14.7/11.27)

# %% [markdown]
# ## Part 3

# %%
import string


# %%
mess = 'Sample message! Notice: it has punctuation'


# %%
nopunc = [char for char in mess if char not in string.punctuation]


# %%
nopunc = ''.join(nopunc)


# %%
nopunc


# %%
from nltk.corpus import stopwords


# %%
stopwords.words('english')[0:10]


# %%
nopunc.split()


# %%
clean_message = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# %%
clean_message


# %%
def text_process(mess):
    '''
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    '''
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again from the string
    nopunc = ''.join(nopunc)

    # Now just remove the stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# %%
messages.head()


# %%
messages['message'].head(5).apply(text_process)

# %% [markdown]
# 
# ## Continuing Normalization
# 
# There are a lot of ways to continue normalizing this text. Such as [Stemming](https://en.wikipedia.org/wiki/Stemming) or distinguishing by [part of speech](http://www.nltk.org/book/ch05.html).
# 
# NLTK has lots of built-in tools and great documentation on a lot of these methods. Sometimes they don't work well for text-messages due to the way a lot of people tend to use abbreviations or shorthand, For example:
# 
# 'Nah dawg, IDK! Wut time u headin to da club?'
# 
# versus
# 
# 'No dog, I don't know! What time are you heading to the club?'
# 
# Some text normalization methods will have trouble with this type of shorthand and so I'll leave you to explore those more advanced methods through the [NLTK book online](http://www.nltk.org/book/).
# 
# For now we will just focus on using what we have to convert our list of words to an actual vector that SciKit-Learn can use.
# %% [markdown]
# ## Part 4: Vectorization
# 
# Currently, we have the messages as lists of tokens (also known as [lemmas](https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html)) and now we need to convert each of those messages into a vector the SciKit Learn's algorithm models can work with.
# 
# Now we'll convert each message, represented as a list of tokens (lemmas) above, into a vector that machine learning models can understand.
# 
# We'll do that in three steps using the bag-of-words model:
# 
# Count how many times does a word occur in each message (Known as term frequency)
# 
# Weigh the counts, so that frequent tokens get lower weight (inverse document frequency)
# 
# Normalize the vectors to unit length, to abstract from the original text length (L2 norm)
# 
# Let's begin the first step:
# 
# Each vector will have as many dimensions as there are unique words in the SMS corpus. We will first use SciKit Learn's CountVectorizer. This model will convert a collection of text documents to a matrix of token counts.
# 
# We can imagine this as a 2-Dimensional matrix. Where the 1-dimension is the entire vocabulary (1 row per word) and the other dimension are the actual documents, in this case a column per text message.
# 
# Since there are so many messages, we can expect a lot of zero counts for the presence of that word in that document. Because of this, SciKit Learn will output a [Sparse Matrix](https://en.wikipedia.org/wiki/Sparse_matrix).

# %%
from sklearn.feature_extraction.text import CountVectorizer


# %%
bow_transformer = CountVectorizer(analyzer=text_process)


# %%
bow_transformer.fit(messages['message'])


# %%
message4 = messages['message'][3]


# %%
print(message4)


# %%
bow4 = bow_transformer.transform([message4])


# %%
print(bow4) # worrds of the message4, and times that each of them appear in the message


# %%
print(bow_transformer.get_feature_names()[9554]) # you can look for the word also here


# %%
messages_bow = bow_transformer.transform(messages['message'])


# %%
print('Shape of Sparse Matrix: ', messages_bow.shape)
print('Amount of Non-Zero occurences: ', messages_bow.nnz)
print('Sparsity: %.2f%%' % (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1])))


# %%



