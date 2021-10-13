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

# %% [markdown]
# After the counting, the term weighting and normalization can be done with [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf), using scikit-learn's TfidfTransformer.
# %% [markdown]
# ## So what is TF-IDF?¶
# 
# TF-IDF stands for term frequency-inverse document frequency, and the tf-idf weight is a weight often used in information retrieval and text mining. This weight is a statistical measure used to evaluate how important a word is to a document in a collection or corpus. The importance increases proportionally to the number of times a word appears in the document but is offset by the frequency of the word in the corpus. Variations of the tf-idf weighting scheme are often used by search engines as a central tool in scoring and ranking a document's relevance given a user query.
# 
# One of the simplest ranking functions is computed by summing the tf-idf for each query term; many more sophisticated ranking functions are variants of this simple model.
# 
# Typically, the tf-idf weight is composed by two terms: the first computes the normalized Term Frequency (TF), aka. the number of times a word appears in a document, divided by the total number of words in that document; the second term is the Inverse Document Frequency (IDF), computed as the logarithm of the number of the documents in the corpus divided by the number of documents where the specific term appears.
# 
# **TF: Term Frequency**, which measures how frequently a term occurs in a document. Since every document is different in length, it is possible that a term would appear much more times in long documents than shorter ones. Thus, the term frequency is often divided by the document length (aka. the total number of terms in the document) as a way of normalization:
# 
# TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document).
# 
# **IDF: Inverse Document Frequency**, which measures how important a term is. While computing TF, all terms are considered equally important. However it is known that certain terms, such as "is", "of", and "that", may appear a lot of times but have little importance. Thus we need to weigh down the frequent terms while scale up the rare ones, by computing the following:
# 
# IDF(t) = log_e(Total number of documents / Number of documents with term t in it).
# 
# See below for a simple example.
# 
# **Example:**
# 
# Consider a document containing 100 words wherein the word cat appears 3 times.
# 
# The term frequency (i.e., tf) for cat is then (3 / 100) = 0.03. Now, assume we have 10 million documents and the word cat appears in one thousand of these. Then, the inverse document frequency (i.e., idf) is calculated as log(10,000,000 / 1,000) = 4. Thus, the Tf-idf weight is the product of these quantities: 0.03 * 4 = 0.12.
# %% [markdown]
# ## Lecture 4

# %%
from sklearn.feature_extraction.text import TfidfTransformer # Inverse Document Frequency

tfidf_transformer = TfidfTransformer().fit(messages_bow) # Training the model


# %%
tfidf4 = tfidf_transformer.transform(bow4) # Transforming a bag of words


# %%
print(tfidf4)


# %%
print(tfidf_transformer.idf_[bow_transformer.vocabulary_['u']]) # Finding out specific Inverse Document Frequency Number


# %%
messages_tfidf = tfidf_transformer.transform(messages_bow) # Transforming all bag of words


# %%
print(messages_tfidf.shape)

# %% [markdown]
# ## Part 5: Training a model¶
# 
# With messages represented as vectors, we can finally train our spam/ham classifier. Now we can actually use almost any sort of classification algorithms. For a [variety of reasons](http://www.inf.ed.ac.uk/teaching/courses/inf2b/learnnotes/inf2b-learn-note07-2up.pdf), the Naive Bayes classifier algorithm is a good choice.
# 
# We'll be using scikit-learn here, choosing the [Naive Bayes classifier](http://en.wikipedia.org/wiki/Naive_Bayes_classifier) to start with:

# %%
from sklearn.naive_bayes import MultinomialNB


# %%
spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['labels'])


# %%
print('Predicted: ', spam_detect_model.predict(tfidf4)[0]) # Predicted result
print('Expected: ', messages['labels'][3]) # Real result from our set

# %% [markdown]
# Fantastic! We've developed a model that can attempt to predict spam vs ham classification!
# %% [markdown]
# ## Part 6: Model Evaluation¶
# 
# Now we want to determine how well our model will do overall on the entire dataset. Let's beginby getting all the predictions:

# %%
all_predictions = spam_detect_model.predict(messages_tfidf)
print(all_predictions)

# %% [markdown]
# We can use SciKit Learn's built-in classification report, which returns [precision and recall](https://www.youtube.com/watch?v=qWfzIYCvBqo&ab_channel=KimberlyFessel), [f1-score](https://www.youtube.com/watch?v=8d3JbbSj-I8&ab_channel=Scarlett%27sLog), and a column for support (meaning how many cases supported that classification). Check out the links for more detailed info on each of these metrics and the figure below:

# %%
# metrics
from sklearn.metrics import classification_report
print(classification_report(messages['labels'], all_predictions))


# %%
from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['labels'], test_size=0.2)


# %%
print(len(msg_train), len(msg_test), len(msg_train), len(msg_test))

# %% [markdown]
# ## Part 7: Creating a Data Pipeline¶
# 
# Let's run our model again and then predict off the test set. We will use SciKit Learn's [pipeline](http://scikit-learn.org/stable/modules/pipeline.html) capabilities to store a pipeline of workflow. This will allow us to set up all the transformations that we will do to the data for future use. Let's see an example of how it works:

# %%
from sklearn.pipeline import Pipeline


# %%
pipeline = Pipeline([('bow', CountVectorizer(analyzer=text_process)),
                        ('tfidf', TfidfTransformer()),
                        ('classifier', MultinomialNB())])


# %%
pipeline.fit(msg_train, label_train)


# %%
predictions = pipeline.predict(msg_test)


# %%
print(classification_report(predictions, label_test))

# %% [markdown]
# 
# ## More Resources
# 
# Check out the links below for more info on Natural Language Processing:
# 
# [NLTK Book Online](http://www.nltk.org/book/)
# 
# [Kaggle Walkthrough](https://www.kaggle.com/c/word2vec-nlp-tutorial)
# 
# [SciKit Learn's Tutorial](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)

