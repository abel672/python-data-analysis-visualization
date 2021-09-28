# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
from __future__ import division

# For data
import pandas as pd
from pandas import Series, DataFrame
import numpy as np

# For visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# %%
import requests

from io import StringIO


# %%
url = 'https://elections.huffingtonpost.com/pollster/2012-general-election-romney-vs-obama.csv'

source = requests.get(url, verify=False).text

poll_data = StringIO(source)


# %%
poll_df = pd.read_csv(poll_data)


# %%
poll_df.info()


# %%
poll_df.head()


# %%
sns.catplot(x='Affiliation',data=poll_df,kind='count')


# %%
sns.catplot(x='Affiliation',data=poll_df,hue='Population',kind='count')


# %%
poll_df.head()


# %%
avg = pd.DataFrame(poll_df.mean())

avg.drop('Number of Observations',axis=0,inplace=True)
# std.drop('Question Text',axis=0,inplace=True)
# std.drop('Question Iteration',axis=0,inplace=True)


# %%
avg.head()


# %%
std = pd.DataFrame(poll_df.std())
std.drop('Number of Observations',axis=0,inplace=True)


# %%
std.head()


# %%
avg.plot(yerr=std,kind='bar',legend=False) # plot avg and std together in a plot


# %%
poll_avg = pd.concat([avg,std],axis=1)


# %%
poll_avg.columns = ['Average','STD']


# %%
poll_avg


# %%
# Quick time series analysis
poll_df.head()


# %%
poll_df.plot(x='End Date',y=['Obama','Romney','Undecided'],linestyle='',marker='o')


# %%
# difference vs time
from datetime import datetime


# %%
poll_df['Difference'] = (poll_df.Obama - poll_df.Romney) / 100

poll_df.head()


# %%
poll_df = poll_df.groupby(['Start Date'],as_index=False).mean() # To keep the original indexes

poll_df.head()


# %%
poll_df.plot(x='Start Date',y='Difference',figsize=(12,4),marker='o',linestyle='-',color='purple')


# %%
# Exercise: Look for the particular dates when Romney won .min() is the hint for this.
lower_rate = poll_df['Difference'].min()
poll_df.loc[poll_df['Difference'] == lower_rate]


# %%
# Markdown debates in October 2012
row_in = 0
xlimit = []

for date in poll_df['Start Date']:
    if date[0:7] == '2012-10':
        xlimit.append(row_in)
        row_in += 1
    else:
        row_in += 1

xlimit_min = min(xlimit)
xlimit_max = max(xlimit)

print(xlimit_min)
print(xlimit_max)


# %%
poll_df.plot(x='Start Date',y='Difference',figsize=(12,4),marker='o',linestyle='-',color='purple',xlim=(xlimit_min, xlimit_max))

# Oct 3rd
plt.axvline(x=xlimit_min+2,linewidth=4,color='grey')

# Oct 11th
plt.axvline(x=xlimit_min+10,linewidth=4,color='grey')

# Oct 22nd
plt.axvline(x=xlimit_min+21,linewidth=4,color='grey')

# %% [markdown]
# # Donor Data Set
# Let's go ahead and switch gears and take a look at a data set consisting of information on donations to the federal campaign.
# 
# This is going to be the biggest data set we've looked at so far. You can download it here , then make sure to save it to the same folder your iPython Notebooks are in.
# 
# The questions we will be trying to answer while looking at this Data Set is:
# 
# 1.) How much was donated and what was the average donation?
# 
# 2.) How did the donations differ between candidates?
# 
# 3.) How did the donations differ between Democrats and Republicans?
# 
# 4.) What were the demographics of the donors?
# 
# 5.) Is there a pattern to donation amounts?
# %% [markdown]
# 

# %%
donor_df = pd.read_csv('Election_Donor_Data.csv')


# %%
donor_df.info()


# %%
donor_df.head()


# %%
donor_df['contb_receipt_amt'].value_counts()


# %%
donor_mean = donor_df['contb_receipt_amt'].mean()

donor_std = donor_df['contb_receipt_amt'].std()

print('The average donation was {0:.2f} with a std {1:.2f}'.format(donor_mean,donor_std))


# %%
top_donor = donor_df['contb_receipt_amt'].copy()

top_donor.sort_values()

top_donor


# %%
top_donor = top_donor[top_donor > 0] # filtering positive values

top_donor.sort_values()


# %%
top_donor.value_counts().head(10)


# %%
com_don = top_donor[top_donor < 2500]

com_don.hist(bins=100)


# %%
# Separate donations by party


# %%
# Create parties column
candidates = donor_df.cand_nm.unique()

candidates


# %%

# Dictionary of party affiliation
party_map = {'Bachmann, Michelle': 'Republican',
            'Cain, Herman': 'Republican',
            'Gingrich, Newt': 'Republican',
            'Huntsman, Jon': 'Republican',
            'Johnson, Gary Earl': 'Republican',
            'McCotter, Thaddeus G': 'Republican',
            'Obama, Barack': 'Democrat',
            'Paul, Ron': 'Republican',
            'Pawlenty, Timothy': 'Republican',
            'Perry, Rick': 'Republican',
            "Roemer, Charles E. 'Buddy' III": 'Republican',
            'Romney, Mitt': 'Republican',
            'Santorum, Rick': 'Republican'}

# Now map the party with candidate
donor_df['Party'] = donor_df.cand_nm.map(party_map)


# %%
"""
for i in range(0, len(donor_df)):
    if donor_df.cand_nm[i] == 'Obama, Barack':
        donor_df['Party'][i] = 'Democrat'
    else:
        donor_df['Party'][i] = 'Republican'
"""


# %%
donor_df = donor_df[donor_df.contb_receipt_amt > 0]


# %%
donor_df.head()


# %%
donor_df.groupby('cand_nm')['contb_receipt_amt'].count() # how many people donated to each campaign


# %%
donor_df.groupby('cand_nm')['contb_receipt_amt'].sum() # amount of dollars contributed for each campaign


# %%
cand_amount = donor_df.groupby('cand_nm')['contb_receipt_amt'].sum()

i = 0

for don in cand_amount:
    print('The candidate {0} raise {1:.0f} dollars'.format(cand_amount.index[i],don))
    print('\n')
    i += 1


# %%
cand_amount.plot(kind='bar')


# %%
donor_df.groupby('Party')['contb_receipt_amt'].sum().plot(kind='bar')


# %%
occupation_df = donor_df.pivot_table('contb_receipt_amt',
                                    index='contbr_occupation',
                                    columns='Party',
                                    aggfunc='sum')


# %%
occupation_df.head()


# %%
occupation_df.tail()


# %%
occupation_df.shape


# %%
occupation_df = occupation_df[occupation_df.sum(1) > 1000000]


# %%
occupation_df.shape


# %%
occupation_df.plot(kind='bar')  # not very clear


# %%
occupation_df.plot(kind='barh',figsize=(10,12),cmap='seismic') # more clear (bigger size and horizontal)


# %%
occupation_df.drop(['INFORMATION REQUESTED PER BEST EFFORTS','INFORMATION REQUESTED'],axis=0,inplace=True) # inplace=True is a permanent effect into the DataFrame


# %%
# merging the CEO columns into one
occupation_df.loc['CEO'] = occupation_df.loc['CEO'] + occupation_df.loc['C.E.O.']

occupation_df.drop('C.E.O.',inplace=True)


# %%
occupation_df.plot(kind='barh',figsize=(10,12),cmap='seismic')

# %% [markdown]
# 
# Great Job!
# There's still so much to discover in these rich datasets! Come up with your own political questions you want answered! Or just play around with different methods of visualizing the data!
# 
# For more on general data analysis of politics, I highly suggest the 538[https://fivethirtyeight.com/politics/] website!
# 
# Again, great job on getting through the course this far! Go ahead and search the web for more data to discover!

