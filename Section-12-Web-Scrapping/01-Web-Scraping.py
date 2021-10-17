# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from bs4 import BeautifulSoup
import requests


# %%
import pandas as pd
from pandas import Series, DataFrame


# %%
url = 'http://www.ucop.edu/operating-budget/budgets-and-reports/legislative-reports/2013-14-legislative-session.html'


# %%
# get content from the page
result = requests.get(url)
c = result.content

# Find the tables in the HTML
soup = BeautifulSoup(c)


# %%
# Go to the section of interest
summary = soup.find('div', {'class': 'list-land','id': 'content'})

# getting tables
tables = summary.find_all('table')


# %%
data = []

rows = tables[0].findAll('tr')

# now grab every HTML cell in every row
for tr in rows:
    cols = tr.findAll('td')

    # Check to see if text is in the row
    for td in cols:
        text = td.find(text=True)
        print(text)
        data.append(text)


# %%
data

# %% [markdown]
# Now we'll use a for loop to go through the list and grab only the cells with a pdf file in them, we'll also need to keep track of the index to set up the date of the report.

# %%
reports = []
date = []

index = 0

# Go to find the pdf cells
for item in data:
    if 'pdf' in item:
        
        date.append(data[index-1])

        reports.append(item.replace(u'\xa0', u' '))
    
    index += 1

# %% [markdown]
# You'll notice a line to take care of '\xa0 ' This is due to a unicode error that occurs if you don't do this. Web pages can be messy and inconsistent and it is very likely you'll have to do some research to take care of problems like these.
# 
# Here's the link I used to solve this particular issue: [StackOverflow Page](https://stackoverflow.com/questions/10993612/how-to-remove-xa0-from-string-in-python)
# 
# Now all that is left is to organize our data into a pandas DataFrame!

# %%
# Setup Dates and Reports as Series
date = Series(date)
reports = Series(reports)


# %%
legislative_df = pd.concat([date, reports], axis=1)


# %%
legislative_df.columns = ['Date', 'Reports']


# %%
legislative_df


# %%



