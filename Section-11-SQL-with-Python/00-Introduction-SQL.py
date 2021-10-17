# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# [Notebook](https://github.com/jmportilla/SQL-Appendix/blob/master/Introduction%20to%20SQL%20with%20Python.ipynb)

# %%
# Imports
import sqlite3
import pandas as pd


# %%
# create connection with DB
con = sqlite3.connect('sakila.db')


# %%
# Create query
sql_query = '''SELECT * FROM customer'''

# Perform query with pandas and add results into a DataFrame
df = pd.read_sql(sql_query, con)

# Results
df.head()


# %%



