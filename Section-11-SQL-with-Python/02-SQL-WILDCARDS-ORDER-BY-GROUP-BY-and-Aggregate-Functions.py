# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import sqlite3
import pandas as pd
con = sqlite3.connect('sakila.db')

def sql_to_df(sql_query):

    df = pd.read_sql(sql_query, con)

    return df

# %% [markdown]
# 
# Before we begin with Wildcards, ORDER BY, and GROUP BY. Let's take a look at aggregate functions.
# 
# AVG() - Returns the average value.
# 
# COUNT() - Returns the number of rows.
# 
# FIRST() - Returns the first value.
# 
# LAST() - Returns the last value.
# 
# MAX() - Returns the largest value.
# 
# MIN() - Returns the smallest value.
# 
# SUM() - Returns the sum.
# 
# You can call any of these aggregate functions on a column to get the resulting values back. For example

# %%
query = ''' SELECT COUNT(customer_id)
            FROM customer;'''

sql_to_df(query).head()


# %%
query = ''' SELECT *
            FROM customer
            WHERE first_name LIKE 'M%'; '''

sql_to_df(query).head()


# %%
query = ''' SELECT *
            FROM customer
            WHERE last_name LIKE '_ING' '''

sql_to_df(query).head()

# %% [markdown]
# ## IMPORTANT NOTE!
# 
# Using [charlist] with SQLite is a little different than with other SQL formats, such as MySQL.
# 
# In MySQL you would use:
# 
# WHERE value LIKE '[charlist]%'
# 
# In SQLite you use:
# 
# WHERE value GLOB '[charlist]*'

# %%
query = ''' SELECT *
            FROM customer
            WHERE first_name GLOB '[AB]*'; '''

sql_to_df(query).head()


# %%
query = '''SELECT *
            FROM customer
            ORDER BY last_name;'''

sql_to_df(query).head()


# %%
query = '''SELECT *
            FROM customer
            ORDER BY last_name DESC; '''

sql_to_df(query).head()


# %%
query = '''SELECT store_id, COUNT(customer_id)
            FROM customer
            GROUP BY store_id; '''

sql_to_df(query).head()


