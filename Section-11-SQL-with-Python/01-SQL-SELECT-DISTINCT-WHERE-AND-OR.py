# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import sqlite3
import pandas as pd
con = sqlite3.connect('sakila.db')

def sql_to_df(sql_query):

    df = pd.read_sql(sql_query, con)

    return df


# %%
# Select multiple columns
query = '''SELECT first_name,last_name
            FROM customer'''

sql_to_df(query).head()


# %%
# Select everything
query = '''SELECT * 
            FROM customer'''

sql_to_df(query).head()

# %% [markdown]
# ## Syntax for the SQL DISTINCT Statement
# 
# In a table, a column may contain duplicate values; and sometimes you only want to list the distinct (unique) values. The DISTINCT keyword can be used to return only distinct (unique) values.
# 
# SELECT DISTINCT column_name
# FROM table_name;

# %%
query = '''SELECT DISTINCT(country_id)
            FROM city'''

sql_to_df(query).head()


# %%
# SQL WHERE
query = '''SELECT * 
            FROM customer
            WHERE store_id = 1'''

sql_to_df(query).head()


# %%
query = ''' SELECT *
            FROM customer
            WHERE first_name = 'MARY' '''

sql_to_df(query)


# %%
query = ''' SELECT *
            FROM film
            WHERE release_year = 2006
            AND rating = 'R' '''

sql_to_df(query).head()


# %%
query = '''SELECT *
            FROM film
            WHERE rating = 'PG'
            OR rating = 'R' '''

sql_to_df(query)


