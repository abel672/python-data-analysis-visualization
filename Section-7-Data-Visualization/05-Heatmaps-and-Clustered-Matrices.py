# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
# Standard
import numpy as np
import pandas as pd
from numpy.random import randn

# Stats
from scipy import stats

# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# %%
flight_dframe = sns.load_dataset('flights')


# %%
flight_dframe.head()


# %%
flight_dframe = flight_dframe.pivot('month','year','passengers') # pivot (row, columns, values)


# %%
flight_dframe.head()


# %%
sns.heatmap(flight_dframe)


# %%
sns.heatmap(flight_dframe,annot=True,fmt='d')


# %%
sns.heatmap(flight_dframe,center=flight_dframe.loc['Jan',1955]) # Jan and 1995 are the central (blank) values of the heatmap now


# %%
f,(axis1, axis2) = plt.subplots(2,1)

yearly_flights = flight_dframe.sum() # all flights of every year

years = pd.Series(yearly_flights.index.values)
years = pd.DataFrame(years)

flights = pd.Series(yearly_flights.values)
flight = pd.DataFrame(flights)

year_dframe = pd.concat((years,flights),axis=1)
year_dframe.columns = ['Year','Flights']

sns.barplot(x='Year', y='Flights',data=year_dframe,ax=axis1)

sns.heatmap(flight_dframe,cmap='Blues',ax=axis2,cbar_kws={'orientation':'horizontal'})


# %%
sns.clustermap(flight_dframe) # cluster map


# %%
sns.clustermap(flight_dframe, col_cluster=False)


# %%
sns.clustermap(flight_dframe,standard_scale=1) # scale the columns


# %%
sns.clustermap(flight_dframe, standard_scale=0) # scale the rows


# %%
sns.clustermap(flight_dframe,z_score=1) # score of activity


