# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np

import pandas as pd

from pandas import Series, DataFrame


# %%
import webbrowser
website = 'https://en.wikipedia.org/wiki/NFL_win–loss_records'
webbrowser.open(website)


# %%
nfl_frame = pd.read_clipboard()


# %%
nfl_frame


# %%
nfl_frame.columns


# %%
nfl_frame.Rank


# %%
nfl_frame.Team


# %%
nfl_frame['First NFL season']


# %%
# creating data frame from other data frame columns
DataFrame(nfl_frame, columns=['Team', 'First NFL season', 'Total Games'])


# %%
# get first number of rows
nfl_frame.head(3)


# %%
# get last number of rows
nfl_frame.tail(3)


# %%
# grabing a row by index number
nfl_frame.iloc[3]


# %%
nfl_frame['Stadium'] = "Levi's Stadium"


# %%
nfl_frame


# %%
nfl_frame['Stadium'] = np.arange(5)


# %%
nfl_frame


# %%
stadiums = Series(["Levi's Stadium", "AT&T Stadium"], index=[5,0])


# %%
stadiums


# %%
nfl_frame['Stadium'] = stadiums


# %%
nfl_frame


# %%
# delete column
del nfl_frame['Stadium']


# %%
nfl_frame


# %%
data = {'City': ['SF', 'LA', 'NYC'], 'Population': [837000, 3880000, 8400000]}


# %%
city_frame = DataFrame(data)


# %%
city_frame


# %%



