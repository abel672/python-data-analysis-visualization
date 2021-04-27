# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
from pandas import Series, DataFrame


# %%
from pandas import read_html


# %%
url = 'http://www.fdic.gov/bank/individual/failed/banklist.html'


# %%
# pip install beautifulsoup4
# pip install html5lib


# %%
dframe_list = pd.io.html.read_html(url)


# %%



