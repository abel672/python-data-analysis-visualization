# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
from pandas import Series, DataFrame
import pandas as pd


# %%
json_obj = """
{   "zoo_animal": "Lion",
    "food": ["Meat", "Veggies", "Honey"],
    "fur": "Golden",
    "clothes": null, 
    "diet": [{"zoo_animal": "Gazelle", "food":"grass", "fur": "Brown"}]
}
"""


# %%
import json


# %%
data = json.loads(json_obj)


# %%
data


# %%
json.dumps(data)


# %%
dframe = DataFrame(data['diet'])


# %%
dframe


# %%



