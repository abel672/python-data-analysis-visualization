# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd


# %%
# pip install xlrd
# pip install openpyxl


# %%
xlsfile = pd.ExcelFile('Lec_28_test.xlsx')


# %%
dframe = xlsfile.parse('Sheet1')


# %%
dframe


# %%



