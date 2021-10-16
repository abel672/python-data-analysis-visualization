# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# [T-Distribution](https://www.youtube.com/watch?v=32CuxWdOlow&ab_channel=365DataScience)
# 
# [Another](https://www.youtube.com/watch?v=5ABpqVSx33I&ab_channel=KhanAcademy)
# 
# 

# %%
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from scipy.stats import t

import numpy as np

x = np.linspace(-5,5,100)

rv = t(3)

plt.plot(x, rv.pdf(x))

# %% [markdown]
# Additional resources can be found here:
# 
# 1.) [http://en.wikipedia.org/wiki/Student%27s_t-distribution](http://en.wikipedia.org/wiki/Student%27s_t-distribution)
# 
# 2.) [http://mathworld.wolfram.com/Studentst-Distribution.html](http://mathworld.wolfram.com/Studentst-Distribution.html)
# 
# 3.) [http://stattrek.com/probability-distributions/t-distribution.aspx](http://stattrek.com/probability-distributions/t-distribution.aspx)

