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
dataset = randn(25)


# %%
sns.rugplot(dataset)

plt.ylim(0,1)


# %%
plt.hist(dataset, alpha=0.3) # include histogram into the dataset
sns.rugplot(dataset)


# %%
# Galcium basis function
sns.rugplot(dataset)

x_min = dataset.min() - 2 # -2 and +2 is to give some range outside of the dataset
x_max = dataset.max() + 2

x_axis = np.linspace(x_min, x_max, 100) # 100 space points from x_min to x_max

# practical estimation equation of the bandwidth ((4 x (standard deviation of the dataset)) by the power of 5 / (3 x (number of points in the dataset))) by the power of 0.2 (1 over 5)
bandwidth = ( (4*dataset.std()**5) / (3*len(dataset))) ** 0.2

kernel_list = []

for data_point in dataset:
    # Create a kernel for each point and append it to the kernel list
    kernel = stats.norm(data_point, bandwidth).pdf(x_axis)
    kernel_list.append(kernel)

    # Scale for plotting
    kernel = kernel / kernel.max()
    kernel = kernel * 0.4

    plt.plot(x_axis, kernel, color='grey', alpha=0.5)

plt.ylim(0,1)


# %%
# Kernel Density Estimation Plot creation
sum_of_kde = np.sum(kernel_list, axis=0)

fig = plt.plot(x_axis, sum_of_kde, color='indianred')

sns.rugplot(dataset)

plt.yticks([])

plt.suptitle("Sum of the basis functions")


# %%
# Kernel Density Estimation Plot by Seaborn
sns.kdeplot(dataset)


# %%
sns.rugplot(dataset, color='black')

for bw in np.arange(0.5,2,0.25):
    sns.kdeplot(dataset, bw_method=bw, lw=1.8, label=bw)


# %%
url = 'https://en.wikipedia.org/wiki/Kernel_(statistics)'
explanation = 'https://www.youtube.com/watch?v=DCgPRaIDYXA&ab_channel=KimberlyFessel'


# %%
kernel_options = ['biw','cos','epa','gau','tri','triw']

for kern in kernel_options:
    sns.kdeplot(dataset, kernel=kern, label=kern, shade=True)


# %%
sns.kdeplot(dataset, y=True)


# %%
url = 'https://en.wikipedia.org/wiki/Cumulative_distribution_function'


# %%
sns.kdeplot(dataset, cumulative=True)


# %%
mean = [0,0]

cov = [[1,0], [0,100]]

dataset2 = np.random.multivariate_normal(mean, cov, 1000)

dframe = pd.DataFrame(dataset2, columns=['X','Y'])

sns.kdeplot(data=dframe,x='X',y='Y')


# %%
sns.kdeplot(dframe.X, dframe.Y, shade=True) # passing two vectors separately


# %%
sns.kdeplot(data=dframe,x='X',y='Y', bw_adjust=1) # specify bandwith


# %%
sns.kdeplot(data=dframe, x='X', y='Y', bw_method='silverman') # use silverman estimation bandwidth (silverman rule of thumb)


# %%
sns.jointplot('X','Y',dframe,kind='kde') # specifying Kernel Density Estimation Plot


# %%



