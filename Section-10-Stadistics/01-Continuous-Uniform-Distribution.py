# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# [Notebook](https://github.com/jmportilla/Statistics-Notes/blob/master/Continuous%20Uniform%20Distributions.ipynb)
# 
# A continuous random variable X with a probability density function is a continuous uniform random variable when:
# 
# $$f(x)=\frac{1}{(b-a)}\\\\a&lt;=x&lt;=b$$
# 
# This makes sense, since for a discrete uniform distribution the f(x)=1/n but in the continuous case we don't have a specific n possibilities, we have a range from the min (a) to the max (b)!
# 
# The mean is simply the average of the min and max:
# 
# $$\frac{(a+b)}{2}$$
# The variance is defined as:
# 
# $$ \sigma^2 = \frac{(b-a)^2}{12}$$
# So what would an example problem look like? Let's say on average, a taxi ride in NYC takes 22 minutes. After taking some time measurements from experiments we gather that all the taxi rides are uniformly distributed between 19 and 27 minutes. What is the probability density function of a taxi ride, or f(x)?

# %%
# Let's solve this with Python

# Lower bound time
a = 19

# Uppper bound time
b = 27

# Then using our probability density function we get
fx = 1.0/(b-a)

# show
print('The probability density function results in %1.3f' %fx)


# %%
# We can also get the variance
var = ((b-a)**2 ) / 12

# Show
print('The variance of the continuous unifrom distribution is 1%.1f' %var)


# %%
# This is the same as the Probability Density Function of f(27) (the entire space) minus the probability space Less than 25 minutes. 
fx_1 = 27.0 / (b-a)

fx_2 = 25.0 / (b-a)

# Our answer is then
answer = fx_1 - fx_2

print('The probability that the taxi ride will last at least 25 minutes is %2.1f' %(100*answer))


# %%
# Let's do this with scipy
from scipy.stats import uniform
import numpy as np
import matplotlib.pyplot as plt

# Let's set A and B
A=0
B=5

# Set x as 100 Linearly spaced points between A and B
x = np.linspace(A,B,100)

# Start point and end point
rv = uniform(loc=A, scale=B)

# Plot the PDF of that uniform distribution
plt.plot(x, rv.pdf(x))

# %% [markdown]
# Note the above line is at 0.2, as we would expect since 1/(5-0) is 1/5 or 0.2.
# %% [markdown]
# ## That's it for Uniform Continuous Distributions. Here are some more resource for you:
# 
# [Continuous Unique Distribution](https://www.youtube.com/watch?v=-qt8CPIadWQ)
# 
# [Scipy continous unique distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.uniform.html)
# 
# [Uniform Distribution](https://mathworld.wolfram.com/UniformDistribution.html)

