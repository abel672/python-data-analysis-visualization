# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# [Factorials](https://www.youtube.com/watch?v=pxh__ugRKz8)
# 
# [Binomial Distribution](https://www.youtube.com/watch?v=WWv0RUxDfbs)
#
# [Probability Mass Function](https://www.youtube.com/watch?v=auAvt7CIezM)
# %% [markdown]
# 
# The formula for a Binomial Distribution Probability Mass Function turns out to be:
# 
# $$Pr(X=k)=C(n,k)p^k (1-p)^{n-k}$$
# 
# Where n= number of trials,k=number of successes,p=probability of success,1-p=probability of failure (often written as q=1-p).
# 
# This means that to get exactly 'k' successes in 'n' trials, we want exactly 'k' successes:$$p^k$$and we want 'n-k' failures:$$(1-p)^{n-k}$$Then finally, there are$$C(n,k)$$ways of putting 'k' successes in 'n' trials. So we multiply all these together to get the probability of exactly that many success and failures in those n trials!
# 
# Quick note, C(n,k) refers to the number of possible combinations of N things taken k at a time.
# 
# This is also equal to:$$C(n,k) =  \frac{n!}{k!(n-k)!}$$

# %%
# Set up player A

# Probability of success for A
p_A = .72
# Number of shots for A
n_A = 11

# Make 6 shots
k = 6

# Now import scipy for combination
import scipy.special as sc

# Set up C(n,k)
comb_A = sc.comb(n_A, k)

# Now put it together to get the probability
answer_A = comb_A * (p_A**k) * ((1 - p_A) ** (n_A - k))

# Put the answer in percentage form!
answer_A = 100*answer_A

# Quickly repeat all steps for Player B
p_B = .48
n_B = 15
comb_B = sc.comb(n_B, k)
answer_B = 100 * comb_B * (p_B**k) * ((1 - p_B) ** (n_B - k))

#Print Answers
print(' The probability of player A making 6 shots in an average game is %1.1f%% ' %answer_A)
print(' The probability of player B making 6 shots in an average game is %1.1f%% ' %answer_B)

# %% [markdown]
# So now we know that even though player B is technically a worse shooter, because she takes more shots she will have a higher chance of making 6 shots in an average game!
# 
# But wait a minute... what about a higher amount of shots, will player's A higher probability take a stronger effect then? What's the probability of making 9 shots a game for either player?

# %%
#Let's find out!

#Set number of shots
k = 9

#Set new combinations
comb_A = sc.comb(n_A,k)
comb_B = sc.comb(n_B,k)

# Everything else remains the same
answer_A = 100 * comb_A * (p_A**k) * ((1-p_A)**(n_A-k))
answer_B = 100 * comb_B * (p_B**k) * ((1-p_B)**(n_B-k))

#Print Answers
print(' The probability of player A making 6 shots in an average game is %1.1f%% ' %answer_A)
print(' The probability of player B making 6 shots in an average game is %1.1f%% ' %answer_B)

# %% [markdown]
# 
# ## Now let's investigate the mean and standard deviation for the binomial distribution
# 
# The mean of a binomial distribution is simply:$$\mu=n*p$$
# 
# This intuitively makes sense, the average number of successes should be the total trials multiplied by your average success rate.
# 
# Similarly we can see that the standard deviation of a binomial is:$$\sigma=\sqrt{n*q*p}$$
# 
# So now we can ask, whats the average number of shots each player will make in a game +/- a standard distribution?

# %%
# Let's go ahead and plug in to the formulas

# Get the mean
mu_A = n_A * p_A
mu_B = n_B * p_B

# Get the standard deviation
sigma_A = (n_A * p_A * (1-p_A) )**0.5
sigma_B = (n_B * p_B * (1-p_B) )**0.5

# Now prints results
print('Player A will make an average of %1.0f +/- %1.0f shots per game' %(mu_A,sigma_A))
print('Player B will make an average of %1.0f +/- %1.0f shots per game' %(mu_B,sigma_B))
print("NOTE: It's impossible to make a decimal of a shot so '%1.0f' was used to replace the float!")

# %% [markdown]
# ### Let's see how to automatically make a binomial distribution.
# 

# %%
from scipy.stats import binom

# We can get stats: Mean('m'), variance('v'), skew('s'), and/or kurtosis('k')
mean, var = binom.stats(n_A, p_A)

print(mean)
print(var**0.5)

# %% [markdown]
# ### We can also get the probability mass function:
# 
# Let's try another example to see the full PMF (Probability Mass Function) and plotting it.
# 
# Imagine you flip a fair coin. Your probability of getting a heads is p=0.5 (success in this example).
# 
# So what does your probability mass function look like for 10 coin flips?

# %%
import numpy as np

n = 10
p = 0.5

x = range(n+1)

Y = binom.pmf(x,n,p)

Y

# %% [markdown]
# ### Finally, let's plot the binomial distribution to get the full picture.

# %%
import matplotlib.pyplot as plt 

# For simple plots, matplotlib is fine, seaborn is unnecessary.

plt.plot(x,Y,'o')

#Title (use y=1.08 to raise the long title a little more above the plot)
plt.title('Binomial Distribution PMF: 10 coin Flips, Odds of Success for Heads is p=0.5',y=1.08)

#Axis Titles
plt.xlabel('Number of Heads')
plt.ylabel('Probability')


# %%



