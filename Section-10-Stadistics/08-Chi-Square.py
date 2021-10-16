# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# [Chi Square Intro 1](https://www.youtube.com/watch?v=dXB3cUGnaxQ&ab_channel=KhanAcademy)
# 
# [Chi Square Intro 2](https://www.youtube.com/watch?v=hcDb12fsbBU&ab_channel=jbstatistics)
# 
# [Pearson's chi square test](https://www.youtube.com/watch?v=2QeDRsxSF9M&ab_channel=KhanAcademy)
# %% [markdown]
# # Chi-Square¶
# 
# In this Statistics Appendix Lecture, we'll go over the Chi-Square Distribution and the Chi-Square Test.
# 
# Note: Before viewing this lecture, see the Hypothesis Testing Notebook Lecture.
# 
# Let's start by introducing the general idea of observed and theoretical frequencies, then later we'll approach the idea of the Chi-Sqaure Distribution and its definition. After that we'll do a qcuik example with Scipy on using the Chi-Square Test.
# 
# Supppose that you tossed a coin 100 times. Theoretically you would expect 50 tails and 50 heads, however it is pretty unlikely you get that result exactly. Then a question arises... how far off from you expected/theoretical frequency would you have to be in order to conclude that the observed result is statistically significant and is not just due to random variations.
# 
# Since we wanted to know whether observed frequencies differ significantly from the expected frequencies we'll have to define a term for a measure of discrepency.
# 
# We'll define this measure as Chi-Square, which will be the sum of the squared difference between the observed and expected frequency divided by the expected frequency for all events. To show this more clearly, this is mathematically written as:$$ \chi ^2 =  \frac{(o_1 - e_1)^2}{e_1}+\frac{(o_2 - e_2)^2}{e_2}+...+\frac{(o_k - e_k)^2}{e_k} $$Which is the same as:$$\chi ^2 = \sum^{k}_{j=1} \frac{(o_j - e_j)^2}{e_j} $$
# 
# If the total frequency is N
# 
# $$ \sum o_j = \sum e_j = N $$
# Then we could rewrite the Chi-Square Formula to be:$$ \chi ^2 = \sum \frac{o_j ^2}{e_j ^2} - N$$
# 
# We can now see that if the Chi Square value is equal to zero, then the observed and theoretical frequencies agree exactly. While if the Chi square value is greater than zero, they do not agree.
# 
# The sampling distribution of Chi Square is approximated very closely by the Chi-Square distribution
# %% [markdown]
# The [Chi Square Test for Goodness](https://www.youtube.com/watch?v=2QeDRsxSF9M&ab_channel=KhanAcademy) of Fit
# We can now use the Chi-Square test can be used to determine how well a theoretical distribution fits an observed empirical distribution.
# 
# Scipy will basically be constructing and looking up this table for us:

# %%
url='http://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Chi-square_distributionCDF-English.png/300px-Chi-square_distributionCDF-English.png'

from IPython.display import Image
Image(url)

# %% [markdown]
# Let's go ahead and do an example problem. Say you are at a casino and are in charge of monitoring a craps(a dice game where two dice are rolled). You are suspcious that a player may have switched out the casino's dice for their own. How do we use the Chi-Square test to check whether or not this player is cheating?
# 
# You will need some observations in order to begin. You begin to keep track of this player's roll outcomes.You record the next 500 rolls taking note of the sum of the dice roll result and the number of times it occurs.
# 
# You record the following:
# 
# Sum of Dice Roll	2	3	4	5	6	7	8	9	10	11	12
# 
# Number of Times Observed	8	32	48	59	67	84	76	57	34	28	7
# 
# Now we also know the expected frequency of these sums for a fair dice. That frequency distribution looks like this:
# 
# Sum of Dice Roll	2	3	4	5	6	7	8	9	10	11	12
# 
# Expected Frequency	1/36	2/36	3/36	4/36	5/36	6/36	5/36	4/36	3/36	2/36	1/36
# 
# Now we can calculated the expected number of rolls by multiplying the expected frequency with the total sum of the rolls (500 rolls).

# %%
# Check sum of the rolls
observed = [8,32,48,59,67,84,76,57,34,28,7]
roll_sum = sum(observed)
roll_sum


# %%
# The expected frequency
freq = [1,2,3,4,5,6,5,4,3,2,1]

possible_rolls = 1.0/36

freq = [possible_rolls*dice for dice in freq]

freq


# %%
expected = [roll_sum*f for f in freq]
expected


# %%
# Perform Chi Square Test
from scipy import stats

chisp, p = stats.chisquare(observed, expected)

print('The chi-squared test statistic is %.2f' %chisp)
print('The p-value for the test is %.2f' %p)


