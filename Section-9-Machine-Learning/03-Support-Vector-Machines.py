# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# Notebook: [https://github.com/jmportilla/Udemy---Machine-Learning/blob/master/Support%20Vector%20Machines.ipynb](https://github.com/jmportilla/Udemy---Machine-Learning/blob/master/Support%20Vector%20Machines.ipynb)

# %%
from IPython.display import Image
url = 'http://docs.opencv.org/_images/separating-lines.png'
Image(url)


# %%
url= 'http://docs.opencv.org/_images/optimal-hyperplane.png'
Image(url)

# %% [markdown]
# So how do we actually mathematically compute that optimal hyperplane? I'll explain with a very brief overview below in Part 3, but I highly suggest you check out the full explanation on [Wikipedia](https://en.wikipedia.org/wiki/Support-vector_machine) or in the lecture videos following Part 3. Here video [tutorial](https://www.youtube.com/watch?v=efR1C6CvhmE)
# %% [markdown]
# ## Part 3: Computing the Hyperplane
# Let's go ahead and start by defining the Hyperplane in this case with the equation of a line, where Beta tranposed is the known weight vector of the features we've seen before and Beta nought is the bias.
# 
# $$ f(x) = \beta_{0} + \beta^{T} x $$
# There are an infinite number of ways we could scale the weight vector and the bias, but remember we want to maximize the margin between the two classes. So we realize through some math (explained in detail the videos below) can set this as:
# 
# $$ |\beta_{0} + \beta^{T} x| = 1 $$
# where x symbolizes the training examples closest to the hyperplane. In general, the training examples that are closest to the hyperplane are called support vectors. These support vectors are filled in with color in the image above. This representation is known as the canonical hyperplane.
# 
# From geometry we know that the distance betweeen a point x and the hyperplane (Beta,Beta0)is:
# 
# $$\mathrm{distance} = \frac{|\beta_{0} + \beta^{T} x|}{||\beta||}.$$
# In particular, for the canonical hyperplane, the numerator is equal to one and the distance to the support vectors is
# 
# $$\mathrm{distance}_{\text{ support vectors}} = \frac{|\beta_{0} + \beta^{T} x|}{||\beta||} = \frac{1}{||\beta||}$$
# Recall that the margin introduced in the previous section, here denoted as M, is twice the distance to the closest examples:
# 
# $$M = \frac{2}{||\beta||}$$
# Finally, the problem of maximizing M is equivalent to the problem of minimizing a function L(Beta) subject to some constraints. The constraints model the requirement for the hyperplane to classify correctly all the training examples xi.
# 
# Formally,
# 
# $$\min_{\beta, \beta_{0}} L(\beta) = \frac{1}{2}||\beta||^{2} \text{ subject to } y_{i}(\beta^{T} x_{i} + \beta_{0}) \geq 1 \text{ } \forall i$$
# where yi represents each of the labels of the training examples.
# 
# This is a problem of [Lagrangian optimization](https://en.wikipedia.org/wiki/Lagrange_multiplier) that can be solved using Lagrange multipliers to obtain the weight vector Beta and the bias Beta0 of the optimal hyperplane.
# 
# If we want to do non-linear classification we can employ the [kernel](https://en.wikipedia.org/wiki/Kernel_method) trick. Using the kernel trick we can "slice" the feature space with a Hyperplane. For a quick illustration of what this looks like, check out both the image and the video below!

# %%
# Kernel Trick for the Feature Space (tutorial: https://www.youtube.com/watch?v=Toet3EiSFcM&t=0s)
url='http://i.imgur.com/WuxyO.png'
Image(url)


