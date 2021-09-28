All resources to check: https://github.com/jmportilla/Udemy---Machine-Learning/blob/master/Introduction%20to%20Machine%20Learning.ipynb

  Book to read: http://robotics.stanford.edu/people/nilsson/MLBOOK.pdf
  
  Intro to read: https://www.toptal.com/machine-learning/machine-learning-theory-an-introductory-primer

  Formal Definition:

    A machine learning program is said to learn from experience E with respect to some class of tasks T and performance measure P,
    if its performance at tasks in T, as measured by P, improves with experience E.

  So what does that actually mean?

    We start with data, which is called experience E.

    We decide to perform some sort of task or analysis, which we call T.

    Then we use some validation measure to test our accuracy, we call this performance measure P.

    Let's go ahead and learn about the different types of Machine Learning problems!




  There are two kinds of learning

  1) Supervised Learning: We have a dataset consisting of both features and labels. The task is to construct an estimator which is able to predict the label of an object, given the set of features.

    Is divided in two categories:
      
      -Classification: is discrete, meaning an example belongs precisely to one class, and the set of classes cover the whole possible output space.
        
        Example: Classifying a tumor as either malignant of benign base on input data.

      -Regression: Given some data, the machine assumes that those values come from some sort of function, and attempts to find out what the function is. It tries to fit a mathematical function that describes a curve, such that the curve passes as close as possible to all the data points.

        Example: Predicting house prices based on input data.


  2) Unsupervised Learning: Here the data has not labels, we are interested in finding similarities between the objects in question. You can think on unsupervised learning as a means of discovering labels from the data itself.

    Example: Given a mixture of sounds sources, try to separate the sources of each particular sound.