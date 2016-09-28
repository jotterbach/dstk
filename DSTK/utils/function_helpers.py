import numpy as np


def _sigmoid(x):
    """
    Numerically-stable sigmoid function.
    Taken from:
      - http://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
      - http://stackoverflow.com/questions/3985619/how-to-calculate-a-logistic-sigmoid-function-in-python
    """
    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        # if x is less than zero then z will be small, denom can't be
        # zero because it's 1+z.
        z = np.exp(x)
        return z / (1 + z)

sigmoid = np.frompyfunc(_sigmoid, 1, 1)