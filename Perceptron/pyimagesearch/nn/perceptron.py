#import all necessary packages
import numpy as np

class Perceptron:
    # N: number of columns in our input feature vectors. here->(N=2)
    def __init__(self, N, alpha=0.1):
        # initialize the weight matrix and store the learning rate
        # (N+1) enteries -> N inputs and one for the bias
        self.W = np.random.randn(N + 1) / np.sqrt(N) 
        self.alpha = alpha
    # define the step function:
    def step(self, x):
        return 1 if x > 0 else 0
    # train 
    def fit(X, y, epochs=10):
        # applying the bias trick bias trick
        X = np.c_[X, np.ones((X.shape[0]))]