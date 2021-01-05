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
        # actual training procedure
         # loop over epoches
        for epoch in np.arange(0, epochs):
        # loop over each individual data point
            for (x, target) in zip(X, y):
                '''take the "dot" product between the input features
                and the weight matrix, then pass this value
                through the "step function" to obtain the prediction'''
                p = self.step(np.dot(x, self.W))
                '''only perform a weight update if our prediction
                 does not match the target'''
                if p != target:
                    # determine the error
                    error = p - target
                    # update the weight matrix
                    self.W += -self.alpha * error * x
    # predict the class labels
    def predict(self, X, addBias=True):
        # ensure our input is a matrix
        X = np.atleast_2d(x)
        # check to see if the bias column should be added
        # bias trick
        if addBias:
            X = np.c_[X, np.ones((X.shape[0]))]
        return self.step(np.dot(X, self.W))


