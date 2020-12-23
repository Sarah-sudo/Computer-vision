# import necessary packages
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse

def sigmond_activation(x):
    return 1.0 / (1 + np.exp(-x))

def predict(X, W):
    preds = sigmond_activation(X.dot(W))
    preds[preds <= 0.5] = 0
    preds[preds > 0.5] = 1
    return preds
    
def next_batch(X, y,batchSize):
    for i in np.arrange(0, X.shape[0], batchSize):
        yield (X[i:i + batchSize], y[i:i + batchSize])

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-e', "--epochs", type=float, default=100, help='# of epochs')
ap.add_argument('-a', "--alpha", type=float, default=0.01, help='learning rate')
ap.add_argument('-b', "--batch-size", type=int, default=32, help='size of SGD mini-batches')
args = vars(ap.parse_args())

# generate a 2-class classification problem with 1,000 data points,
# where each data point is a 2D feature vector
(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
y = y.reshape(y.shape[0], 1)
# bias trick
X = np.c_[X, np.ones((X.shape[0]))]
# split 
(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.5, random_state=42)
# initialize W and list of losses
print("[INFO] training...")
W = np.random.randn(X.shape[1], 1)
losses = [] 

# loop over epochs
