# import the necessary packages
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# blob : Binary Large Object
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse

# activation function
''' sigmoid is a non-linear activation function 
that we can use to threshold our predictions '''
def sigmond_activation(x):
    # compute the sigmoid activation value for a given input
    return 1.0 / (1 + np.exp(-x))

# predict method
def predict(X, W):
    # take the dot product between our features and weight matrix
    # f (xi,W) = W.xi
    preds = sigmond_activation(X.dot(W))
    # apply a step function to threshold the outputs to binary
    preds[preds <= 0.5] = 0
    preds[preds > 0.5] = 1
    return preds

# parse our command line arguments
# construct the argument parse and parse the arguments 
#epoches & learning rate
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=100, help="# of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.01, help="Learning rate")
args = vars(ap.parse_args())

# generate some data to classify
''' generate a 2-class classification problem with 1,000 data points,
where each data point is a 2D feature vector'''
# images = 1000, 2D, 2 output class(labels)
(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
y = y.reshape(y.shape[0], 1)

# Bias trick
''' insert a column of 1’s as the last entry in the feature
matrix -- this little trick allows us to treat the bias
as a trainable parameter within the weight matrix '''
X = np.c_[X, np.ones((X.shape[0]))]

# Data split
''' partition the data into training and testing splits using 50% of
the data for training and the remaining 50% for testing '''
(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.5, random_state=42)

# initialize our weight matrix and list of losses
print("[INFO] training...")
W = np.random.randn(X.shape[1], 1)
losses = []

''' All of our variables are now initialized '''

# Loop over epochs
# use 'np.arange' not 'range' because we are using arguments 
for epoch in np.arange(0, args['epochs']):
    '''take the dot product between our features ‘X‘ and the weight
       matrix ‘W‘, then pass this value through our sigmoid activation
       function, thereby giving us our predictions on the dataset'''
    preds = sigmond_activation(trainX.dot(W))
    # determine the ‘error‘
    error = preds - trainY
    loss = np.sum(error ** 2)
    losses.append(loss)
    # compute gradient
    gradient = trainX.T.dot(error)
    W +=  -args['alpha'] * gradient

    # check to see if an update should be displayed
    if epoch == 0 or (epoch + 1) % 5 == 0 :
        print("[INFO] epoch={}, loss={:.7f}".format(int(epoch + 1), loss))

# evaluate our model
print("[INFO] evaluating...")
preds = predict(testX, W)
print(classification_report(testY, preds))

# plot the (testing) classification data
plt.style.use("ggplot")
plt.figure()
plt.title("Data")
plt.scatter(testX[:, 0], testX[:, 1], marker="o", c=testY, s=30)
# construct a figure that plots the loss over time
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, args["epochs"]), losses)
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()
