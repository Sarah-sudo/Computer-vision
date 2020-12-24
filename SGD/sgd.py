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
    for i in np.arange(0, X.shape[0], batchSize):
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
for epoch in np.arange(0, args["epochs"]):    
    epochloss=[]
    # loop over batches
    for (batchX, batchY) in next_batch(X, y, args["batch_size"]):
        #For each batch,compute the dot product between the batch and W
        preds = sigmond_activation(batchX.dot(W))
        error = preds - batchY
        epochloss.append(np.sum(error ** 2))
        '''the gradient descent update is the dot product between our
           current batch and the error on the batch'''
        gradient = batchX.T.dot(error)
        # update W
        W += -args['alpha'] * gradient
    # update loss history by taking the average loss across all batches
    loss = np.average(epochloss)
    losses.append(loss)
    # check to see if an update should be displayed
    if epoch == 0 or epoch % 5 == 0:
        print("[INFO] epoch={}, loss={:.7f}".format(int(epoch + 1), loss))

# evaluate our model
print("[INFO] evaluating...")
preds = predict(testX, W)
print(classification_report(testY, preds))

# plot data
plt.style.use('ggplot')
plt.figure()
plt.title("Data")
# c->marker colors   s->marker size in points**2
plt.scatter(testX[:, 0], testX[:, 1], marker="o", c=testY, s=30)
# plot loss
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, args["epochs"]), losses)
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()

