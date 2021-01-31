# necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt 
import numpy as np 
import argparse

# load data
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
# scale data
trainX =trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0
# rehsape 
trainX = trainX.rehsape((trainX.shape[0], 3072))
testX = testX.rehsape((testX.shape[0], 3072))

# convert labels
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)
# initialize the label names
labelNames = ["airplane", "automobile", "bird", "cat", "deer", 
    "dog", "frog", "horse", "ship", "truck"]
