# impoort the necessary packages
# for numerical processing
import numpy as np
# for loading images from the disk 
import cv2

''' initialize the class labels and set the seed of the
pseudorandom number generator so we can reproduce our results '''
labels = ["dog", "cat", "panda"]
np.random.seed(1)

# initialize our weight matrix and bias vector
''' randomly initialize our weight matrix and bias vector -- in a
*real* training and classification task, **these parameters would
be *learned* by our model**, but for the sake of this example, let’s 
use random values. '''
# W[k*d]
W = np.random.randn(3,3072)
# b[k*1]
b = np.random.randn(3)

# load our example image from disk
orig = cv2.imread("F:\Git_test\Computer-vision\Data\images\dog.jpg")
#resize
image = cv2.resize(orig, (32, 32)).flatten()

# applying our scoring function to compute the output:
scores = W.dot(image) + b

''' scoring function values for each of the class
    display the result to our screen
'''
# loop over the scores + labels and display them
for (label, score) in zip(labels, scores):
    print("[INFO] {}: {:.2f}".format(label, score))
# draw the label with the highest score on the image as our prediction
cv2.putText(orig, "Label: {}".format(labels[np.argmax(scores)]), 
            (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0),
             2)
# display our input image
cv2.imshow("image", orig)
cv2.waitKey(10000)