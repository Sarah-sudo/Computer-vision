# import the necessary packages
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pre_processing.simplepreprocessor import SimplePreprocessor
from datasets.simpledatasetloader import SimpleDatasetLoader
from imutils import paths
import argparse

# grab the list of images from disk:
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to input dataset")
args = vars(ap.parse_args())
# input images path
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
# resize images
# load them from disk into memory
# reshape and flatten them into a 3,072-dim array
sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0], 3072))
# encode labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)
''' partition the data into training and testing splits using 75% of
    the data for training and the remaining 25% for testing '''
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=5)

# Regularization
# apply a few different types of regularization
# loop over our set of regularizers
for r in(None, "l1", "l2"):
    ''' train a SGD classifier using a softmax loss function and the 
        specified regularization function for 10 epochs '''
    print("[INFO] training model with ‘{}‘ penalty".format(r))
    #initialize and train the SGDClassifier
    model = SGDClassifier(loss="log", penalty=r, max_iter=10, learning_rate="constant",
    eta0=0.01, random_state=42)
    # train
    model.fit(trainX, trainY)
    # evaluate model
    acc = model.score(testX, testY)
    print("[INFO] ‘{}‘ penalty accuracy: {:.2f}%".format(r, acc * 100))



