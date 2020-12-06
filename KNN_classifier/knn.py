# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier          # implementation of the k-NN algorithm,
from sklearn.preprocessing import LabelEncoder              # convert labels represented as strings to integers
from sklearn.model_selection import train_test_split        # create our training and testing splits
from sklearn.metrics import classification_report           # evaluate the performance of our classifier

from pre_processing.simplepreprocessor import SimplePreprocessor # call simplepreprocessor by python library (preprocessor)
from datasets.simpledatasetloader import SimpleDatasetLoader     # call simpledatasetloader by python library (datasetloader)

from imutils import paths                                    #path
import argparse





# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()  #Object for parsing command line strings into Python objects.
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")    #The path to where our input image dataset resides on disk.

ap.add_argument("-k", "--neighbors", type=int, default=1,
help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1,
help="# of jobs for k-NN distance (-1 uses all available cores)")

args = vars(ap.parse_args())






# grab the list of images that we’ll be describing
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))   # a list of the path for each labeled image

sp = SimplePreprocessor(32, 32)                         #resize each image to 32*32 pixels
sdl = SimpleDatasetLoader(preprocessors=[sp])           #implying that sp will be applied to every image in the dataset
(data, labels) = sdl.load(imagePaths, verbose=500)      #load resized data with labels   3000images=> output shape: (3000, 32, 32,3)   

data = data.reshape((data.shape[0], 3072))              #flatten our images from a 3D representation to a single list of pixel intensities=> output shape:(3000, 3072)
print("[INFO] feature matrix: {:.1f}MB".format(data.nbytes / (1024 * 1000.0)))  #how much memory it takes to store these 3,000 images in memory






#building our training and testing splits
le = LabelEncoder()
labels = le.fit_transform(labels)                      # encode the labels(string) as integers 
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)  # X: examples, Y:labels








#create our k-NN classifier and evaluate it
print("[INFO] evaluating k-NN classifier...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"], n_jobs=args["jobs"])  #initialize the KNeighborsClassifier class
model.fit(trainX,trainY)                    #train the classifier without learning
print(classification_report(testY, model.predict(testX), target_names=le.classes_))   #evaluate our classifier by using the classification_report function.










