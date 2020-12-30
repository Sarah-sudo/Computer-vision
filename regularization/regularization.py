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
imagePaths = list(paths.list_images(args("dataset")))
# resize images
# load them from disk into memory
# reshape and flatten them into a 3,072-dim array
sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])