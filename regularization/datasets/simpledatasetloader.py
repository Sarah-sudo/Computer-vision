import numpy as np  #numerical processing
import cv2  #OpenCV bindings
import os  #extract the names of subdirectories in image paths
#our required packages

class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        # store the image preprocessor
        self.preprocessors = preprocessors

        if self.preprocessors is None:
            self.preprocessors = []
        # if the preprocessors are None, initialize them as an empty list

    def load(self, imagePaths, verbose=-1):
        # initialize the list of features and labels
        data = []
        labels = []

        # loop over the input images
        for (i, imagePath) in enumerate(imagePaths):
        # load the image and extract the class label assuming that our path has the following format:
        # /path/to/dataset/{class}/{image}.jpg
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]


            # check to see if our preprocessors are not None
            if self.preprocessors is not None:   
          # loop over the preprocessors and apply each to # the image    
                for p in self.preprocessors:  #loop over each of the preprocessors
                    image = p.preprocess(image)  #apply preprocessors to the image

            # treat our processed image as a "feature vector"  by updating the data list followed by the labels
             #update data and label list
            data.append(image)
            labels.append(label)


             # show an update every ‘verbose‘ images
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1,
                len(imagePaths)))
               # return a tuple of the data and labels
        return (np.array(data), np.array(labels))
