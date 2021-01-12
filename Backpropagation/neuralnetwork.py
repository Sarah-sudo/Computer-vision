# import the necessary packages
import numpy as np

class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        ''' initialize the list of weights matrices,
        then store the network architecture and learning rate '''
        self.W = []
        self.layers = layers
        self.alpha = alpha

        '''start looping from the index of the first layer but
        stop before we reach the last two layers'''
        for i in np.arange(0, len(layers) - 2):
             ''' randomly initialize a weight matrix connecting the
            number of nodes in each respective layer together,
            adding an extra node for the bias '''
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
            self.W.append(w / np.sqrt(layers[i]))

        '''the last two layers are a special case where the input
        connections need a bias term but the output does not'''
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))
