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

    def __repr__(self):
        # construct and return a string that represents the network architecture
        return "NeuralNetwork: {}".format("-".join(str(l) for l in self.layers))

    # forward pass
    def sigmoid(self, x):
        # compute and return the sigmoid activation value for a given input value
        return 1.0 / (1+np.exp(-x))

    #Backpropagation
    def sigmoid_deriv(self, x):
        '''compute the derivative of the sigmoid function ASSUMING
        that ‘x‘ has already been passed through the ‘sigmoid‘ function'''
        return x * (1 - x)

    # training our NeuralNetwork
    def fit(self, X, y, epochs=1000, displayUpdate=100):
        # bias term
        X = np.c_[X, np.ones((X.shape[0]))]
        # loop over epochs
        for epoch in np.arange(0, epochs):
            # loop over data points
            for (x, target) in zip(X, y):
                self.fit_partial(x, target)

            # check to see if we should display a training update
            if epoch == 0 or (epoch + 1) % displayUpdate == 0:
                loss = self.calculate_loss(X, y)
                print("[INFO] epoch={}, loss={:.7f}".format(epoch + 1, loss))
    
    # backpropagation
    # heart of the backpropagation algorithm -> fit_partial
    def fit_partial(self, x, y):
        #storing activation function for each layer
        A = [np.atleast_2d(x)]

        # FEEDFORWARD:
        # loop over the layers
        for layer in np.arange(0, len(self.W)):
            # net input
            net = A[layer].dot(self.W[layer])
            # net output
            out = self.sigmoid(net)
            # add net output to the list of activation function
            A.append(out)
        
        # BACKPROPAGATION
        error = A[-1] - y
        # applying the chain rule
        D = [error * self.sigmoid_deriv(A[-1])]
        for layer in np.arange(len(A) - 2, 0 ,-1):
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_deriv(A[layer])
            D.append(delta)

        # the weight update phase
        D = D[::-1]
        for layer in np.arange(0, len(self.W)):
            # actual learning
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])

    # predict on test set
    def predict(self, X, addBias=True):
        p = np.atleast_2d(X)

        if addBias:
            p = np.c_[p, np.ones((p.shape[0]))]
    
        for layer in np.arange(0, len(self.W)):
            p = self.sigmoid(np.dot(p, self.W[layer]))

        return p

    def calculate_loss(self, X, targets):
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, addBias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)
        return loss
        