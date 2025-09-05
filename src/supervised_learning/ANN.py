import numpy as np

def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A, Z

def sigmoidDerivative(dA, Z):
    A, _ = sigmoid(Z)
    dZ = dA * A * (1 - A)
    return dZ

def relu(Z):
    A = np.maximum(0, Z)
    return A, Z

def reluDerivative(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def linear(Z):
    return Z, Z

def linearDerivative(dA, Z):
    return dA

def softmax(Z):
    exps = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exps / np.sum(exps, axis=0, keepdims=True), Z

def binaryCrossEntropyLoss(A, Y):
    if Y.ndim == 1:
        Y = Y.reshape(1, -1)
    m = Y.shape[1]
    A = np.clip(A, 1e-10, 1 - 1e-10)
    loss = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) / m
    dA = -(np.divide(Y, A) - np.divide(1 - Y, 1 - A))
    return loss, dA

def mseLoss(A, Y):
    if Y.ndim == 1:
        Y = Y.reshape(1, -1)
    m = Y.shape[1]
    loss = np.sum(np.square(A - Y)) / (2 * m)
    dA = (A - Y) / m
    return loss, dA

class ANN:
    def __init__(self, layerDims, activations, lossFunction='cross_entropy', regularization=None, l2Lambda=0.01, epochs=10000, learningRate=0.001):
        self.layerDims = layerDims
        self.activations = activations
        self.lossFunction = self._getLossFunction(lossFunction)
        self.regularization = regularization
        self.l2Lambda = l2Lambda
        self.epochs = epochs
        self.learningRate = learningRate
        self.parameters = {}
        self.grads = {}
        self.caches = {}
        self._initializeParameters()
    
    def _getActivationFunction(self, name):
        if name == 'sigmoid':
            return sigmoid, sigmoidDerivative
        if name == 'relu':
            return relu, reluDerivative
        if name == 'linear':
            return linear, linearDerivative
        if name == 'softmax':
            return softmax, None
        raise ValueError("Invalid activation function name")

    def _getLossFunction(self, name):
        if name == 'binary_cross_entropy':
            return binaryCrossEntropyLoss
        if name == 'cross_entropy':
            return self._crossEntropyWarning
        if name == 'mse':
            return mseLoss
        raise ValueError("Invalid loss function name")

    def _crossEntropyWarning(self, A, Y):
        print("Warning: 'cross_entropy' is generally for multi-class. Using 'binary_cross_entropy' is recommended for binary classification.")
        return crossEntropyLoss(A, Y)

    def _initializeParameters(self):
        np.random.seed(1)
        for l in range(1, len(self.layerDims)):
            if self.activations[l-1] == 'relu':
                self.parameters['W' + str(l)] = np.random.randn(self.layerDims[l], self.layerDims[l-1]) * np.sqrt(2 / self.layerDims[l-1])
            else:
                self.parameters['W' + str(l)] = np.random.randn(self.layerDims[l], self.layerDims[l-1]) * 0.01
            self.parameters['b' + str(l)] = np.zeros((self.layerDims[l], 1))

    def _forwardPropagation(self, X):
        A_prev = X
        L = len(self.layerDims)
        for l in range(1, L):
            W = self.parameters['W' + str(l)]
            b = self.parameters['b' + str(l)]
            Z = np.dot(W, A_prev) + b
            
            activationFunc, _ = self._getActivationFunction(self.activations[l-1])
            A, Z_cache = activationFunc(Z)
            
            self.caches['A' + str(l-1)] = A_prev
            self.caches['Z' + str(l)] = Z_cache
            A_prev = A
        
        return A

    def _backwardPropagation(self, AL, Y):
        L = len(self.layerDims)
        self.grads = {}
        m = AL.shape[1]

        loss, dAL = self.lossFunction(AL, Y)
        
        for l in reversed(range(1, L)):
            dA = dAL
            Z_cache = self.caches['Z' + str(l)]
            A_prev = self.caches['A' + str(l-1)]
            
            _, activationDeriv = self._getActivationFunction(self.activations[l-1])
            
            if self.activations[l-1] == 'softmax':
                dZ = AL - Y
            else:
                dZ = activationDeriv(dA, Z_cache)

            self.grads['dW' + str(l)] = np.dot(dZ, A_prev.T) / m
            self.grads['db' + str(l)] = np.sum(dZ, axis=1, keepdims=True) / m
            
            if self.regularization == 'l2':
                self.grads['dW' + str(l)] += (self.l2Lambda / m) * self.parameters['W' + str(l)]

            if l > 1:
                W = self.parameters['W' + str(l)]
                dAL = np.dot(W.T, dZ)
        
        return loss

    def _updateParameters(self):
        L = len(self.layerDims)
        for l in range(1, L):
            self.parameters['W' + str(l)] -= self.learningRate * self.grads['dW' + str(l)]
            self.parameters['b' + str(l)] -= self.learningRate * self.grads['db' + str(l)]

    def fit(self, X, Y):
        self.losses = []
        for i in range(self.epochs):
            AL = self._forwardPropagation(X)
            
            loss = self._backwardPropagation(AL, Y)
            self.losses.append(loss)
            
            self._updateParameters()
            
            if i % 1000 == 0:
                print(f"Epoch {i}, Loss: {loss:.4f}")

    def predict(self, X):
        AL = self._forwardPropagation(X)
        predictions = (AL > 0.5).astype(int)
        return predictions.T