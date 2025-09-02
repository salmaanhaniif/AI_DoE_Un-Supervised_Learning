import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

class LogisticRegression():
    def __init__(self, learnRate=0.001, iteration=10000, regTerm='l2', regLambda=0):
        self.learnRate = learnRate
        self.iteration = iteration
        self.regTerm = regTerm
        self.regLambda = regLambda
        self.weights = None
        self.bias = None

        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

        n_Data = X_train.shape[0]
        n_Features = X_train.shape[1]

        self.weights = np.zeros(n_Features)
        self.bias = 0

        for i in range(self.iteration):
            linearPrediction = np.dot(self.X_train, self.weights) + self.bias
            predictions = sigmoid(linearPrediction)
            # print(f"linearPrediction : {linearPrediction}, predictions: {predictions}")

            # Weight Gradient
            xTrain_Transpose = self.X_train.T
            dw = (1/n_Data) * np.dot(xTrain_Transpose, (predictions-self.y_train)) # Transpose matric X
            # Bias
            db = (1/n_Data) * np.sum(predictions-self.y_train)

            if self.regTerm == "l2":
                dw += (self.regLambda / n_Data) * 2 * self.weights
            elif self.regTerm == "l1":
                dw += (self.regLambda / n_Data) * np.sign(self.weights)

            self.weights = self.weights - self.learnRate*dw
            self.bias = self.bias - self.learnRate*db

            # print(f"weight, bias = {self.weights}, {self.bias}")
            # print(f"dw, db = {dw}, {db}")
        
    
    def predict(self, X):
        linearPredictions = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linearPredictions) # logistic prediction
        # print(f"linear pred : {linearPredictions}, logistics : {y_pred}")
        predictions = [0 if y<=0.5 else 1 for y in y_pred]
        return predictions