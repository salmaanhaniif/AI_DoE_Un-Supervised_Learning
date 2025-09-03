import numpy as np
import matplotlib.pyplot as plt

class SVM():

    def __init__(self, learnRate = 0.001, lambdaa=0.01, iterations=10000):
        self.learnRate = learnRate
        self.lambdaa = lambdaa
        self.iterations = iterations
        self.w = None
        self.b = None
    
    def fit(self, X_train, y_train):
        nData, nFeats = X_train.shape

        y = np.where(y_train == 0, -1, 1)

        self.w = np.zeros(nFeats)
        self.b = 0

        for _ in range(self.iterations):
            for idx, x in enumerate(X_train):
                condition = y[idx]  * (np.dot(x, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.learnRate * (2 * self.lambdaa * self.w)
                else:
                    self.w -= self.learnRate * (2 * self.lambdaa * self.w - np.dot(x, y[idx]))
                    self.b -= self.learnRate * y[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        prediction = np.where(approx >= 0, 1, -1)
        pred_return = np.where(prediction == -1, 0, 1)
        return pred_return

    def visualizeSVM(self, X, y):
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', alpha=0.7)
        ax = plt.gca()

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100),
                             np.linspace(ylim[0], ylim[1], 100))
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('SVM Decision Boundary')
        plt.show()