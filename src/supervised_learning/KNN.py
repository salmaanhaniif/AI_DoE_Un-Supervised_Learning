import numpy as np

def euclideanDistance(a, b):
    return np.sqrt(np.sum((a-b)**2))

def manhattanDistance(a, b):
    return np.sum(abs(a-b))

def minkowskiDistance(a, b, p=3):
    return np.power(np.sum(np.abs(a - b)**p), 1/p)

class KNN:
    def __init__(self, k=4, method='euclidean', p=3):
        self.k = k
        self.method = method
        self.p = p
        self.X_train = None
        self.y_train = None
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X):
        #do something
        predictions = []
        for x in X:
            predictions.append(self.subPredict(x))
        return predictions
    
    def getMostCommonLabel(self, labels):
        # most common = modus
        labelCounts = {}
        for label in labels:
            if label in labelCounts:
                labelCounts[label] += 1
            else:
                labelCounts[label] = 1
        
        mostCommonLabel = None
        maxCount = 0
        for label, count in labelCounts.items():
            if count > maxCount:
                maxCount = count
                mostCommonLabel = label
        
        return mostCommonLabel

    def subPredict(self, x):
        # Hitung jarak x yang akan diprediksi dengan semua X di X_train
        distanceList = []
        for i, _x in enumerate(self.X_train):
            if self.method == 'euclidean':
                distance = (euclideanDistance(x, _x))
            elif self.method == 'manhattan':
                distance = (manhattanDistance(x, _x))
            else :
                distance = (minkowskiDistance(x, _x, p=self.p))
        
            distanceList.append((distance, i)) # pasangan jarak,index

        # Urutkan berdasarkan jarak 
        distanceList.sort(key=lambda item: item[0])

        k_NearestItems = distanceList[:self.k]
        k_Indices = []
        for item in k_NearestItems:
            k_Indices.append(item[1])
        
        k_NearestLabel = []
        for i in k_Indices:
            k_NearestLabel.append(self.y_train[i])

        # Majority vote
        mostCommon = self.getMostCommonLabel(k_NearestLabel)
        return mostCommon