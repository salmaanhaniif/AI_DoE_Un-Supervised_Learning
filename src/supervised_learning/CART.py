import numpy as np

def getModusLabel(lst):
    labels, counts = np.unique(lst, return_counts=True)
    return labels[np.argmax(counts)]


class Node:
    def __init__(self, feature=None, threshold=None, leftChild=None, rightChild=None, value=None, isLeaf=False):
        self.feature = feature
        self.threshold = threshold
        self.leftChild = leftChild
        self.rightChild = rightChild
        self.value = value
        self.isLeaf = isLeaf

    def isLeafNode(self):
        return self.isLeaf


class DecisionTree:
    def __init__(self, minDataSplit=5, maxDepth=100, n_Features=None):
        self.minDataSplit = minDataSplit
        self.maxDepth = maxDepth
        self.n_Features = n_Features
        self.root = None
    
    def fit(self, X, y):
        if self.n_Features is None:
            self.n_Features = X.shape[1]
        else:
            self.n_Features = min(X.shape[1], self.n_Features)
        self.root = self.growTree(X, y, depth=0)
    
    def growTree(self, X, y, depth=0):
        n_Data = X.shape[0]
        n_Labels = len(np.unique(y))

        # stopping criteria
        if (n_Labels == 1 or depth >= self.maxDepth or n_Data < self.minDataSplit):
            leafValue = getModusLabel(y)
            return Node(value=leafValue, isLeaf=True)
        
        featIdxs = np.arange(X.shape[1])
        
        # find best split
        bestFeature, bestThres = self.bestSplit(X, y, featIdxs)

        # split data
        leftIdxs, rightIdxs = self.split(X[:, bestFeature], bestThres)

        # Cek kalau ada split yang kosong
        if len(leftIdxs) == 0 or len(rightIdxs) == 0:
            leafValue = getModusLabel(y)
            return Node(value=leafValue, isLeaf=True)

        # recursive grow
        leftChild = self.growTree(X[leftIdxs, :], y[leftIdxs], depth+1)
        rightChild = self.growTree(X[rightIdxs, :], y[rightIdxs], depth+1)

        return Node(feature=bestFeature, threshold=bestThres, leftChild=leftChild, rightChild=rightChild)

    def bestSplit(self, X, y, featIdxs):
        bestGain = -1
        splitIdx = None 
        splitThres = None

        for featIdx in featIdxs:
            X_Column = X[:, featIdx]
            thresholds = np.unique(X_Column)

            for thres in thresholds:
                gain = self.informationGain(y, X_Column, thres)
                if gain > bestGain:
                    bestGain = gain
                    splitIdx = featIdx
                    splitThres = thres
        
        return splitIdx, splitThres
    
    def informationGain(self, y, X_Column, thres):
        parEntr = self.entropy(y)

        leftIdxs, rightIdxs = self.split(X_Column, thres)
        if len(leftIdxs) == 0 or len(rightIdxs) == 0:
            return 0
        
        n = len(y)
        nLeft, nRight = len(leftIdxs), len(rightIdxs)
        leftEntr = self.entropy(y[leftIdxs])
        rightEntr = self.entropy(y[rightIdxs])

        ChildEntr = (nLeft/n)*leftEntr + (nRight/n)*rightEntr
        return parEntr - ChildEntr

    def split(self, X_Column, splitThres):
        leftIdxs = np.argwhere(X_Column <= splitThres).flatten()
        rightIdxs = np.argwhere(X_Column > splitThres).flatten()
        return leftIdxs, rightIdxs

    def entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p > 0])

    def predict(self, X):
        return np.array([self.traverseTree(x, self.root) for x in X])

    def traverseTree(self, x, node):
        if node.isLeafNode():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self.traverseTree(x, node.leftChild)
        return self.traverseTree(x, node.rightChild)