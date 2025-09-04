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
        
        n_features = X.shape[1]
        featIdxs = np.random.choice(n_features, self.n_Features, replace=False)
        
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

class RandomForest:
    def __init__(self, n_trees=10, minDataSplit=5, maxDepth=100, n_features=None):
        self.n_trees = n_trees
        self.minDataSplit = minDataSplit
        self.maxDepth = maxDepth
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        # Inisialisasi daftar untuk menampung semua pohon
        self.trees = []
        n_samples, n_features = X.shape
        
        # Jika n_features tidak dispesifikasi, gunakan akar kuadrat dari jumlah fitur
        if self.n_features is None:
            self.n_features = int(np.sqrt(n_features))

        # Loop untuk membuat dan melatih setiap pohon
        for _ in range(self.n_trees):
            # Buat instance DecisionTree baru
            tree = DecisionTree(
                minDataSplit=self.minDataSplit,
                maxDepth=self.maxDepth,
                n_Features=self.n_features
            )
            
            # ambil sampel acak dengan pengembalian
            sample_indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[sample_indices]
            y_sample = y[sample_indices]
            
            # Latih pohon pada sampel data ini
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        # Kumpulkan prediksi dari semua pohon
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        
        # Lakukan voting untuk setiap sampel
        # Transpose array untuk memudahkan iterasi per sampel
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        
        # Hitung modus (label paling sering muncul) dari prediksi setiap sampel
        y_pred = [getModusLabel(preds) for preds in tree_preds]
        
        return np.array(y_pred)

# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# # Buat dataset sintetis
# X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Latih model Random Forest
# rf_model = RandomForest(n_trees=20, maxDepth=10)
# rf_model.fit(X_train, y_train)

# # Lakukan prediksi
# y_pred = rf_model.predict(X_test)

# # Hitung akurasi
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Akurasi model Random Forest: {accuracy:.2f}")