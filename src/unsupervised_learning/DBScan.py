import numpy as np
from collections import deque

class DBScan:

    def __init__(self, eps, minSamples, distanceMetric='Euclidean', p=3):
        self.eps = eps
        self.minSamples = minSamples
        self.distanceMetric = distanceMetric.lower()
        self.p = p
        self.labels_ = None
        self.X_ = None

    def _getDistance(self, p1, p2):
        p1 = np.asarray(p1)
        p2 = np.asarray(p2)
        if self.distanceMetric == 'euclidean':
            return np.sqrt(np.sum((p1 - p2) ** 2))
        elif self.distanceMetric == 'manhattan':
            return np.sum(np.abs(p1 - p2))
        elif self.distanceMetric == 'minkowski':
            return np.sum(np.abs(p1 - p2) ** self.p) ** (1 / self.p)
        return 0

    def fit(self, X):
        self.X_ = X
        nSamples = X.shape[0]
        self.labels_ = np.full(nSamples, -2, dtype=int)
        clusterId = 0

        for i in range(nSamples):
            if self.labels_[i] != -2:
                continue

            neighbors = self._getNeighbors(i)
            
            if len(neighbors) < self.minSamples:
                self.labels_[i] = -1
            else:
                self._expandCluster(i, neighbors, clusterId)
                clusterId += 1

    def _getNeighbors(self, i):
        neighbors = []
        for j in range(self.X_.shape[0]):
            if self._getDistance(self.X_[i], self.X_[j]) <= self.eps:
                neighbors.append(j)
        return neighbors

    def _expandCluster(self, i, neighbors, clusterId):
        self.labels_[i] = clusterId
        q = deque(neighbors)

        while q:
            currentPointIdx = q.popleft()
            if self.labels_[currentPointIdx] == -1:
                self.labels_[currentPointIdx] = clusterId

            if self.labels_[currentPointIdx] != -2:
                continue

            self.labels_[currentPointIdx] = clusterId
            
            newNeighbors = self._getNeighbors(currentPointIdx)
            if len(newNeighbors) >= self.minSamples:
                q.extend(newNeighbors)

    def predict(self, XNew):      
        predictedLabels = np.full(XNew.shape[0], -1, dtype=int)

        for i, pointNew in enumerate(XNew):
            minDist = float('inf')
            assignedCluster = -1
            
            # Find the nearest core point from the fitted data
            for j, pointFit in enumerate(self.X_):
                if self.labels_[j] != -1:  # Check if it is a core or border point
                    dist = self._getDistance(pointNew, pointFit)
                    if dist <= self.eps:
                        if dist < minDist:
                            minDist = dist
                            assignedCluster = self.labels_[j]
            predictedLabels[i] = assignedCluster
            
        return predictedLabels
