import numpy as np
import matplotlib.pyplot as plt   # <- harusnya matplotlib.pyplot

def euclideanDistance(a, b):
    return np.sqrt(np.sum((a-b)**2))

class KMeans:
    def __init__(self, k=4, iterations=100, plotSteps=False):
        self.K = k
        self.iterations = iterations
        self.plotSteps = plotSteps

        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]

        # centers of clusters
        self.centroids = []
    
    def predict(self, X):
        self.X = X
        self.nData, self.nFeatures = X.shape

        randomDataIdxs = np.random.choice(self.nData, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in randomDataIdxs]

        for _ in range(self.iterations):
            self.clusters = self.createClusters(self.centroids)

            if self.plotSteps:
                self.plot()

            centroidOld = self.centroids   # <- typo tadi: self.controids
            self.centroids = self.getCentroids(self.clusters)

            if self.isConverged(centroidOld, self.centroids):
                break

            if self.plotSteps:   # <- tadi salah pakai () seolah fungsi
                self.plot()
        
        # classify samples as the index of their cluster
        return self.getClusterLabels(self.clusters)
    
    def getClusterLabels(self, clusters):
        labels = np.empty(self.nData)   # <- tadi salah: self.n_Data
        for clusterIdx, cluster in enumerate(clusters):
            for dataIdx in cluster:
                labels[dataIdx] = clusterIdx
        return labels

    def createClusters(self, centroids):
        clusters = [[] for _ in range(self.K)]
        for idx, data in enumerate(self.X):
            centroidIdx = self.closestCentroid(data, centroids)
            clusters[centroidIdx].append(idx)
        return clusters
    
    def closestCentroid(self, data, centroids):
        distances = [euclideanDistance(data, point) for point in centroids]
        closestIdx = np.argmin(distances)
        return closestIdx
    
    def getCentroids(self, clusters):
        centroids = np.zeros((self.K, self.nFeatures))  # <- tadi self.n
        for clusterIdx, cluster in enumerate(clusters):
            clusterMean = np.mean(self.X[cluster], axis=0)
            centroids[clusterIdx] = clusterMean
        return centroids
    
    def isConverged(self, centroidOld, centroids):
        # distances between old and new centroids
        distances = [euclideanDistance(centroidOld[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, idx in enumerate(self.clusters):
            point = self.X[idx].T
            ax.scatter(*point)
        
        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=3)  # <- typo lindewidth
        
        plt.show()