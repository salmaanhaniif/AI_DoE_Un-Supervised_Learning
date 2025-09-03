import numpy as np

class PCA:
    def __init__(self, n_Components = None, n_Multiplier = 1):
        self.n_components = n_Components
        self.n_Multiplier = n_Multiplier
        self.mean = None
        self.components = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        # covariance, function needs samples as column
        cov = np.cov(X.T)
        # Cari nilai eigenvectors dan eigenvalues
        eigenvectors, eigenvalues = np.linalg.eig(cov)

        # Transpose supaya baris = eigenvector
        eigenvectors = eigenvectors.T
        
        # Urutkan eigen vectors secara descending
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        if self.n_Multiplier<1:
            if self.n_components is None:
                self.n_components = int(self.n_Multiplier * X.shape[1])
            
        self.components = eigenvectors[:self.n_components]
    
    def transform(self, X):
        X = X - self.mean
        return np.dot(X, self.components.T)