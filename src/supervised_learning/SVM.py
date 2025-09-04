import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

class SVM():

    def __init__(self, learnRate=0.001, lambdaa=0.01, iterations=10000):
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

        batch_size = 32
        
        for _ in range(self.iterations):
            # Ambil indeks batch acak
            indices = np.random.choice(nData, batch_size, replace=False)
            X_batch = X_train[indices]
            y_batch = y[indices]

            # Hitung margin
            margins = y_batch * (np.dot(X_batch, self.w) + self.b)
            
            # Cari sampel yang melanggar margin (violation)
            violation_indices = np.where(margins < 1)[0]
            
            # Hitung gradien bobot
            if len(violation_indices) > 0:
                dw = 2 * self.lambdaa * self.w - np.mean(y_batch[violation_indices, np.newaxis] * X_batch[violation_indices], axis=0)
            else:
                dw = 2 * self.lambdaa * self.w
            
            # Hitung gradien bias
            db = 0
            if len(violation_indices) > 0:
                db = -np.mean(y_batch[violation_indices])

            # Perbarui bobot dan bias
            self.w -= self.learnRate * dw
            self.b -= self.learnRate * db

    def predict(self, X):
        approx = np.dot(X, self.w) + self.b
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

# Percobaan pelatihan model dengan scaling

# from sklearn.model_selection import train_test_split
# from sklearn.datasets import make_blobs
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score

# X, y = make_blobs(n_samples=1000, centers=2, random_state=42)

# # 2. Membagi data menjadi set pelatihan dan pengujian
# # test_size = 0.2: 20% data akan digunakan untuk pengujian
# # train_test_split akan memisahkan data X dan y secara acak
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 3. Melakukan Feature Scaling
# # Scaling sangat penting untuk SVM agar performanya optimal
# scaler = StandardScaler()

# # Melatih scaler pada data pelatihan dan mengubahnya
# X_train_scaled = scaler.fit_transform(X_train)

# # Mengubah data pengujian menggunakan scaler yang sama
# X_test_scaled = scaler.transform(X_test)

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# svm_model = SVM()
# svm_model.fit(X_train_scaled, y_train)

# # Saat melakukan prediksi, pastikan data input juga di-scale
# y_pred = svm_model.predict(X_test_scaled)

# accuracy = accuracy_score(y_test, y_pred)
# print(f"Akurasi model SVM: {accuracy:.2f}")