import numpy as np

class GaussianNaiveBayes:
    def __init__(self):
        self.X_train = None
        self.y_train = None
    
    def fit(self, X_train, y_train):
        # Mendefinisikan ada kelas apa saja di fitur target
        # Sebenarnya karena kita binary, bisa langsung dibuat 2, tapi GNB ini diminta untuk klasifikasi secara umum
        self.classes = np.unique(y_train)
        n_Classes = len(self.classes)

        # Untuk setiap kelas, cari mean, var, prior
        n_Data = X_train.shape[0]
        n_Features = X_train.shape[1]
        self.mean = np.zeros((n_Classes, n_Features)) #menyimpan nilai rata-rata (mean) dari setiap fitur untuk setiap kelas
        self.var = np.zeros((n_Classes, n_Features)) #menyimpan nilai varians dari setiap fitur untuk setiap kelas
        # Probabilitas kemunculan untuk setiap kelas
        self.prior = np.zeros((n_Classes))

        for i, classs in enumerate(self.classes):
            X_class = X_train[y_train==classs] # Menyaring data X train yang y nya tergolong class ini
            self.prior[i] = X_class.shape[0]/n_Data # Probabilitas prior : membagi jumlah sampel di kelas tersebut dengan jumlah total sampel
            self.mean[i, :] = X_class.mean(axis=0) # Menghitung rata-rata untuk setiap fitur di dalam kelas tersebut dan menyimpannya di baris
            self.var[i, :] = X_class.var(axis=0)
        
    def predict(self, X):
        y_pred = [self.subPredict(x) for x in X]
        return y_pred
    
    def subPredict(self, x):
        posteriors = [] # probabilitas posterior (log-probabilitas) untuk setiap kelas

        # Hitung posterior probability untuk setiap class
        for i, classs in enumerate(self.classes):
            prior = np.log(self.prior[i]) # Ubah prior hasil train ke logaritma
            posterior = np.sum(np.log(self.pdf(i,x)))
            posterior += prior
            posteriors.append(posterior)
        
        return self.classes[np.argmax(posterior)]
    
    def pdf(self, classIdx, x):
        # Probability Densition Fungction Gaussian
        mean = self.mean[classIdx]
        var = self.var[classIdx]
        return (np.exp(-(x-mean)**2 / 2*var)) / (np.sqrt(2*np.pi*var))