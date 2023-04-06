import numpy as np
import pandas as pd
import operator
from copy import deepcopy
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

class MVMCNN():
    def __init__(self, k):
        self.k = k
        
    # Fungsi yang digunakan utnuk mengelompokkan data berdasarkan kelasnya
    def groupbyclass (self, df):
        train = {}
        by_class = df.groupby('Class')
        for groups, data in by_class:
            train[groups] = data.values.tolist()
        return train

    # Penghitungan jumlah cluster yang optimal menggunakan silhouette score
    def silhouette (self, train_class):
        silhouette_scores = [] 
        for n_cluster in range(2, 6): # Jumlah cluster minimal dan maksimal yang akan diukur menggunakan silhouette score
            silhouette_scores.append( 
                silhouette_score(train_class, KMeans(n_clusters = n_cluster).fit_predict(train_class))) 
        max_score = silhouette_scores.index(max(silhouette_scores)) + 2         # +2 dikarenakan indeks dimulai dari 0
        return max_score

    # Clustering dengan menggunakan K-Means Clustering
    def kmeans_clustering(self, train_class):
        clust_train = np.array(train_class)

        n = clust_train.shape[0]
        c = clust_train.shape[1]
        std = np.std(clust_train, axis = 0)
        mean = np.mean(clust_train, axis = 0)

        K = self.silhouette(train_class)  # Jumlah K pada K-Means clustering didasarkan pada silhouette score yang sudah didapat

        centroid = np.random.randn(K,c) * std + mean
        centroid0 = np.zeros(centroid.shape) 
        centroid1 = deepcopy(centroid) 
        space = np.zeros((n,K))
        cluster = np.zeros(n)
        max_iter = 100

        for centroids in range(max_iter):
            for i in range(K):
                space[:,i] = np.linalg.norm(clust_train - centroid1[i], axis=1)
            cluster = np.argmin(space, axis = 1)
            centroid0 = deepcopy(centroid1)
            for i in range(K):
                centroid1[i] = np.mean(clust_train[cluster == i], axis=0)
            if centroid0[i].any == centroid1[i].any:
                break
                                            # Output yang diberikan berupa golongan cluster dari masing masing data (tanpa ada value dari data) 
        return cluster                      # Contoh: Kelas 0.0 memiliki 2 cluster, hasil yang diberikan adalah [1 1 1 1 0 0 1 1 0 1 1 0 0 0 1 1 0 0 0 1 0 0 0 1 0 1 1 1 0 1 1] dimana 0 berarti cluster ke 0 dan 1 berarti cluster ke 1

    # Pengelompokan isi cluster berdasarkan golongannya dan proses memasukkan data ke dalam cluster
    def clust_divider(self, train_class, cluster):
        clust_values = pd.DataFrame(train_class)
        clust_values['Clust'] = cluster.tolist()

        clustset = {}
        by_class = clust_values.groupby('Clust')

        for groups, data in by_class:
            data = data.drop(columns=data.columns[-1])
            clustset[groups] = np.array(data)
            
        return clustset

    # Fungsi untuk melakukan penghitungan jarak antara data test dengan seluruh data train
    def index_distance (self, train, test, length, k):
        point1 = np.array(train)
        point2 = np.array(test)
        index_dist, sorted_trainset = [], []
        for i in range(length):
            dist = []
            dist.append(np.linalg.norm(point2[:-1] - point1[i][:-2])) # Penghitungan Euclidean distance data test ke data train ke i
            dist.append(point1[i][:-1])
            index_dist.append(dist) # Hasil penghitungan euclidean distance dan nilai dari data train ke i dimasukkan ke dalam list
        sorted_index = sorted(index_dist,key=lambda l:l[0]) # Pengurutan isi list berdasarkan distance
        for i in range(k):
            sorted_trainset.append(sorted_index[i][1])
        return sorted_trainset # Output yang dihasilkan berupa urutan index / nilai dari data train yang sudah diurutkan berdasarkan distancenya

    def distancetocluster(self, train, test, k):
        length = len(train)
        point2 = test
        index_dist = self.index_distance(train, test, length, k)
        total_distance = []
        distancetotest = 0
        final_distance = 0
        temp = np.zeros(len(point2), dtype = np.float64) # Inisiasi list berisi 0 berdasarkan panjang dari data
        for i in range(k):        # Perapatan dimensi yang mengacu pada formulasi LMPNN pada paper Improved pseudo nearest neighbor classification dan rumus yang diberikan Prof. Suyanto (1/1 * newdistance(1) + 1/2 * newdistance(2) + ... + 1/k * newdistance(k))
            temp += index_dist[i]   
            meanpos = temp/(i+1)                                        # distance * 1/(nomor index) digunakan untuk menentukan posisi baru
            distancetotest = np.linalg.norm(point2[:-1] - meanpos[:-1]) # Penghitungan jarak antara data test dengan posisi terbaru dari data train 
            total_distance.append((1/(i+1))*(distancetotest))           # Jarak baru * 1/(nomor index)
            final_distance = sum(total_distance)                        # Penjumlahan total dari semua distance dari posisi baru
        
        return final_distance

    # Private function untuk fit/ penggabungan x_train dengan y_train (pelabelan data train)
    def __fit(self, x_train, y_train):
        train = []
        tmp = pd.DataFrame(x_train)
        tmp['Class'] = y_train
        train = self.groupbyclass(pd.DataFrame(tmp)) # Trainset dikelompokkan berdasarkan kelasnya
        self.trainset = train
        return self

    # Fungsi fit yang dipanggil pada main code
    def fit(self, x_train, y_train):
        return self.__fit(x_train, y_train)

    def predict(self, x_test):
        k = self.k
        predictions, clustneigh = [], []
        for cls in self.trainset:                                           # Looping sepanjang kelas pada trainset (karena trainset berbentuk dictionary dengan kelas sebagai key nya)
            clust_list = self.kmeans_clustering(self.trainset[cls])         # Proses clustering data dari satu kelas pada trainset
            tempclust = self.clust_divider(self.trainset[cls], clust_list)  # Pengelompokan data berdasarkan golongan clusternya
            for n in range(len(tempclust)):                                 # Loop yang digunakan untuk memisah misahkan kelompok cluster agar tidak terbentuk list multi dimensi
                if n in tempclust:
                    clustneigh.append(tempclust[n])
        for i in range(len(x_test)):                                        # Loop pengujian sepanjang test set
            clustdist = []
            for j in range(len(clustneigh)):
                clust_size = len(clustneigh[j])
                if (clust_size >= k):                                               # Exception yang digunakan untuk mengatasi jumlah data pada cluster lebih sedikit daripada K
                    voters = self.distancetocluster(clustneigh[j], x_test[i], k)    # Penghitungan jarak dengan formulasi LMPNN pada tiap cluster
                    clustdist.append((voters, clustneigh[j][1][-1]))
                elif (clust_size < k):
                    continue
            clustdist.sort(key=operator.itemgetter(0), reverse=False)
            if len(clustdist) == 0:                                                 # Exception yang digunakan untuk mengatasi apabila isi dari semua cluster lebih kecil daripada K
                predictions.append(0)
            else:
                result = clustdist[0][-1]
                predictions.append(result)
        return predictions


       