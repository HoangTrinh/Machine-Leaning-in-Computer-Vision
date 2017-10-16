# Apply Clustering Algorithm in face datasets
# Author: Trinh Man Hoang - 14520320
# Last Updated: 16/10/2017

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from ThucHanh_1.BaiTap_2 import Agglomerative, Kmeans, Spectral, DBSCAN
from ThucHanh_1.BaiTap_3.PreparedData import load


(data, target) = load()

# Kmeans
kmeansResult = Kmeans.kmeans(data, 7)

# Spectral
spectralResult = Spectral.spectral(data, 7)

# DBSCAN
DBSCAN = DBSCAN.dbscan(data, 20, 1)

# Agglomerative
aggResult = Agglomerative.agglomerative(data, 7)

# Visualize result
X = PCA(n_components=2).fit_transform(data)

plt.figure(1)
plt.scatter(X[:, 0], X[:, 1], c=target)
plt.title('Bai Tap 3 - True Result')

plt.figure(2)
plt.subplot(221)
plt.scatter(X[:, 0], X[:, 1], c=kmeansResult)
plt.title('Bai Tap 3 - Kmeans')
plt.subplot(222)
plt.scatter(X[:, 0], X[:, 1], c=spectralResult)
plt.title('Bai Tap 3 - Spectral')
plt.subplot(223)
plt.scatter(X[:, 0], X[:, 1], c=DBSCAN)
plt.title('Bai Tap 3 - DBSCAN')
plt.subplot(224)
plt.scatter(X[:, 0], X[:, 1], c=aggResult)
plt.title('Bai Tap 3 - Agglomerative')

plt.show()