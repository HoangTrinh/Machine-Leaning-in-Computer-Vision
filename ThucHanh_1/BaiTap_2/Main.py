# Apply Clustering Algorithm in hand digits
# Author: Trinh Man Hoang - 14520320
# Last Updated: 16/10/2017



import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from ThucHanh_1.BaiTap_2 import Agglomerative, Kmeans, Spectral, DBSCAN

digits = load_digits()
data = digits.data

# Kmeans
kmeansResult = Kmeans.kmeans(data, 10)

# Spectral
spectralResult = Spectral.spectral(data, 10)

# DBSCAN
DBSCAN = DBSCAN.dbscan(data, 17.6, 1)

# Agglomerative
aggResult = Agglomerative.agglomerative(data, 10)

# Visualize result
X = PCA(n_components=2).fit_transform(data)

plt.figure(1)
plt.scatter(X[:, 0], X[:, 1], c=digits.target)
plt.title('Bai Tap 2 - True Result')

plt.figure(2)
plt.subplot(221)
plt.scatter(X[:, 0], X[:, 1], c=kmeansResult)
plt.title('Bai Tap 2 - Kmeans')
plt.subplot(222)
plt.scatter(X[:, 0], X[:, 1], c=spectralResult)
plt.title('Bai Tap 2 - Spectral')
plt.subplot(223)
plt.scatter(X[:, 0], X[:, 1], c=DBSCAN)
plt.title('Bai Tap 2 - DBSCAN')
plt.subplot(224)
plt.scatter(X[:, 0], X[:, 1], c=aggResult)
plt.title('Bai Tap 2 - Agglomerative')

plt.show()