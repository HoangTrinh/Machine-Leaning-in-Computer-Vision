# Apply Clustering Algorithm in cars dataset
# Author: Trinh Man Hoang - 14520320
# Last Updated: 16/10/2017


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from ThucHanh_1.BaiTap_2 import Agglomerative, Kmeans, Spectral, DBSCAN
from ThucHanh_1.BaiTap_4.PreparedData import load
from sklearn.preprocessing import scale

spec_data = load()
data = scale(load())
X = PCA(n_components=2).fit_transform(data)

# Kmeans
kmeansResult = Kmeans.kmeans(X, 7)

# Spectral

spectralResult = Spectral.spectral(spec_data, 7)

# DBSCAN
DBSCAN = DBSCAN.dbscan(X, 0.6, 3)

# Agglomerative
aggResult = Agglomerative.agglomerative(X, 7)

# Visualize result


plt.subplot(221)
plt.scatter(X[:, 0], X[:, 1], c=kmeansResult)
plt.title('Bai Tap 4 - Kmeans')
plt.subplot(222)
plt.scatter(X[:, 0], X[:, 1], c=spectralResult)
plt.title('Bai Tap 4 - Spectral')
plt.subplot(223)
plt.scatter(X[:, 0], X[:, 1], c=DBSCAN)
plt.title('Bai Tap 4 - DBSCAN')
plt.subplot(224)
plt.scatter(X[:, 0], X[:, 1], c=aggResult)
plt.title('Bai Tap 4 - Agglomerative')

plt.show()