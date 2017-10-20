# Apply Clustering Algorithm in face datasets
# Author: Trinh Man Hoang - 14520320
# Last Updated: 16/10/2017

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from ThucHanh_1.BaiTap_2 import Agglomerative, Kmeans, Spectral, DBSCAN
from ThucHanh_1.BaiTap_3.PreparedData import load
from sklearn import metrics


(data, target) = load()
# Kmeans
kmeansResult = Kmeans.kmeans(data, 7)
k_score = metrics.adjusted_mutual_info_score(target, kmeansResult)

# Spectral
spectralResult = Spectral.spectral(data, 7)
s_score = metrics.adjusted_mutual_info_score(target, spectralResult)

# DBSCAN
DBSCAN = DBSCAN.dbscan(data,40, 2)
d_score = metrics.adjusted_mutual_info_score(target, DBSCAN)

# Agglomerative
aggResult = Agglomerative.agglomerative(data, 7)
a_score = metrics.adjusted_mutual_info_score(target, aggResult)

# Visualize result
X = PCA(n_components=2).fit_transform(data)

plt.figure(1)
plt.scatter(X[:, 0], X[:, 1], c=target)
plt.title('Bai Tap 3 - True Result')

plt.figure(2)
plt.subplot(221)
plt.scatter(X[:, 0], X[:, 1], c=kmeansResult)
plt.title('Bai Tap 3 - Kmeans %0.2f'%(k_score*100) + '\u0025')
plt.xticks([])
plt.yticks([])
plt.subplot(222)
plt.scatter(X[:, 0], X[:, 1], c=spectralResult)
plt.title('Bai Tap 3 - Spectral %0.2f'%(s_score*100) + '\u0025')
plt.xticks([])
plt.yticks([])
plt.subplot(223)
plt.scatter(X[:, 0], X[:, 1], c=DBSCAN)
plt.xticks([])
plt.yticks([])
plt.title('Bai Tap 3 - DBSCAN %0.2f'%(d_score*100) + '\u0025')
plt.subplot(224)
plt.scatter(X[:, 0], X[:, 1], c=aggResult)
plt.title('Bai Tap 3 - Agglomerative %0.2f'%(a_score*100) + '\u0025')
plt.xticks([])
plt.yticks([])

plt.show()