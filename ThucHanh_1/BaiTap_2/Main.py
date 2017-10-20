# Apply Clustering Algorithm in hand digits
# Author: Trinh Man Hoang - 14520320
# Last Updated: 16/10/2017



import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from ThucHanh_1.BaiTap_2 import Agglomerative, Kmeans, Spectral, DBSCAN
from sklearn import metrics

digits = load_digits()
data = digits.data

# Kmeans
kmeansResult = Kmeans.kmeans(data, 10)
k_score = metrics.adjusted_mutual_info_score(digits.target, kmeansResult)


# Spectral
spectralResult = Spectral.spectral(data, 10)
s_score = metrics.adjusted_mutual_info_score(digits.target, spectralResult)


# DBSCAN
DBSCAN = DBSCAN.dbscan(data, 21.5, 9)
d_score = metrics.adjusted_mutual_info_score(digits.target, DBSCAN)


# Agglomerative
aggResult = Agglomerative.agglomerative(data, 10)
a_score = metrics.adjusted_mutual_info_score(digits.target, aggResult)


# Visualize result
X = PCA(n_components=2).fit_transform(data)

plt.figure(1)
plt.scatter(X[:, 0], X[:, 1], c=digits.target)
plt.title('Bai Tap 2 - True Result')

plt.figure(2)
plt.subplot(221)
plt.scatter(X[:, 0], X[:, 1], c=kmeansResult)
plt.title('Bai Tap 2 - Kmeans %0.2f'%(k_score*100) + '\u0025')
plt.xticks([])
plt.yticks([])
plt.subplot(222)
plt.scatter(X[:, 0], X[:, 1], c=spectralResult)
plt.title('Bai Tap 2 - Spectral %0.2f'%(s_score*100) + '\u0025')
plt.xticks([])
plt.yticks([])
plt.subplot(223)
plt.scatter(X[:, 0], X[:, 1], c=DBSCAN)
plt.xticks([])
plt.yticks([])
plt.title('Bai Tap 2 - DBSCAN %0.2f'%(d_score*100) + '\u0025')
plt.subplot(224)
plt.scatter(X[:, 0], X[:, 1], c=aggResult)
plt.title('Bai Tap 2 - Agglomerative %0.2f'%(a_score*100) + '\u0025')
plt.xticks([])
plt.yticks([])


plt.show()