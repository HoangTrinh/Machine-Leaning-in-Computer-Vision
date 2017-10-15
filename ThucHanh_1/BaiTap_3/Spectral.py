# Apply Spectral on labeled face dataset
# Author: Trinh Man Hoang - 14520320
# Last Updated: 5/10/2017


import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from ThucHanh_1.BaiTap_3.PreparedData import load


# Load features from prepared file
X = load()

# Calculate similarity matrix
similar_data = cosine_similarity(X)

# Apply Spectral
y = SpectralClustering(n_clusters=7, eigen_solver='arpack', affinity='precomputed').fit_predict(similar_data)

# Visualize result
reduced_data = PCA(n_components=2).fit_transform(X)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=y)
plt.title('Bai Tap 3 - Spectral')
plt.show()
