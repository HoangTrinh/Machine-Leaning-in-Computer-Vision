# Apply Spectral on car data set
# Author: Trinh Man Hoang - 14520320
# Last Updated: 11/10/2017


import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from ThucHanh_1.BaiTap_4.PreparedData import load


# Load features from prepared file
data = load()

# Calculate similarity matrix
similar_data = cosine_similarity(data)

# Apply Spectral
y = SpectralClustering(n_clusters=7, eigen_solver='arpack', affinity='precomputed').fit_predict(similar_data)

# Visualize result
reduced_data = PCA(n_components=2).fit_transform(data)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=y)
plt.title('Bai Tap 4 - Spectral')
plt.show()