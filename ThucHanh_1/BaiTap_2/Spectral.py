# Apply Spectral
# Author: Trinh Man Hoang - 14520320
# Last Updated: 2/10/2017


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.metrics.pairwise import cosine_similarity

def spectral(data,nCluster):
    # Load data
    X = data

    # Calculate similarity matrix
    similar_data = cosine_similarity(X)

    # Apply Spectral
    y = SpectralClustering(n_clusters=nCluster, eigen_solver='arpack', affinity='precomputed').fit_predict(similar_data)

    return y
