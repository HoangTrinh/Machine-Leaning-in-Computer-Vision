# Apply Kmeans
# Author: Trinh Man Hoang - 14520320
# Last Updated: 30/09/2017


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

def kmeans(data,k):
    # Load data
    X = data

    # Apply Kmeans
    y = KMeans(n_clusters=k).fit_predict(X)

    return y

