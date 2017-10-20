# Prepare data for execution
# Author: Trinh Man Hoang - 14520320
# Last Updated: 4/10/2017


from sklearn.datasets import fetch_lfw_people
from skimage.feature import local_binary_pattern
import numpy as np


# Extract LBP features and save in data.npy
def save():
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    np.save(file='target.npy', arr=lfw_people.target)
    X = []
    for i in range(len(lfw_people.images)):
        lbt_image = local_binary_pattern(lfw_people.images[i], P=24, R=3, method='uniform')
        (lbt_hist,_) = np.histogram(lbt_image.ravel(), bins=int(lbt_image.max() + 1), range=(0, 24 + 2))
        X.append(lbt_hist)
    X = np.array(X)
    np.save(file='data.npy',arr=X)

# Just uncomment & run the line below 1 time
#save()


# Load features matrix
def load():
    return (np.load('data.npy'), np.load('target.npy'))