# Prepare data for execution
# Author: Trinh Man Hoang - 14520320
# Last Updated: 10/10/2017



import numpy as np
from sklearn.datasets import fetch_lfw_people
from skimage.feature import hog

def save():
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    np.save(file='target.npy', arr=lfw_people.target)
    X = []
    for image in lfw_people.images:
        lbt_image = hog(image, block_norm='L2')
        X.append(lbt_image)
    X = np.array(X)
    np.save(file='data.npy', arr=X)
#save()

# Load features matrix
def load():
    return (np.load('data.npy'), np.load('target.npy'))