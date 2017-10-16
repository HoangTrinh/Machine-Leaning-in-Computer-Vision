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
    X = np.array([]).reshape(0,1850)
    for image in lfw_people.images:
        lbt_image = local_binary_pattern(image,P=8,R=0.5).flatten()
        X = np.append(X,[lbt_image],axis=0)
    np.save(file='data.npy',arr=X)

# Just uncomment & run the line below 1 time
#save()


# Load features matrix
def load():
    return (np.load('data.npy'), np.load('target.npy'))