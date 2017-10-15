# Prepare data for execution
# Author: Trinh Man Hoang - 14520320
# Last Updated: 10/10/2017


import glob
from skimage.io import imread
import numpy as np
from skimage.transform import resize
from skimage.feature import hog


## Extract HOG features and save in data.npy
def save():
    X = np.array([]).reshape(0, 756)
    for filename in glob.glob('cars_train/*.jpg'):
        im = imread(filename,as_grey=True)
        im = resize(im,(50,75), mode='constant')
        x = hog(im, block_norm='L2', orientations=3)
        X = np.append(X,[x],axis=0)
    np.save(file='data.npy', arr=X)

# Just download data set, put it into current folder, uncomment & run the line below 1 time
#save()


# Load features matrix
def load():
    return np.load('data.npy')