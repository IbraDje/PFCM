import numpy as np
import matplotlib.image as mpimg
from PFCM import pfcm
import time


def creat_image(labels, centers):
    """ color each pixel with the cluster's center color that it belongs to """
    img = np.zeros(shape=(labels.shape[0], labels.shape[1], 3))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] = centers[labels[i, j]]
    if(img.max() > 1):
        img /= 255
    mpimg.imsave('Image Result.jpg', img) # image result
    return img


clusters = 4 # define the number of clusters
img = mpimg.imread('rainbow-page2.jpg') # read the test image
start_time = time.clock() # start calculating the execution time
centers, U, T, obj_fcn = pfcm(
    img.reshape(img.shape[0]*img.shape[1], img.shape[2]), clusters) # calling the pfcm function on the image after reshaping it
elapsed_time = time.clock() - start_time # end calculating the execution time
labels = np.argmax(U, axis=0).reshape(img.shape[0], img.shape[1]) # assing each pixel to its closest cluster
creat_image(labels, centers) # creat an image with the assigned clusters
print(f'elapsed time : {round(elapsed_time, 3)} seconds') # printing the execution time
