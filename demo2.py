import numpy as np
import matplotlib.image as mpimg
from PFCM import pfcm
import time


def creat_image(labels, centers):
    img = np.zeros(shape=(labels.shape[0], labels.shape[1], 3))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] = centers[labels[i, j]]
    if(img.max() > 1):
        img /= 255
    mpimg.imsave('Image Result.jpg', img) # image result
    return img


clusters = 4
img = mpimg.imread('rainbow-page2.jpg') # test image
start_time = time.clock()
centers, U, T, obj_fcn = pfcm(
    img.reshape(img.shape[0]*img.shape[1], img.shape[2]), clusters)
elapsed_time = time.clock() - start_time
labels = np.argmax(U, axis=0).reshape(img.shape[0], img.shape[1])
creat_image(labels, centers)
print(f'elapsed time : {round(elapsed_time, 3)} seconds')
