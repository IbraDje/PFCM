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
    mpimg.imsave('Image Result.jpg', img)
    return img


def compactness(img, labels, centers):
    WSS = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            WSS += np.sum(np.power(img[i, j]-centers[labels[i, j]], 2))
    return WSS/(labels.shape[0]*labels.shape[1])


def separation(labels, centers):
    BSS = 0
    cluster_size = np.zeros(centers.shape[0])
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            cluster_size[labels[i, j]] += 1
    mean = np.mean(centers, axis=0)
    for k in range(centers.shape[0]):
            BSS += cluster_size[k]*np.sum(np.power(mean - centers[k], 2))
    return BSS/(labels.shape[0]*labels.shape[1])


clusters = 4
img = mpimg.imread('Image thermique et num\\1373.jpg')
# img = mpimg.imread('rainbow-page2.jpg')
start_time = time.clock()
centers, U, T, obj_fcn = pfcm(
    img.reshape(img.shape[0]*img.shape[1], img.shape[2]), clusters)
elapsed_time = time.clock() - start_time
labels = np.argmax(U, axis=0).reshape(img.shape[0], img.shape[1])
WSS = compactness(img, labels, centers)
BSS = separation(labels, centers)
creat_image(labels, centers)
print(f'compactness = {WSS} | separation = {BSS}')
print(f'elapsed time : {round(elapsed_time, 3)} seconds')
