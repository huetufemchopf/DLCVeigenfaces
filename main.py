import cv2
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

path  = "/Users/celine/Library/Mobile Documents/com~apple~CloudDocs/NSC/VLL/DLCV/Exercises/hw1/p3_data/"
n_components = None
data = [],[],[]
categories = ["banana", "fountain", "reef", "tractor"]


for i in categories:
    n = 0
    for j in os.listdir(path + i ):
        data[0].append(i)
        data[1].append(n)
        n +=1
        im = cv2.imread(path + i + "/" + j, )
        b, g, r = cv2.split(im)  # get b,g,r
        rgb_img = cv2.merge([r, g, b])
        data[2].append(rgb_img)


X_train, X_test, y_train, y_test = train_test_split(data[2], data[0],
                                                    stratify=data[0],
                                                    test_size=0.25)
'''



X_train = []
X_test = []
y_train = []
y_test = []
n = 0
for i in files2[0]:
    if int(i) <= 375:
        X_train.append(files2[1][n])
        y_train.append(files2[0][n])

    elif int(i) > 125:
        X_test.append(files2[1][n])
        y_test.append(files2[0][n])
    n += 1

X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

X_train_sliced = []
X_test_sliced = []
y_train_sliced = []
y_test_sliced = []

for i in X_train:
    print(i.shape)
    # image_blocks = view_as_blocks(i, block_shape = (16, 16, 3)).squeeze()
# print(image_blocks[0][0].shape)

plt.figure()
plt.imshow(image_blocks[0][0])


'''
