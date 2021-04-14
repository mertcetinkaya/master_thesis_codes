import numpy as np
from os import listdir,makedirs
from os.path import isfile, join, exists
import cv2
from skimage import exposure, feature, transform
import matplotlib.pyplot as plt
import time
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout


def get_hog_feature(image_path,ppc,edge_length):
    im=cv2.imread(image_path)
    grayim = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    grayim = transform.resize(grayim,(edge_length,edge_length))
    (hogFeature, hogImage) = feature.hog(grayim, orientations=9, pixels_per_cell=(ppc, ppc),
    cells_per_block=(2, 2), transform_sqrt=True, visualize=True, block_norm="L1")
    return hogFeature

def produce_basic_cnn_list(edge_length,kernel_list,neuron_list,class_number):
    models=[]
    for i in kernel_list:
        for j in neuron_list:
            model = Sequential()
            model.add(Conv2D(filters=32, kernel_size=(i,i), activation='relu', input_shape=(edge_length,edge_length,3)))
            model.add(Conv2D(filters=32, kernel_size=(i,i), activation='relu'))
            model.add(MaxPool2D(pool_size=(2, 2)))
            model.add(Dropout(rate=0.25))
            model.add(Conv2D(filters=64, kernel_size=(i, i), activation='relu'))
            model.add(Conv2D(filters=64, kernel_size=(i, i), activation='relu'))
            model.add(MaxPool2D(pool_size=(2, 2)))
            model.add(Dropout(rate=0.25))
            model.add(Flatten())
            model.add(Dense(j, activation='relu'))
            model.add(Dropout(rate=0.5))
            model.add(Dense(class_number, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            models.append(model)
    return models