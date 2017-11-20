# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

from timeit import default_timer as timer
import os

start = timer()

#initializing CNN
classifier = Sequential()

#Step 1 - Convolution 
classifier.add(Convolution2D(32,3,input_shape=(64,64,3),activation='relu'))

#Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Step 3 - Flattening
classifier.add(Flatten())

#Step 4 - Full Connection
classifier.add(Dense(units=258,activation='relu'))

classifier.add(Dense(units=10,activation='sigmoid'))

classifier.compile(optimizer='adam',loss = 'categorical_crossentropy',metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        './newgen/localization_data/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        './newgen/localization_data/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)

end = timer()
print(end - start)
os.system('say "your program has finished"')