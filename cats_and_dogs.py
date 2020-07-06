## Cats and Dogs Classification Project

import tensorflow as tf
from tensorflow import keras
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import pydot


## Data
# Get data from: https://www.kaggle.com/c/dogs-vs-cats/data and save them into 3 directories:
train_dir = 'cats_and_dogs_small/train'
validation_dir = 'cats_and_dogs_small/validation' 
test_dir = 'cats_and_dogs_small/test'


## Data preprocessing
# Read picture files
# Decode the jpeg content to RGB grid of pixels
# Convert these into floating point tensors
# Rescale the pixel values from 0-255 to 0-1 interval

from tensorflow.keras.preprocessing.image import ImageDataGenerator
# generating batches of tensor image data
train_datagen = ImageDataGenerator(rescale=1./255) 
test_datagen = ImageDataGenerator(rescale=1./255) 

train_generator = train_datagen.flow_from_directory(train_dir, 
                                                    target_size=(150,150),
                                                    batch_size=20,
                                                    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(validation_dir, 
                                                    target_size=(150,150),
                                                    batch_size=20,
                                                    class_mode='binary')


## Model Architecture

from tensorflow.keras import layers, models

model = models.Sequential()

model.add(layers.Conv2D(32, (3,3), activation='relu',
                        input_shape=(150,150,3)))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64, (3,3), activation='relu',   # filter size increase by 2 and image size decrease by 2 from layer to layer
                        input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128, (3,3), activation='relu',  
                        input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128, (3,3), activation='relu',
                        input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))

print(model.summary())


# Compile and train model

from tensorflow.keras import optimizers

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),   # RMSprop have some advantage over SGD while doing image processing 
              metrics=['acc'])

history = model.fit_generator(train_generator, 
                              steps_per_epoch=100,   # stopping point for generator (2000 images/20 images per batch = 100)
                              epochs=20,
                              validation_data=validation_generator,
                              validation_steps=50)

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()    # graph suggest that there is overfitting in our model

## to fix this, we will modify our existing data in many different forms (data augmentation), then train the model again


## Data augmentation
# Rotate
# Shear
# Scale
# Zoom 

from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,    # set augmentation ranges, imagedatagenerator will randomise the augmentation process
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True) 
test_datagen = ImageDataGenerator(rescale=1./255) 

train_generator = train_datagen.flow_from_directory(train_dir, 
                                                    target_size=(150,150),
                                                    batch_size=32,
                                                    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(validation_dir, 
                                                    target_size=(150,150),
                                                    batch_size=32,
                                                    class_mode='binary')


## Augmented Model Architecture 

from tensorflow.keras import layers, models

model = models.Sequential()

model.add(layers.Conv2D(32, (3,3), activation='relu',
                        input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64, (3,3), activation='relu',   # filter size increase by 2 and image size decrease by 2 from layer to layer
                        input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128, (3,3), activation='relu',  
                        input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128, (3,3), activation='relu',
                        input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())

model.add(layers.Dropout(0.5))   # deactivate 50% neurons during each training epoch randomly, dropout helps to avoid overfitting 

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))

print(model.summary())


# Compile model and repeat training

from tensorflow.keras import optimizers
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),   
              metrics=['acc'])

history = model.fit_generator(train_generator, 
                              steps_per_epoch=100, 
                              epochs=100,
                              validation_data=validation_generator,
                              validation_steps=50)

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()    # no more evidence of overfitting :)


## Transfer learning using VGG16 model

# Implement similar preprocessing

from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,   
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True) 
test_datagen = ImageDataGenerator(rescale=1./255) 

train_generator = train_datagen.flow_from_directory(train_dir, 
                                                    target_size=(150,150),
                                                    batch_size=20,
                                                    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(validation_dir, 
                                                    target_size=(150,150),
                                                    batch_size=20,
                                                    class_mode='binary')


# Using VGG16 convolutional base

from tensorflow.keras.applications import VGG16
conv_base = VGG16(weights='imagenet',
                  include_top=False,   # only use the conv base
                  input_shape=(150,150,3))

print(conv_base.summary())

# Build own fully connected neural network

modeltl = models.Sequential()
modeltl.add(conv_base)
modeltl.add(layers.Flatten())
modeltl.add(layers.Dense(256, activation='relu'))
modeltl.add(layers.Dense(1, activation='sigmoid'))

print(modeltl.summary())

# to freeze the training parameters of the conv base, use:
# conv_base.trainable = False 
# (able to save time)


# Compile and train modeltl 

from tensorflow.keras import optimizers
modeltl.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),   # smaller lr as we want to finetune the existing model (which is already optimized well)
              metrics=['acc'])

checkpoint_cb = keras.callbacks.ModelCheckpoint('modeltl - {epoch:02d}.h5')

history = modeltl.fit_generator(train_generator, 
                              steps_per_epoch=100,  
                              epochs=30,
                              validation_data=validation_generator,
                              validation_steps=50,
                              callbacks=[checkpoint_cb])

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()   # very accurate model! (~ 97.5% accuracy for both train set and validation set)


# Testing the model

test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=(150,150),
                                                  batch_size=20,
                                                  class_mode='binary')
                                                  
modeltl.evaluate_generator(test_generator, steps=50)
