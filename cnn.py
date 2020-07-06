## CNN example

import tensorflow as tf
from tensorflow import keras
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import pydot 


fashion_mnist = keras.datasets.fashion_mnist
(x_train_full, y_train_full), (x_test, y_test) = fashion_mnist.load_data()
class_names = ['T-shirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

# Data Reshape and normalization
x_train_full = x_train_full.reshape((60000,28,28,1))   # no. of images and height, width and channel dimensions
x_test = x_test.reshape((10000,28,28,1))
x_train_norm = x_train_full/255.0
x_test_norm = x_test/255.0 

x_valid, x_train = x_train_norm[:5000], x_train_norm[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
x_test = x_test_norm

# Model building 

## Model Architecture
# Input layer - 28 x 28 x 1 (filter size - 3 x 3 with stride 1)
# Convo layer - 26 x 26 x 32 (26 x 26 padding pixels with 32 filters)
# Pooling layer - 13 x 13 x 32 (2 x 2 max pooling)
# Flatten layer - 5408 values 
# Dense layer 1 - 300 neurons
# Dense layer 2 - 100 neurons
# Output layer - 10 neurons

model2 = keras.models.Sequential()
model2.add(keras.layers.Conv2D(filters=32, 
                               kernel_size=(3,3),
                               strides=1,
                               padding='valid',
                               activation='relu',
                               input_shape=(28,28,1)))
model2.add(keras.layers.MaxPooling2D((2,2)))

model2.add(keras.layers.Flatten())
model2.add(keras.layers.Dense(300, activation='relu'))
model2.add(keras.layers.Dense(100, activation='relu'))
model2.add(keras.layers.Dense(10, activation='softmax'))
print(model2.summary())

keras.utils.plot_model(model2)   # visualise model

# Model training
model2.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model2_history = model2.fit(x_train, y_train, epochs=30, batch_size=64, validation_data=(x_valid, y_valid))

pd.DataFrame(model2_history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()

# Evaluate model and predict data
ev = model2.evaluate(x_test_norm, y_test)
print(ev)

x_newsample = x_test[:3]
y_pred = model2.predict_classes(x_newsample)
print(y_pred)
