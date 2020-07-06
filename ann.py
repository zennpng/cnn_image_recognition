# ANN - Image Classifier example using tf and keras

import tensorflow as tf
from tensorflow import keras
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import pydot


fashion_mnist = keras.datasets.fashion_mnist    # we use the keras inbuild dataset, fashion_mnist 
(x_train_full, y_train_full), (x_test, y_test) = fashion_mnist.load_data()

plt.imshow(x_train_full[0])    # show 1st image of training set
print(y_train_full[0])    # the image above corresponds to the no.9 category 
class_names = ['T-shirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
print(class_names[y_train_full[0]]) 

# Normalization and train test validation split
# Training set - Used to train the model 
# Validation set - Used to tune the hyperparameters and evaluate the model
# Testing set - Used to test the model after the model has gone through initial vetting by the validation set

# normalize the pixel intensity 
x_train_norm = x_train_full/255.0
x_test_norm = x_test/255.0

x_valid, x_train = x_train_norm[:5000], x_train_norm[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
x_test = x_test_norm

# Model building

## Model Architecture:
# Input - 28 x 28 pixel
# Hidden - x2 with ReLu Activation 
# Output - 10 with Softmax Activation 

model1 = keras.models.Sequential()
model1.add(keras.layers.Flatten(input_shape=[28,28]))
model1.add(keras.layers.Dense(300, activation='relu'))
model1.add(keras.layers.Dense(100, activation='relu'))
model1.add(keras.layers.Dense(10, activation='softmax'))
print(model1.summary())

keras.utils.plot_model(model1)   # visualise model using pydot
weights, biases = model1.layers[1].get_weights()   # generate weights and biases

# Compile and train model

model1.compile(loss='sparse_categorical_crossentropy',  # we use this loss as we have labels in our output, 
                                                        # omit sparse if we have probabilities instead, 
                                                        # use binary crossentropy if we use binary outputs.
              optimizer='sgd',
              metrics=['accuracy'])

model1_history = model1.fit(x_train, y_train, epoches=100,    # 100 epoch training 
                          validation_data=(x_valid, y_valid))
print(model1_history.params)

pd.DataFrame(model1_history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()

# Evaluating model performances

model1.evaluate(x_test, y_test)    # output: (loss, accuracy)

x_sample = x_test[:3]
y_proba = model1.predict(x_sample)
print(y_proba.round(2))    # get probabilities of the categories for the first 3 data: 10,3,2

y_pred = model1.predict_classes(x_sample)
print(y_pred)    # get the same results but get classes directly

print(np.array(class_names)[y_pred])    # get the name of the category instead of the class number

# Saving and Restoring model

model1.save('model1.h5')    # save model to path 

model1 = keras.models.load_model('model1.h5')    # load model from path 
model1.summary()    # check model to be correct version 


## SAVING BEST MODEL ONLY 
# add this line before fitting the model
checkpoint_cb = keras.callbacks.ModelCheckpoint('model1_best.h5', save_best_only=True)    # add checkpoint (save best)


## INCLUDE EARLY STOPPING IN TRAINING to prevent overfitting (good to always implement with high epochs)
# add this line before fitting the model
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,   # patience is the no. of epoches with no improvement after which training will be stopped
                                                  restore_best_weights=True)   # get the best result, might not be the most recent 

