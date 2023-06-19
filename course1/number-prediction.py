#!/usr/bin/env python

import numpy as np
import tensorflow as tf
from tensorflow import keras

print(tf.__version__)

'''
# this function is the basis for the prediction
def hw_function(x):
    y = (2 * x) - 1
    return y
'''

# build a simple sequential model
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# compile the model
model.compile(loss="mean_squared_error", optimizer="sgd")

# declare model inputs and outputs for training
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# train the model
model.fit(xs, ys, epochs=500)

# make a prediction
print("")
print("The prediction is...")
print(model.predict([5.0]))
