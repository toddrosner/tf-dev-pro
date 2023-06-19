#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

print(tf.__version__)

# Load the Fashion MNIST dataset
fmnist = tf.keras.datasets.fashion_mnist

# Load the training and test split of the Fashion MNIST dataset
(training_images, training_labels), (test_images, test_labels) = fmnist.load_data()

# You can put between 0 to 59999 here
index = 0

# Set number of characters per row when printing
np.set_printoptions(linewidth=320)

# Print the label and image
print(f'LABEL: {training_labels[index]}')
print(f'\nIMAGE PIXEL ARRAY:\n {training_images[index]}')

# Visualize the image
plt.imshow(training_images[index])

# Normalize the pixel values of the train and test images
training_images  = training_images / 255.0
test_images = test_images / 255.0

# Build the classification model
model = tf.keras.models.Sequential([tf.keras.layers.Dense(1024, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# Compile model
model.compile(optimizer = tf.optimizers.Adam(),
    loss = 'sparse_categorical_crossentropy',
    metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)

# Evaluate the model on unseen data
model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)

print("Printing classifications...")
print(classifications[0])

print("Printing prediction...")
print(test_labels[0])
