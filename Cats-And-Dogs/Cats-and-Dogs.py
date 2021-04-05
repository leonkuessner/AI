import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
# Dense ends often with a dense layer, Flatten before ending with dense
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
from tensorflow.keras.callbacks import TensorBoard
import time

# NAME = "Cats-vs-dog-cnn-64x2-{}".format(int(time.time()))
# NAME = f"Cats-vs-dog-cnn-64x2-{int(time.time())}"


X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

# Normalize
X = np.array(X/255.0)
y = np.array(y)


dense_layers = [0, 1, 2]
layer_sizes = [32, 64, 128]
conv_layers = [1, 2, 3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = f"{conv_layer}-conv-{layer_size}-nodes-{dense_layer}-dense-{int(time.time())}"
            tensorboard = TensorBoard(log_dir=f'logs/{NAME}')
            model = Sequential()

            # (3,3) being window size
            model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for l in range(conv_layer-1):   # Weve alrdy got a conv layer
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            # Flatten because convolutional is 2D whereas Dense wants 1D
            model.add(Flatten())

            for l in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            model.compile(loss="binary_crossentropy",
                          optimizer="adam",
                          metrics=["accuracy"])

            # How many at once. Not too high
            model.fit(X, y, batch_size=32, epochs=10,
                      validation_split=0.3, callbacks=[tensorboard])
