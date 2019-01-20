from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D
from keras.models import Model

from keras.datasets import mnist
from keras.callbacks import TensorBoard

import numpy as np

"""
Create the AutoEncoder model which will perform feature set reduction

factor dictates the amount of dimension reduction
"""                
def initAE(xTrain, xTest, factor=4):
    #Reduce from 12 dimensions to 3
    (x_train, _), (x_test, _) = mnist.load_data()

    #This was only for MNIST
    """ x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1)) 
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1)) """

    numFeatures = xTrain.shape[0]

    inputText = Input(shape=(numFeatures,))
    x = Conv1D(3, 3, activation="relu", padding="same")(inputText)
    encoded = MaxPooling1D(2, padding="same")(x)

    print("Encoding done")

    xDec = UpSampling1D(2)(encoded)
    decoded = Conv1D(1, 3, activation="sigmoid", padding="same")(xDec)
    print("Decoding done")

    convAuto = Model(inputText, decoded)
    convAuto.compile(optimizer="adadelta", loss="binary_crossentropy")
    convAuto.save("autoencoder1D.h5")

    convAuto.fit(x_train, x_train,
                    epochs=50, batch_size=128,
                    shuffle=True,
                    validation_data=(x_test, x_test),
                    callbacks=[TensorBoard(log_dir="/tmp/autoencoder")])

    return convAuto


"""
Encodes input
"""
def autoEncode():
    pass