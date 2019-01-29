from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D
from keras.models import Model

from keras.datasets import mnist
from keras.callbacks import TensorBoard

import numpy as np

from math import ceil

from characterisation.classification import sklearnHelper as skh

"""
Create an AutoEncoder model to learn feature set reduction.
This differs from the succeeding method as it specifies a dataset to use for
training, instead of straight-up passing in the training data.

It doesn't actually matter the number of folds we pass into split(),
as long as it's resonable.
"""
def initAE(dataset="serverfault", factor=16, mini=True):
    xTrain, _, xTest, _ = skh.split(dataset, mini=mini)

    return initAE(xTrain, xTest, factor)

"""
Create the AutoEncoder model which will perform feature set reduction
factor dictates the amount of dimension reduction

A factor of 16 takes the input down from 305-dimensional to being 80-dimensional
The output shape of the encoder is (?, 20, 4) 
"""                
def initAE(xTrain, xTest, factor=16):
    numFeatures = xTrain.shape[1]
    numUsers = xTrain.shape[0]
    testUsers = xTest.shape[0]

    inputLayer = Input(shape=(numFeatures,))
    encoder = initEncoder(xTrain.shape, factor)
    print("Encoding done")

    decoder = initDecoder(xTrain.shape, factor)
    print("Decoding done")

    #1 is the number of channels
    xTrain = np.reshape(xTrain, (numUsers, numFeatures, 1))
    xTest = np.reshape(xTest, (testUsers, numFeatures, 1))

    print(xTrain.shape)
    print(xTest.shape)

    print(f"Encoder input shape {encoder.input.shape}")
    print(f"Encoder output shape {encoder.output.shape}")

    convAuto = decoder(encoder.output)
    convAuto = Model(encoder.input, convAuto)
    convAuto.compile(optimizer="adadelta", loss="binary_crossentropy")

    print(f"Conv Auto input shape {convAuto.input.shape}")
    print(f"Conv Auto output shape {convAuto.output.shape}")

    convAuto.fit(xTrain, xTrain,
                    epochs=10, batch_size=128,
                    shuffle=True,
                    validation_data=(xTest, xTest),
                    callbacks=[TensorBoard(log_dir="/tmp/autoencoder")])

    convAuto.save("autoencoder1D.h5")
    return convAuto


"""
Creates the network responsible for encoding data
Data is scaled down by a given factor
This factor must be a square number given the current implementation
"""
def initEncoder(trainShape, factor=16):
    numFeatures = trainShape[1]

    #Deep AutoEncoding, scaling down by factor over the course of 4 layers
    #16 is just a hyper-parameter for the number of time steps in the ConvNet
    inputLayer = Input(shape=(numFeatures, 1))
    xEnc = Conv1D(8, 4, activation="relu", padding="same")(inputLayer)
    print(f"\n1st Conv, xEnc Shape: {xEnc.shape}")
    xEnc = MaxPooling1D(int(factor ** 0.5), padding="same")(xEnc)
    print(f"1st Max Pooling, xEnc Shape: {xEnc.shape}")
    xEnc = Conv1D(4, 4, activation="relu", padding="same")(xEnc)
    print(f"2nd Conv, xEnc Shape: {xEnc.shape}")
    xEnc = MaxPooling1D(int(factor ** 0.5), padding="same")(xEnc)
    print(f"2nd Max Pooling, xEnc Shape: {xEnc.shape}")

    print(f"Final xEnc Shape: {xEnc.shape}")
    #Mapping a user's features to their reduced representation
    encoder = Model(inputLayer, xEnc)
    print(f"Encoder output Shape: {encoder.output.shape}\n")
    return encoder

"""
Creates the network responsible for decoding data
Scales data up by a given factor
Which must be a square number given the current implementation
"""
def initDecoder(trainShape, factor=16):
    numFeatures = trainShape[1]
    reducedSize = ceil(numFeatures / factor)

    #4 is just calculated from performing the encoding reduction from 16
    inputLayer = Input(shape=(reducedSize, 4))
    xDec = Conv1D(4, 4, activation="relu", padding="same")(inputLayer)
    xDec = UpSampling1D(int(factor ** 0.5))(xDec)
    xDec = Conv1D(8, 4, activation="relu")(xDec)
    xDec = UpSampling1D(int(factor ** 0.5))(xDec)
    xDec = Conv1D(1, 4, activation="relu")(xDec)

    decoder = Model(inputLayer, xDec)
    return decoder

if __name__ == "__main__":
    xTrain, _, xTest, _ = skh.pullData(fullData=True)

    initAE(xTrain, xTest)