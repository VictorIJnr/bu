from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D
from keras.models import Model

from keras.datasets import mnist
from keras.callbacks import TensorBoard

from characterisation.classification import sklearnHelper as skh

import numpy as np

"""
Create the AutoEncoder model which will perform feature set reduction

factor dictates the amount of dimension reduction
"""                
def initAE(xTrain, xTest, factor=16):
    numFeatures = xTrain.shape[0]
    numUsers = xTrain.shape[1]

    inputLayer = Input(shape=(numFeatures,))
    encoder = initEncoder(xTrain.shape, factor)
    print("Encoding done")

    decoder = initDecoder(xTrain.shape, factor)
    print("Decoding done")

    print(encoder.input.shape)

    convAuto = decoder(encoder.output)
    convAuto = Model(encoder.input, convAuto)
    convAuto.compile(optimizer="adadelta", loss="binary_crossentropy")
    convAuto.save("autoencoder1D.h5")

    convAuto.fit(xTrain, xTrain,
                    epochs=50, batch_size=64,
                    shuffle=True,
                    validation_data=(xTest, xTest),
                    callbacks=[TensorBoard(log_dir="/tmp/autoencoder")])

    return convAuto


"""
Creates the network responsible for encoding data
"""
def initEncoder(trainShape, factor=16):
    numFeatures = trainShape[0]
    numUsers = trainShape[1]

    reducedSize = numFeatures / factor

    #Deep AutoEncoding, scaling down by factor over the course of 4 layers
    #16 is just a hyper-parameter for the number of time steps in the ConvNet
    inputLayer = Input(shape=(numFeatures, 1))
    xEnc = Conv1D(8, 4, activation="relu", padding="same")(inputLayer)
    xEnc = MaxPooling1D(int(factor / 2), padding="same")(xEnc)
    xEnc = Conv1D(4, 4, activation="relu", padding="same")(xEnc)
    xEnc = MaxPooling1D(int(factor / 2), padding="same")(xEnc)

    #Mapping a user's features to their reduced representation
    encoder = Model(inputLayer, xEnc)
    return encoder

"""
Creates the network responsible for decoding data
"""
def initDecoder(trainShape, factor=16):
    numFeatures = trainShape[0]
    numUsers = trainShape[1]

    reducedSize = int(numFeatures / factor)

    #4 is just calculated from performing the encoding reduction from 16
    inputLayer = Input(shape=(reducedSize, 4))
    xDec = Conv1D(4, 4, activation="relu", padding="same")(inputLayer)
    xDec = UpSampling1D(int(factor / 2))(xDec)
    xDec = Conv1D(8, 4, activation="relu")(xDec)
    xDec = UpSampling1D(int(factor / 2))(xDec)
    xDec = Conv1D(1, 4, activation="relu")(xDec)

    decoder = Model(inputLayer, xDec)
    return decoder

if __name__ == "__main__":
    xTrain, _, xTest, _ = skh.pullData()

    initAE(xTrain, xTest)