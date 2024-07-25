import numpy as np
import tensorflow as tf
from keras.layers import Dense, Flatten, Conv2D, Input, Dropout, MaxPooling2D
from keras.models import Model

class AlexNet(Model):
    def __init__(self, **kwargs):
        super(AlexNet, self).__init__(**kwargs)
        #First Layer
        self.conv1 = (Conv2D(96, kernel_size=(11,11), strides=4, 
                        padding='valid', 
                        activation='relu',
                        kernel_initializer='he_normal',
                        input_shape=(224,224,3)))
        self.maxpool1 = (MaxPooling2D(pool_size=(3,3), strides=2, padding='valid'))
        #Second Layer
        self.conv2 = (Conv2D(256, kernel_size=(5,5), strides=1, 
                        padding='same', 
                        activation='relu',
                        kernel_initializer='he_normal'))
        self.maxpool2 = (MaxPooling2D(pool_size=(3,3), strides=2, padding='valid'))
        #Third Layer
        self.conv3 = (Conv2D(384, kernel_size=(3,3), strides=1, 
                        padding='same', 
                        activation='relu',
                        kernel_initializer='he_normal'))
        #Fourth Layer
        self.conv4 = (Conv2D(384, kernel_size=(3,3), strides=1, 
                        padding='same', 
                        activation='relu',
                        kernel_initializer='he_normal'))
        #Fifth Layer
        self.conv5 = (Conv2D(256, kernel_size=(3,3), strides=1, 
                        padding='same', 
                        activation='relu',
                        kernel_initializer='he_normal'))
        self.maxpool5 = (MaxPooling2D(pool_size=(3,3), strides=2, padding='valid'))
        #layer between convolutional and fully connect
        self.flatten = Flatten()
        #Fully connected layer
        self.dense1 = Dense(128, activation='relu')
        self.dropout = Dropout(0.5) #ideal value for drop rate is 0.2<x<0.5
        self.dense2 = Dense(64, activation='relu')
        self.dropout = Dropout(0.5)
        self.dense3 = Dense(1, activation='sigmoid')
    def call(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.maxpool5(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dropout(x)
        return self.dense3(x)

