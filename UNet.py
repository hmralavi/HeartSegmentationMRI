"""
 Created By Hamid Alavi on 7/3/2019
"""
from keras.layers import Input, Conv2D, Concatenate, MaxPooling2D, UpSampling2D, Add, Activation, BatchNormalization, LeakyReLU
import keras.models as keras_models
import numpy as np


def create_UNet(input_shape, output_shape):
    filters = np.array([48, 52, 104, 208, 416])+10
    input_layer = Input(input_shape)

    c1, p1 = down_block(input_layer, filters[0])
    c2, p2 = down_block(p1, filters[1])
    c3, p3 = down_block(p2, filters[2])
    c4, p4 = down_block(p3, filters[3])
    
    bn = bottleneck_block(p4, filters[4])

    _, u1 = up_block(bn, c4, filters[3])
    cc2, u2 = up_block(u1, c3, filters[2])
    cc3, u3 = up_block(u2, c2, filters[1])
    cc4, _ = up_block(u3, c1, filters[0])

    s1 = Conv2D(output_shape[-1], kernel_size=(1, 1), strides=(1, 1), padding='same')(cc2)
    s2 = Conv2D(output_shape[-1], kernel_size=(1, 1), strides=(1, 1), padding='same')(cc3)
    s3 = Conv2D(output_shape[-1], kernel_size=(1, 1), strides=(1, 1), padding='same')(cc4)

    s12 = Add()([up_sampling(s1), s2])
    s123 = Add()([up_sampling(s12), s3])

    output_layer = Activation('softmax')(s123)
    return keras_models.Model(input_layer, output_layer)


def down_block(x, filters):
    c = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    c = BatchNormalization()(c)
    c = LeakyReLU()(c)
    c = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(c)
    c = BatchNormalization()(c)
    c = LeakyReLU()(c)
    p = down_sampling(c)
    return c, p


def bottleneck_block(x, filters):
    c = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    c = BatchNormalization()(c)
    c = LeakyReLU()(c)
    c = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(c)
    c = BatchNormalization()(c)
    c = LeakyReLU()(c)
    us = up_sampling(c)
    return us


def up_block(x, skip_layer, filters):
    concat = Concatenate()([x, skip_layer])
    c = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(concat)
    c = BatchNormalization()(c)
    c = LeakyReLU()(c)
    c = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(c)
    c = BatchNormalization()(c)
    c = LeakyReLU()(c)
    us = up_sampling(c)
    return c, us

def down_sampling(x):
    return MaxPooling2D((2, 2), (2, 2))(x)

def up_sampling(x):
    return UpSampling2D((2, 2))(x)

if __name__ == '__main__':
    UNetModel = create_UNet((480,840, 1), (480,840, 4))
    UNetModel.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
    UNetModel.summary()