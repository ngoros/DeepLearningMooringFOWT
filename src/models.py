import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Dropout
from src.arch import *

def generate_forward_model(arch_fun):
    P = keras.Input(shape=(2,))
    r = keras.Input(shape=(3,))
    #
    #layer1 = Dense(16, activation='relu', name='HL1')([P, r])
    #output_H = Dense(16, activation='relu', name='HL2')(layer1)
    output_H, arch_name = arch_fun([P, r])
    #
    u = Dense(28, activation='sigmoid')(output_H)
    # 
    model = keras.Model(inputs=[P, r], outputs=u, name='FORWARD_MODEL')
    print(model.summary())
    return model, arch_name

def generate_inverse_model(model_forward, arch_fun):
    u = keras.Input(shape=(28,))
    r = keras.Input(shape=(3,))
    #
    #layer1 = Dense(16, activation='relu', name='HL1')([P, r])
    #output_H = Dense(16, activation='relu', name='HL2')(layer1)
    output_H, name = arch_fun([u, r])
    print('-------------------------------------------')
    print(output_H)
    print('-------------------------------------------')
    #
    P = Dense(2, activation='sigmoid')(output_H)
    # 
    model_inverse = keras.Model(inputs=[u, r], outputs=P, name='INVERSE_MODEL')
    print(model_inverse.summary())
    print('------------------------------------------------------------------------')
    #----------------------------------------------------------------------
    #model_ustar = model_forward(name='FORWARD_MODEL')([P, r])
    #print(model_ustar.summary())

    #model = keras.Model(inputs=[u, r], outputs=model_ustar, name='MODEL_FOR_TRAINING')
    model = keras.Model(inputs=[u, r], outputs=model_forward([P, r]), name='MODEL_FOR_TRAINING')


    print(model.summary())
    print(model.layers[-1])
    return model, model_inverse
