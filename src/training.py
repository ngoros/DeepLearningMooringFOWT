import os
import numpy as np
import pandas as pd
import json
import tensorflow as tf
from tensorflow.keras import optimizers, models, utils
from src.models import generate_forward_model, generate_inverse_model
from keras.regularizers import l2

def save_weights(model, arch_name):
    # Guardar configuración JSON en el disco
    json_config = model.to_json()
    with open(os.path.join('weights', 'model_config_'+arch_name+'.json'), 'w') as json_file:
        json_file.write(json_config)
    # Guardar pesos en el disco
    model.save_weights(os.path.join('weights', arch_name+'.h5'))

def load_weights_and_model(arch_name):
    # Recargue el modelo de los 2 archivos que guardamos
    with open(os.path.join('weights', 'model_config_'+arch_name+'.json'), 'r') as json_file:
        json_config = json_file.read()
    
    model = models.model_from_json(json_config)
    model.load_weights(os.path.join('weights', arch_name+'.h5'))
    return model

def save_weights2(model, arch_name):
    weights = []
    for layer in range(0, len(model.layers)):
        weights.append(model.layers[layer].get_weights())

    df = pd.DataFrame(weights)
    df.to_csv(os.path.join('weights', arch_name+'.csv'), index=False)
    #with open(os.path.join('weights',arch_name+'.txt'), 'wb') as f:
    #    np.savetxt(f, weights)

def set_weights2(model, arch_name):

    w = pd.read_csv(os.path.join('weights'), arch_name+'.csv', header=None)
    #with open(os.path.join('weights',arch_name+'.npy'), 'r') as f:
    #    weights = np.load(f)

    print(weights)
    for layer in range(0, len(model.layers)):
        model.layers[layer].set_weights(weights[layer])


def run_training(train_forward, 
                 u_train, u_validation, 
                 r_train, r_validation, 
                 P_train, P_validation,
                 arch_fun_f, arch_fun_i):

    tf.random.set_seed(1234)
    if train_forward:
        model_forward, arch_name = generate_forward_model(arch_fun_f)
        opt = optimizers.Adam(learning_rate = 0.005)
        model_forward.compile(loss = 'mse', optimizer = opt)    
        model_forward.fit(x = [P_train, r_train],
          y = u_train,
          batch_size = 512,
          epochs = 1500,
          shuffle = True,
          validation_data = ([P_validation, r_validation], u_validation))
        
        save_weights(model_forward, arch_name)
        #print(P_validation, r_validation)
        #print(model_forward.predict([P_validation, r_validation]))
        return model_forward, [], arch_fun_f.__name__, []

    else:
        model_forward = load_weights_and_model(arch_fun_f.__name__)
        model_forward.trainable = False
        print(model_forward.summary())
        #print(model_forward.layers[3].get_weights())
        #print(P_validation, r_validation)
        #print(model_forward.predict([P_validation, r_validation]))

        model, model_inverse = generate_inverse_model(model_forward, arch_fun_i)
        opt = optimizers.Adam(learning_rate = 0.005)
        model.compile(loss = 'mse', optimizer = opt)    
        model.fit(x = [u_train, r_train],
          y = u_train,
          batch_size = 512,
          epochs = 1000,
          shuffle = True,
          validation_data = ([u_validation, r_validation], u_validation))
        
        return model, model_inverse, [], arch_fun_i.__name__
        #utils.plot_model(model, "model.png", show_shapes=True)


        
