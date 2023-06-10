from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Dropout
from keras.regularizers import l2

def arch1_f(inputs):
    name = 'arch1_f'
    layer_concat = keras.layers.Concatenate(axis=-1)(inputs)
    layer1 = Dense(16, activation='relu', name='FHL1')(layer_concat)
    output = Dense(8, activation='relu', name='FHL2')(layer1)
    return output, name

def arch2_f(inputs):
    name = 'arch2_f'
    layer_concat = keras.layers.Concatenate(axis=-1)(inputs)
    layer1 = Dense(24, activation='relu', name='FHL1')(layer_concat)
    layer2 = Dense(16, activation='relu', name='FHL2')(layer1)
    output = Dense(8, activation='relu', name='FHL3')(layer2)
    return output, name

def arch3_f(inputs):
    name = 'arch3_f'
    layer_concat = keras.layers.Concatenate(axis=-1)(inputs)
    layer1 = Dense(24, activation='relu', name='FHL1')(layer_concat)
    layer2 = Dense(16, activation='relu', name='FHL2')(layer1)
    layer3 = Dense(16, activation='relu', name='FHL3')(layer2)
    output = Dense(8, activation='relu', name='FHL4')(layer3)
    return output, name

def arch4_f(inputs):
    name = 'arch4_f'
    layer_concat = keras.layers.Concatenate(axis=-1)(inputs)
    layer1 = Dense(24, activation='relu', name='FHL1')(layer_concat)
    layer2 = Dense(16, activation='relu', name='FHL2')(layer1)
    layer3 = Dense(16, activation='relu', name='FHL3')(layer2)
    layer4 = Dense(16, activation='relu', name='FHL4')(layer3)
    output = Dense(8, activation='relu', name='FHL5')(layer4)
    return output, name

def arch5_f(inputs):
    name = 'arch5_f'
    layer_concat = keras.layers.Concatenate(axis=-1)(inputs)
    layer1 = Dense(8, activation='relu', name='FHL1')(layer_concat)
    layer2 = Dense(16, activation='relu', name='FHL2')(layer1)
    layer3 = Dense(16, activation='relu', name='FHL3')(layer2)
    layer4 = Dense(24, activation='relu', name='FHL4')(layer3)
    output = Dense(24, activation='relu', name='FHL5')(layer4)
    return output, name

def arch6_f(inputs):
    name = 'arch6_f'
    layer_concat = keras.layers.Concatenate(axis=-1)(inputs)
    layer1 = Dense(6, activation='relu', name='FHL1')(layer_concat)
    layer2 = Dense(12, activation='relu', name='FHL2')(layer1)
    layer3 = Dense(18, activation='relu', name='FHL3')(layer2)
    layer4 = Dense(24, activation='relu', name='FHL4')(layer3)
    output = Dense(30, activation='relu', name='FHL5')(layer4)
    return output, name

def arch7_f(inputs):
    name = 'arch7_f'
    layer_concat = keras.layers.Concatenate(axis=-1)(inputs)
    layer1 = Dense(8, activation='relu', name='FHL1')(layer_concat)
    layer2 = Dense(16, activation='relu', name='FHL2')(layer1)
    layer3 = Dense(32, activation='relu', name='FHL3')(layer2)
    layer4 = Dense(48, activation='relu', name='FHL4')(layer3)
    output = Dense(40, activation='relu', name='FHL5')(layer4)
    return output, name

def arch8_f(inputs):
    name = 'arch8_f'
    layer_concat = keras.layers.Concatenate(axis=-1)(inputs)
    layer1 = Dense(8, activation='relu', name='FHL1')(layer_concat)
    layer2 = Dense(16, activation='relu', name='FHL2')(layer1)
    layer3 = Dense(16, activation='relu', name='FHL3')(layer2)
    layer4 = Dense(24, activation='relu', name='FHL4')(layer3)
    layer5 = Dense(24, activation='relu', name='FHL5')(layer4)
    output = Dense(24, activation='relu', name='FHL6')(layer5)
    return output, name

def arch9_f(inputs):
    name = 'arch9_f'
    layer_concat = keras.layers.Concatenate(axis=-1)(inputs)
    layer1 = Dense(6, activation='relu', name='FHL1')(layer_concat)
    layer2 = Dense(10, activation='relu', name='FHL2')(layer1)
    layer3 = Dense(14, activation='relu', name='FHL3')(layer2)
    layer4 = Dense(18, activation='relu', name='FHL4')(layer3)
    layer5 = Dense(22, activation='relu', name='FHL5')(layer4)
    output = Dense(26, activation='relu', name='FHL6')(layer5)
    return output, name

def arch10_f(inputs):
    name = 'arch10_f'
    layer_concat = keras.layers.Concatenate(axis=-1)(inputs)
    layer1 = Dense(8, activation='relu', name='FHL1')(layer_concat)
    layer2 = Dense(16, activation='relu', name='FHL2')(layer1)
    layer3 = Dense(24, activation='relu', name='FHL3')(layer2)
    layer4 = Dense(32, activation='relu', name='FHL4')(layer3)
    layer5 = Dense(40, activation='relu', name='FHL5')(layer4)
    output = Dense(40, activation='relu', name='FHL6')(layer5)
    return output, name

def arch11_f(inputs):
    name = 'arch11_f'
    layer_concat = keras.layers.Concatenate(axis=-1)(inputs)
    layer1 = Dense(8, activation='relu', name='FHL1')(layer_concat)
    layer2 = Dense(8, activation='relu', name='FHL2')(layer1)
    layer3 = Dense(12, activation='relu', name='FHL3')(layer2)
    layer4 = Dense(12, activation='relu', name='FHL4')(layer3)
    layer5 = Dense(22, activation='relu', name='FHL5')(layer4)
    output = Dense(22, activation='relu', name='FHL6')(layer5)
    return output, name

def arch12_f(inputs):
    name = 'arch12_f'
    layer_concat = keras.layers.Concatenate(axis=-1)(inputs)
    layer1 = Dense(8, activation='relu', name='FHL1')(layer_concat)
    layer2 = Dense(8, activation='relu', name='FHL2')(layer1)
    layer3 = Dense(16, activation='relu', name='FHL3')(layer2)
    layer4 = Dense(16, activation='relu', name='FHL4')(layer3)
    layer5 = Dense(24, activation='relu', name='FHL5')(layer4)
    layer6 = Dense(24, activation='relu', name='FHL6')(layer5)
    output = Dense(28, activation='relu', name='FHL7')(layer6)
    return output, name

def arch13_f(inputs):
    name = 'arch13_f'
    layer_concat = keras.layers.Concatenate(axis=-1)(inputs)
    layer1 = Dense(6, activation='relu', name='FHL1')(layer_concat)
    layer2 = Dense(10, activation='relu', name='FHL2')(layer1)
    layer3 = Dense(14, activation='relu', name='FHL3')(layer2)
    layer4 = Dense(18, activation='relu', name='FHL4')(layer3)
    layer5 = Dense(22, activation='relu', name='FHL5')(layer4)
    layer6 = Dense(26, activation='relu', name='FHL6')(layer5)
    output = Dense(30, activation='relu', name='FHL7')(layer6)
    return output, name

def arch14_f(inputs):
    name = 'arch14_f'
    layer_concat = keras.layers.Concatenate(axis=-1)(inputs)
    layer1 = Dense(10, activation='relu', name='FHL1')(layer_concat)
    layer2 = Dense(20, activation='relu', name='FHL2')(layer1)
    layer3 = Dense(30, activation='relu', name='FHL3')(layer2)
    layer4 = Dense(40, activation='relu', name='FHL4')(layer3)
    output = Dense(50, activation='relu', name='FHL5')(layer4)
    return output, name

def arch15_f(inputs):
    name = 'arch15_f'
    layer_concat = keras.layers.Concatenate(axis=-1)(inputs)
    layer1 = Dense(10, activation='relu', name='FHL1')(layer_concat)
    layer2 = Dense(25, activation='relu', name='FHL2')(layer1)
    layer3 = Dense(40, activation='relu', name='FHL3')(layer2)
    layer4 = Dense(55, activation='relu', name='FHL4')(layer3)
    output = Dense(50, activation='relu', name='FHL5')(layer4)
    return output, name

def arch16_f(inputs):
    name = 'arch16_f'
    layer_concat = keras.layers.Concatenate(axis=-1)(inputs)
    layer1 = Dense(10, activation='relu', name='FHL1')(layer_concat)
    layer2 = Dense(10, activation='relu', name='FHL2')(layer1)
    layer3 = Dense(20, activation='relu', name='FHL3')(layer2)
    layer4 = Dense(20, activation='relu', name='FHL4')(layer3)
    output = Dense(30, activation='relu', name='FHL5')(layer4)
    return output, name

def arch17_f(inputs):
    name = 'arch17_f'
    layer_concat = keras.layers.Concatenate(axis=-1)(inputs)
    layer1 = Dense(12, activation='relu', name='FHL1')(layer_concat)
    layer2 = Dense(24, activation='relu', name='FHL2')(layer1)
    layer3 = Dense(40, activation='relu', name='FHL3')(layer2)
    layer4 = Dense(60, activation='relu', name='FHL4')(layer3)
    output = Dense(48, activation='relu', name='FHL5')(layer4)
    return output, name

def arch18_f(inputs):
    name = 'arch18_f'
    layer_concat = keras.layers.Concatenate(axis=-1)(inputs)
    layer1 = Dense(10, activation='relu', name='FHL1')(layer_concat)
    layer2 = Dense(20, activation='relu', name='FHL2')(layer1)
    layer3 = Dense(30, activation='relu', name='FHL3')(layer2)
    layer4 = Dense(40, activation='relu', name='FHL4')(layer3)
    layer5 = Dense(50, activation='relu', name='FHL5')(layer4)
    output = Dense(50, activation='relu', name='FHL6')(layer5)
    return output, name

def arch19_f(inputs):
    name = 'arch19_f'
    layer_concat = keras.layers.Concatenate(axis=-1)(inputs)
    layer1 = Dense(7, activation='relu', name='FHL1')(layer_concat)
    layer2 = Dense(10, activation='relu', name='FHL2')(layer1)
    layer3 = Dense(15, activation='relu', name='FHL3')(layer2)
    layer4 = Dense(20, activation='relu', name='FHL4')(layer3)
    layer5 = Dense(30, activation='relu', name='FHL5')(layer4)
    output = Dense(35, activation='relu', name='FHL6')(layer5)
    return output, name

def arch20_f(inputs):
    name = 'arch20_f'
    layer_concat = keras.layers.Concatenate(axis=-1)(inputs)
    layer1 = Dense(12, activation='relu', name='FHL1')(layer_concat)
    layer2 = Dense(24, activation='relu', name='FHL2')(layer1)
    layer3 = Dense(48, activation='relu', name='FHL3')(layer2)
    layer4 = Dense(48, activation='relu', name='FHL4')(layer3)
    layer5 = Dense(48, activation='relu', name='FHL5')(layer4)
    output = Dense(40, activation='relu', name='FHL6')(layer5)
    return output, name

def arch21_f(inputs):
    name = 'arch21_f'
    layer_concat = keras.layers.Concatenate(axis=-1)(inputs)
    layer1 = Dense(8, activation='relu', name='FHL1')(layer_concat)
    layer2 = Dense(25, activation='relu', name='FHL2')(layer1)
    layer3 = Dense(45, activation='relu', name='FHL3')(layer2)
    layer4 = Dense(60, activation='relu', name='FHL4')(layer3)
    output = Dense(45, activation='relu', name='FHL5')(layer4)
    return output, name

def arch22_f(inputs):
    name = 'arch22_f'
    layer_concat = keras.layers.Concatenate(axis=-1)(inputs)
    layer1 = Dense(8, activation='relu', name='FHL1')(layer_concat)
    layer2 = Dense(20, activation='relu', name='FHL2')(layer1)
    layer3 = Dense(36, activation='relu', name='FHL3')(layer2)
    layer4 = Dense(48, activation='relu', name='FHL4')(layer3)
    layer5 = Dense(64, activation='relu', name='FHL5')(layer4)
    output = Dense(50, activation='relu', name='FHL6')(layer5)
    return output, name

def arch23_f(inputs):
    name = 'arch23_f'
    layer_concat = keras.layers.Concatenate(axis=-1)(inputs)
    layer1 = Dense(12, activation='relu', name='FHL1')(layer_concat)
    layer2 = Dense(24, activation='relu', name='FHL2')(layer1)
    layer3 = Dense(48, activation='relu', name='FHL3')(layer2)
    layer4 = Dense(48, activation='relu', name='FHL4')(layer3)
    layer5 = Dense(48, activation='relu', name='FHL5')(layer4)
    output = Dense(40, activation='relu', name='FHL6')(layer5)
    return output, name

def arch24_f(inputs):
    name = 'arch24_f'
    layer_concat = keras.layers.Concatenate(axis=-1)(inputs)
    layer1 = Dense(7, activation='relu', name='FHL1')(layer_concat)
    layer2 = Dense(12, activation='relu', name='FHL2')(layer1)
    layer3 = Dense(25, activation='relu', name='FHL3')(layer2)
    layer4 = Dense(25, activation='relu', name='FHL4')(layer3)
    output = Dense(30, activation='relu', name='FHL5')(layer4)
    return output, name

def arch25_f(inputs):
    name = 'arch25_f'
    layer_concat = keras.layers.Concatenate(axis=-1)(inputs)
    layer1 = Dense(8, activation='relu', name='FHL1')(layer_concat)
    layer2 = Dense(24, activation='relu', name='FHL2')(layer1)
    layer3 = Dense(38, activation='relu', name='FHL3')(layer2)
    layer4 = Dense(52, activation='relu', name='FHL4')(layer3)
    layer5 = Dense(70, activation='relu', name='FHL5')(layer4)
    output = Dense(70, activation='relu', name='FHL6')(layer5)
    return output, name

def arch26_f(inputs):
    name = 'arch26_f'
    layer_concat = keras.layers.Concatenate(axis=-1)(inputs)
    layer1 = Dense(12, activation='relu', name='FHL1')(layer_concat)
    layer2 = Dense(40, activation='relu', name='FHL2')(layer1)
    layer3 = Dense(60, activation='relu', name='FHL3')(layer2)
    layer4 = Dense(80, activation='relu', name='FHL4')(layer3)
    layer5 = Dense(80, activation='relu', name='FHL5')(layer4)
    output = Dense(50, activation='relu', name='FHL6')(layer5)
    return output, name

def arch27_f(inputs):
    name = 'arch27_f'
    layer_concat = keras.layers.Concatenate(axis=-1)(inputs)
    layer1 = Dense(8, activation='relu', name='FHL1')(layer_concat)
    layer2 = Dense(24, activation='relu', name='FHL2')(layer1)
    layer3 = Dense(42, activation='relu', name='FHL3')(layer2)
    layer4 = Dense(60, activation='relu', name='FHL4')(layer3)
    layer5 = Dense(80, activation='relu', name='FHL5')(layer4)
    output = Dense(80, activation='relu', name='FHL6')(layer5)
    return output, name

def arch28_f(inputs):
    name = 'arch28_f'
    layer_concat = keras.layers.Concatenate(axis=-1)(inputs)
    layer1 = Dense(10, activation='relu', name='FHL1')(layer_concat)
    layer2 = Dense(25, activation='relu', name='FHL2')(layer1)
    layer3 = Dense(50, activation='relu', name='FHL3')(layer2)
    layer4 = Dense(75, activation='relu', name='FHL4')(layer3)
    layer5 = Dense(90, activation='relu', name='FHL5')(layer4)
    output = Dense(80, activation='relu', name='FHL6')(layer5)
    return output, name

def arch29_f(inputs):
    name = 'arch29_f'
    layer_concat = keras.layers.Concatenate(axis=-1)(inputs)
    layer1 = Dense(7, activation='relu', name='FHL1')(layer_concat)
    layer2 = Dense(15, activation='relu', name='FHL2')(layer1)
    layer3 = Dense(30, activation='relu', name='FHL3')(layer2)
    layer4 = Dense(60, activation='relu', name='FHL4')(layer3)
    layer5 = Dense(85, activation='relu', name='FHL5')(layer4)
    output = Dense(85, activation='relu', name='FHL6')(layer5)
    return output, name

def arch30_f(inputs):
    name = 'arch30_f'
    layer_concat = keras.layers.Concatenate(axis=-1)(inputs)
    layer1 = Dense(10, activation='relu', name='FHL1')(layer_concat)
    layer2 = Dense(30, activation='relu', name='FHL2')(layer1)
    layer3 = Dense(50, activation='relu', name='FHL3')(layer2)
    layer4 = Dense(70, activation='relu', name='FHL4')(layer3)
    layer5 = Dense(90, activation='relu', name='FHL5')(layer4)
    output = Dense(100, activation='relu', name='FHL6')(layer5)
    return output, name

def arch31_f(inputs):
    name = 'arch31_f'
    layer_concat = keras.layers.Concatenate(axis=-1)(inputs)
    layer1 = Dense(10, activation='relu', name='FHL1')(layer_concat)
    layer2 = Dense(30, activation='relu', name='FHL2')(layer1)
    layer3 = Dense(50, activation='relu', name='FHL3')(layer2)
    layer4 = Dense(70, activation='relu', name='FHL4')(layer3)
    layer5 = Dense(90, activation='relu', name='FHL5')(layer4)
    output = Dense(80, activation='relu', name='FHL6')(layer5)
    return output, name

def arch32_f(inputs):
    name = 'arch32_f'
    layer_concat = keras.layers.Concatenate(axis=-1)(inputs)
    layer1 = Dense(7, activation='relu', name='FHL1')(layer_concat)
    layer2 = Dense(18, activation='relu', name='FHL2')(layer1)
    layer3 = Dense(36, activation='relu', name='FHL3')(layer2)
    layer4 = Dense(60, activation='relu', name='FHL4')(layer3)
    layer5 = Dense(90, activation='relu', name='FHL5')(layer4)
    output = Dense(85, activation='relu', name='FHL6')(layer5)
    return output, name

def arch33_f(inputs):
    name = 'arch33_f'
    layer_concat = keras.layers.Concatenate(axis=-1)(inputs)
    layer1 = Dense(11, activation='relu', name='FHL1')(layer_concat)
    layer2 = Dense(26, activation='relu', name='FHL2')(layer1)
    layer3 = Dense(54, activation='relu', name='FHL3')(layer2)
    layer4 = Dense(80, activation='relu', name='FHL4')(layer3)
    layer5 = Dense(95, activation='relu', name='FHL5')(layer4)
    output = Dense(80, activation='relu', name='FHL6')(layer5)
    return output, name

def arch34_f(inputs):
    name = 'arch34_f'
    layer_concat = keras.layers.Concatenate(axis=-1)(inputs)
    layer1 = Dense(9, activation='relu', name='FHL1')(layer_concat)
    layer2 = Dense(18, activation='relu', name='FHL2')(layer1)
    layer3 = Dense(35, activation='relu', name='FHL3')(layer2)
    layer4 = Dense(70, activation='relu', name='FHL4')(layer3)
    layer5 = Dense(95, activation='relu', name='FHL5')(layer4)
    output = Dense(80, activation='relu', name='FHL6')(layer5)
    return output, name

def arch35_f(inputs):
    name = 'arch35_f'
    layer_concat = keras.layers.Concatenate(axis=-1)(inputs)
    layer1 = Dense(7, activation='relu', name='FHL1')(layer_concat)
    layerD1 = Dropout(0.2, name='FHDL1')(layer1)
    layer2 = Dense(18, activation='relu', name='FHL2')(layerD1)
    layerD2 = Dropout(0.2, name='FHDL2')(layer2)
    layer3 = Dense(36, activation='relu', name='FHL3')(layerD2)
    layerD3 = Dropout(0.2, name='FHDL3')(layer3)
    layer4 = Dense(60, activation='relu', name='FHL4')(layerD3)
    layerD4 = Dropout(0.2, name='FHDL4')(layer4)
    layer5 = Dense(90, activation='relu', name='FHL5')(layerD4)
    layerD5 = Dropout(0.2, name='FHDL5')(layer5)
    output = Dense(85, activation='relu', name='FHL6')(layerD5)
    return output, name

def arch36_f(inputs):
    name = 'arch36_f'
    layer_concat = keras.layers.Concatenate(axis=-1)(inputs)
    layer1 = Dense(7, activation='relu', name='FHL1')(layer_concat)
    layer2 = Dense(18, activation='relu', name='FHL2')(layer1)
    layer3 = Dense(36, activation='relu', name='FHL3')(layer2)
    layer4 = Dense(60, activation='relu', name='FHL4')(layer3)
    layer5 = Dense(90, activation='relu', name='FHL5')(layer4)
    output = Dense(85, activation='relu', name='FHL6')(layer5)
    return output, name

def arch37_f(inputs):
    name = 'arch37_f'
    layer_concat = keras.layers.Concatenate(axis=-1)(inputs)
    layer1 = Dense(7, activation='relu', name='FHL1')(layer_concat)
    layer2 = Dense(18, activation='relu', name='FHL2')(layer1)
    layer3 = Dense(36, activation='relu', name='FHL3')(layer2)
    layerD3 = Dropout(0.2, name='FHDL3')(layer3)
    layer4 = Dense(60, activation='relu', name='FHL4')(layerD3)
    layerD4 = Dropout(0.2, name='FHDL4')(layer4)
    layer5 = Dense(90, activation='relu', name='FHL5')(layerD4)
    layerD5 = Dropout(0.2, name='FHDL5')(layer5)
    output = Dense(85, activation='relu', name='FHL6')(layerD5)
    return output, name

def arch38_f(inputs):
    name = 'arch38_f'
    layer_concat = keras.layers.Concatenate(axis=-1)(inputs)
    layer1 = Dense(7, activation='relu', name='FHL1')(layer_concat)
    layer2 = Dense(18, activation='relu', name='FHL2')(layer1)
    layer3 = Dense(36, activation='relu', name='FHL3')(layer2)
    layer4 = Dense(60, activation='relu', name='FHL4')(layer3)
    layerD4 = Dropout(0.1, name='FHDL4')(layer4)
    layer5 = Dense(90, activation='relu', name='FHL5')(layerD4)
    layerD5 = Dropout(0.2, name='FHDL5')(layer5)
    output = Dense(85, activation='relu', name='FHL6')(layerD5)
    return output, name

def arch39_f(inputs):
    name = 'arch39_f'
    layer_concat = keras.layers.Concatenate(axis=-1)(inputs)
    layer1 = Dense(7, activation='relu', name='FHL1')(layer_concat)
    layer2 = Dense(18, activation='relu', name='FHL2')(layer1)
    layer3 = Dense(36, activation='relu', name='FHL3')(layer2)
    layer4 = Dense(60, activation='relu', name='FHL4')(layer3)
    layer5 = Dense(90, activation='relu', name='FHL5')(layer4)
    layer5D = Dropout(0.2, name='FHDL5')(layer5)
    layer6 = Dense(85, activation='relu', name='FHL6')(layer5D)
    output = Dropout(0.1, name='FHDL6')(layer6)
    return output, name

def arch40_f(inputs):
    name = 'arch40_f'
    layer_concat = keras.layers.Concatenate(axis=-1)(inputs)
    layer1 = Dense(7, activation='relu', name='FHL1')(layer_concat)
    layer2 = Dense(18, activation='relu', name='FHL2')(layer1)
    layer3 = Dense(36, activation='relu', name='FHL3')(layer2)
    layer4 = Dense(60, activation='relu', name='FHL4')(layer3)
    layer5 = Dense(90, activation='relu', name='FHL5')(layer4)
    layer5D = Dropout(0.1, name='FHDL5')(layer5)
    layer6 = Dense(85, activation='relu', name='FHL6')(layer5D)
    output = Dropout(0.05, name='FHDL6')(layer6)
    return output, name

def arch41_f(inputs):
    name = 'arch41_f'
    layer_concat = keras.layers.Concatenate(axis=-1)(inputs)
    layer1 = Dense(7, activation='relu', name='FHL1')(layer_concat)
    layer2 = Dense(18, activation='relu', name='FHL2')(layer1)
    layer3 = Dense(36, activation='relu', name='FHL3')(layer2)
    layer4 = Dense(60, activation='relu', name='FHL4')(layer3)
    layer5 = Dense(90, activation='relu', name='FHL5')(layer4)
    layer5D = Dropout(0.05, name='FHDL5')(layer5)
    layer6 = Dense(85, activation='relu', name='FHL6')(layer5D)
    output = Dropout(0.05, name='FHDL6')(layer6)
    return output, name

def arch42_f(inputs):
    name = 'arch42_f'
    layer_concat = keras.layers.Concatenate(axis=-1)(inputs)
    layer1 = Dense(7, activation='relu', name='FHL1')(layer_concat)
    layer2 = Dense(18, activation='relu', name='FHL2')(layer1)
    layer3 = Dense(36, activation='relu', name='FHL3')(layer2)
    layer4 = Dense(60, activation='relu', name='FHL4')(layer3)
    layer5 = Dense(90, activation='relu', name='FHL5')(layer4)
    layer6 = Dense(85, activation='relu', name='FHL6')(layer5)
    output = Dropout(0.05, name='FHDL6')(layer6)
    return output, name

#########################################################################################################################

def arch1_i(inputs):
    name = 'arch1_i'
    layer_concat = keras.layers.Concatenate(axis=-1)(inputs)
    layer1 = Dense(17, activation='relu', name='IHL1')(layer_concat)
    layerD1 = Dropout(0.2, name='FHDL1')(layer1)
    output = Dense(7, activation='relu', name='IHL2')(layerD1)
    return output, name

def arch2_i(inputs):
    name = 'arch2_i'
    layer_concat = keras.layers.Concatenate(axis=-1)(inputs)
    layer1 = Dense(25, activation='relu', name='IHL1')(layer_concat)
    layer2 = Dense(20, activation='relu', name='IHL2')(layer1)
    layer3 = Dense(16, activation='relu', name='IHL3')(layer2)
    output = Dense(12, activation='relu', name='IHL4')(layer3)
    return output, name

def arch3_i(inputs):
    name = 'arch3_i'
    layer_concat = keras.layers.Concatenate(axis=-1)(inputs)
    layer1 = Dense(25, activation='relu', name='IHL1')(layer_concat)
    layer2 = Dense(20, activation='relu', name='IHL2')(layer1)
    layer3 = Dense(16, activation='relu', name='IHL3')(layer2)
    layer4 = Dense(16, activation='relu', name='IHL4')(layer3)
    output = Dense(12, activation='relu', name='IHL5')(layer4)
    return output, name

def arch4_i(inputs):
    name = 'arch4_i'
    layer_concat = keras.layers.Concatenate(axis=-1)(inputs)
    layer1 = Dense(46, activation='relu', name='IHL1')(layer_concat)
    layer2 = Dense(62, activation='relu', name='IHL2')(layer1)
    layer3 = Dense(48, activation='relu', name='IHL3')(layer2)
    layer4 = Dense(24, activation='relu', name='IHL4')(layer3)
    layer5 = Dense(12, activation='relu', name='IHL5')(layer4)
    output = Dense(6, activation='relu', name='IHL6')(layer5)
    return output, name

def arch5_i(inputs):
    name = 'arch5_i'
    layer_concat = keras.layers.Concatenate(axis=-1)(inputs)
    layer1 = Dense(30, activation='relu', name='IHL1')(layer_concat)
    layer2 = Dense(25, activation='relu', name='IHL2')(layer1)
    layer3 = Dense(20, activation='relu', name='IHL3')(layer2)
    layer4 = Dense(15, activation='relu', name='IHL4')(layer3)
    output = Dense(10, activation='relu', name='IHL5')(layer4)
    return output, name

def arch6_i(inputs):
    name = 'arch6_i'
    layer_concat = keras.layers.Concatenate(axis=-1)(inputs)
    layer1 = Dense(46, activation='relu', name='IHL1')(layer_concat)
    layer2 = Dense(62, activation='relu', name='IHL2')(layer1)
    layer2D = Dropout(0.05, name='IHDL2')(layer2)
    layer3 = Dense(48, activation='relu', name='IHL3')(layer2D)
    layer4 = Dense(24, activation='relu', name='IHL4')(layer3)
    layer5 = Dense(12, activation='relu', name='IHL5')(layer4)
    output = Dense(6, activation='relu', name='IHL6')(layer5)
    return output, name

def arch7_i(inputs):
    name = 'arch7_i'
    layer_concat = keras.layers.Concatenate(axis=-1)(inputs)
    layer1 = Dense(46, activation='relu', name='IHL1')(layer_concat)
    layer2 = Dense(62, activation='relu', name='IHL2')(layer1)
    layer3 = Dense(62, activation='relu', name='IHL3')(layer2)
    layer4 = Dense(48, activation='relu', name='IHL4')(layer3)
    layer5 = Dense(24, activation='relu', name='IHL5')(layer4)
    layer6 = Dense(12, activation='relu', name='IHL6')(layer5)
    output = Dense(6, activation='relu', name='IHL7')(layer6)
    return output, name

def arch8_i(inputs):
    name = 'arch8_i'
    layer_concat = keras.layers.Concatenate(axis=-1)(inputs)
    layer1 = Dense(46, activation='relu', name='IHL1')(layer_concat)
    layer2 = Dense(60, activation='relu', name='IHL2')(layer1)
    layer3 = Dense(76, activation='relu', name='IHL3')(layer2)
    layer4 = Dense(60, activation='relu', name='IHL4')(layer3)
    layer5 = Dense(36, activation='relu', name='IHL5')(layer4)
    output = Dense(12, activation='relu', name='IHL6')(layer5)
    return output, name

def arch9_i(inputs):
    name = 'arch9_i'
    layer_concat = keras.layers.Concatenate(axis=-1)(inputs)
    layer1 = Dense(50, activation='relu', name='IHL1')(layer_concat)
    layer2 = Dense(70, activation='relu', name='IHL2')(layer1)
    layer3 = Dense(90, activation='relu', name='IHL3')(layer2)
    layer4 = Dense(70, activation='relu', name='IHL4')(layer3)
    layer5 = Dense(50, activation='relu', name='IHL5')(layer4)
    output = Dense(24, activation='relu', name='IHL6')(layer5)
    return output, name

def arch10_i(inputs):
    name = 'arch10_i'
    layer_concat = keras.layers.Concatenate(axis=-1)(inputs)
    layer1 = Dense(50, activation='relu', name='IHL1')(layer_concat)
    layer2 = Dense(70, activation='relu', name='IHL2')(layer1)
    layer3 = Dense(90, activation='relu', name='IHL3', kernel_regularizer=l2(1e-4))(layer2)
    layer4 = Dense(70, activation='relu', name='IHL4')(layer3)
    layer5 = Dense(50, activation='relu', name='IHL5')(layer4)
    output = Dense(24, activation='relu', name='IHL6')(layer5)
    return output, name
