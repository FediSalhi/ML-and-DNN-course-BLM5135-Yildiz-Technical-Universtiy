################ imports ################
import numpy as np
from global_vars import *
#########################################

def prepare_dataset(method):
    """

    :param font_type:
    :param method:
    :return:
    """

    data_inputs = []
    data_targets = []

    letters_list_bipolar = [A_FONT_1_BIPOLAR, B_FONT_1_BIPOLAR, C_FONT_1_BIPOLAR, D_FONT_1_BIPOLAR, E_FONT_1_BIPOLAR,
                            J_FONT_1_BIPOLAR, K_FONT_1_BIPOLAR, A_FONT_1_BIPOLAR, B_FONT_1_BIPOLAR, C_FONT_1_BIPOLAR,
                            D_FONT_1_BIPOLAR, E_FONT_1_BIPOLAR, J_FONT_1_BIPOLAR, K_FONT_1_BIPOLAR, A_FONT_1_BIPOLAR,
                            B_FONT_1_BIPOLAR, C_FONT_1_BIPOLAR, D_FONT_1_BIPOLAR, E_FONT_1_BIPOLAR, J_FONT_1_BIPOLAR,
                            K_FONT_1_BIPOLAR]

    targets_list_bipolar = [A_TARGET_BIPOLAR, B_TARGET_BIPOLAR, C_TARGET_BIPOLAR, D_TARGET_BIPOLAR, E_TARGET_BIPOLAR,
                            J_TARGET_BIPOLAR, A_TARGET_BIPOLAR]

    # letters_list_bipolar = [A_FONT_1_BINARY, B_FONT_1_BINARY, C_FONT_1_BINARY, D_FONT_1_BINARY, E_FONT_1_BINARY,
    #                         J_FONT_1_BINARY, K_FONT_1_BINARY, A_FONT_1_BINARY, B_FONT_1_BINARY, C_FONT_1_BINARY,
    #                         D_FONT_1_BINARY, E_FONT_1_BINARY, J_FONT_1_BINARY, K_FONT_1_BINARY, A_FONT_1_BINARY,
    #                         B_FONT_1_BINARY, C_FONT_1_BINARY, D_FONT_1_BINARY, E_FONT_1_BINARY, J_FONT_1_BINARY,
    #                         K_FONT_1_BINARY] #TODO: complete these variables

    targets_list_binary = [A_TARGET_BINARY, B_TARGET_BINARY, C_TARGET_BINARY, D_TARGET_BINARY, E_TARGET_BINARY,
                            J_TARGET_BINARY, K_TARGET_BINARY]


    if (method == Encode_methods.BIPOLAR):

        letters_list = letters_list_bipolar
        targets_list = targets_list_bipolar

        for letter in letters_list:
            data_inputs.append(letter)

        for target in targets_list:
            data_targets.append(target)

    # elif (method == Encode_methods.BINARY):
    #
    #     letters_list = letters_list_binary
    #     targets_list = targets_list_binary
    #
    #     for letter in letters_list:
    #         data_inputs.append(letter)
    #
    #     for target in targets_list:
    #         data_targets.append(target)

    return np.array(data_inputs), np.array(data_targets)



def bipolar_activation(net):
    for i in net:
        if i < 0:
            net[i] = -1
        else:
            net[i] = 1

def binary_activation(net):
    for i in net:
        if i < 0:
            net[i] = 0
        else:
            net[i] = 1



# def update_weights_and_biases:
    #TODO: from here



# def train_neural_network(epochs, data_x, data_t, learning_rate=1):
#     #TODO: add shuffling option
#     #TODO: add other learning rules
#
#     weights = np.zeros((NUMBER_OF_BITS_PER_LETTER,))
#     biases = np.zeros((NUMBER_OF_CLASSES, ))
#
#     old_weights = np.zeros((NUMBER_OF_BITS_PER_LETTER,))
#     old_biases = np.zeros((NUMBER_OF_CLASSES,))
#
#     for epoch in range(epochs):
#
#         for pattern_idx in range(NUMBER_OF_PATTERNS):
#             net = biases + np.dot(weights, data_x[pattern_idx])
#             y_out = bipolar_activation(net)
#
#             if (y_out != data_t[pattern_idx]):
#                 weights, biases = update_weights_and_biases(weights, biases, learning_rate, data_t[pattern_idx], data_x[pattern_idx])
#             else:
#                 continue
#
#         if (weights == old_weights and biases == old_biases):
#             break

inp, out = prepare_dataset(Encode_methods.BIPOLAR)
print(inp)
