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
    for idx in range(NUMBER_OF_CLASSES):
        if net[idx] < 0:
            net[idx] = -1
        else:
            net[idx] = 1
    return net

def binary_activation(net):
    for idx in range(NUMBER_OF_CLASSES):
        if net[idx] < 0:
            net[idx] = 0
        else:
            net[idx] = 1
    return net

def matrix_product(V1, V2):

    #TODO: not well written, but it works

    """

    :param V1:
    :param V2:
    :return:
    """
    V1 = V1.reshape(-1, 1)
    V2 = V2.reshape(1, -1)

    l1 = V1.shape[0]
    c1 = V1.shape[1]

    l2 = V2.shape[0]
    c2 = V2.shape[1]

    V1_cross_V2 = np.zeros((l1, c2))

    for line in range(l1):
        for column in range(c2):
            V1_cross_V2[line, column] = V1[line, 0] * V2[0, column]

    return V1_cross_V2


def  update_weights_and_biases(learning_rule, weights, biases, learning_rate, data_input, data_target):
    """

    :param learning_rule:
    :param weights:
    :param biases:
    :param learning_rate:
    :param data_input:
    :param data_target:
    :return:
    """
    if (learning_rule == Learning_rules.PERCEPTRON):
        weights += learning_rate * data_target * data_input #TODO: change with cross product

        np.array

        biases += learning_rate * data_target

    elif (learning_rule == Learning_rules.DELTA):
        pass

    elif (learning_rule == Learning_rules.HEBB):
        pass

    return weights, biases

def train_neural_network(learning_rule, epochs, data_inputs, data_targets, learning_rate=1):
    #TODO: add shuffling option
    #TODO: add other learning rules

    weights = np.zeros((NUMBER_OF_BITS_PER_LETTER * NUMBER_OF_CLASSES,))
    biases = np.zeros((NUMBER_OF_CLASSES, ))

    old_weights = np.zeros((NUMBER_OF_BITS_PER_LETTER,))
    old_biases = np.zeros((NUMBER_OF_CLASSES,))

    for epoch in range(epochs):

        old_weights = weights
        old_biases = biases

        for pattern_idx in range(NUMBER_OF_LETTERS_PER_FONT * NUMBER_OF_FONTS):
            net = biases + np.dot(weights, data_inputs[pattern_idx])
            y_out = bipolar_activation(net)

            if ((y_out != data_targets[pattern_idx % 7]).any()):
                weights, biases = update_weights_and_biases(Learning_rules.PERCEPTRON, weights, biases, learning_rate, data_inputs[pattern_idx], data_targets[pattern_idx % 7])
            else:
                continue

        # if ((weights == old_weights).any() and (biases == old_biases).any()):
        #     return weights, biases

    return weights, biases


# data_inputs, data_targets = prepare_dataset(Encode_methods.BIPOLAR)
#
# weights, biases = train_neural_network(1, 100, data_inputs, data_targets, 1)
#
# print(weights)
# print(biases)

x = cross_product(np.array([1,2,3]), np.array([6,1,9]))
print(x)


