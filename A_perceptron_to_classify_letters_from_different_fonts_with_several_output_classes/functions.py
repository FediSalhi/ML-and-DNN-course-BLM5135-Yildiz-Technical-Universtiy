################ imports ################
import numpy as np
from global_vars import *
import copy
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



def activate_bipolar(net):
    """

    :param net:
    :return:
    """
    if (net <= 0):
        activated_net = -1
    else:
        activated_net = 1
    return activated_net


def activate_binary(net):
    """

    :param net:
    :return:
    """
    if (net <= 0):
        activated_net = 0
    else:
        activated_net = 1
    return activated_net



def matrix_multiplication(M1, M2):
    result = np.matmul(M1, M2)
    return result

def  multiply_vectors(V1, V2):
    """
    dot multiplication of two vectors
    :param V1: shape --> (1,n)
    :param V2: shape --> (n,1)
    :return: V1.V2  shape --> scaler
    """

    assert(np.ndim(V1) == 2, "X")
    assert(np.ndim(V2) == 2, "X")
    assert(V1.shape[0] == V2.shape[1], "X") #TODO: fill in this message
    assert(V1.shape[1] == V2.shape[0], "X")

    result = np.dot(V1, V2)
    return result



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

        for output in range(NUMBER_OF_CLASSES):









        t_multiply_x = matrix_multiplication(data_target.reshape(7, 1), data_input.reshape(1, 63))

        weights += learning_rate * np.transpose(t_multiply_x)

        biases = np.add(biases, (learning_rate * data_target).reshape(7,1))

    elif (learning_rule == Learning_rules.DELTA):
        pass

    elif (learning_rule == Learning_rules.HEBB):
        pass

    return weights, biases

def train_neural_network(learning_rule, epochs, data_inputs, data_targets, learning_rate=1):
    #TODO: add shuffling option
    #TODO: add other learning rules

    weights = np.zeros((NUMBER_OF_BITS_PER_LETTER, NUMBER_OF_CLASSES))
    biases = np.zeros((NUMBER_OF_CLASSES, 1))

    for epoch in range(epochs):

        old_weights = copy.copy(weights)
        old_biases = copy.copy(biases)


        for pattern_idx in range(NUMBER_OF_LETTERS_PER_FONT * NUMBER_OF_FONTS):
            x = data_inputs[pattern_idx].reshape(1, 63)
            activated_output = []
            # Compute activation for each output unit
            for output_idx in range(NUMBER_OF_CLASSES):
                wj = weights[:, output_idx]
                bj = biases[output_idx]
                net = compute_net(x, wj, bj)
                yj = activate_bipolar(net)
                activated_output.append(yj)

            target_j = data_targets[pattern_idx % 7]

           #TODO: here
            # Update biases and weights
            for output_idx in range(NUMBER_OF_CLASSES):
                if (target_j[output_idx] != activated_output[output_idx]):
                    w_old = weights[:, output_idx]
                    b_old = biases[output_idx]
                    b_new, w_new = update_weights_and_biases(Learning_rules.PERCEPTRON, learning_rate, b_old, w_old, target_j, x, )

            if (~np.array_equal(y_out, data_targets[pattern_idx % 7].reshape(7,1))):
                weights, biases = update_weights_and_biases(Learning_rules.PERCEPTRON, weights, biases, learning_rate, data_inputs[pattern_idx], data_targets[pattern_idx % 7])
            else:
                continue

        if (np.array_equal(weights, old_weights) and np.array_equal(biases, old_biases)):
            print(epoch)
            return weights, biases
    print(epoch)
    return weights, biases


data_inputs, data_targets = prepare_dataset(Encode_methods.BIPOLAR)

weights, biases = train_neural_network(1, 100, data_inputs, data_targets, 1)

print(weights.shape)
print(biases.shape)

print(weights)
print(biases)



a = np.array([1,2])
np.ndim



