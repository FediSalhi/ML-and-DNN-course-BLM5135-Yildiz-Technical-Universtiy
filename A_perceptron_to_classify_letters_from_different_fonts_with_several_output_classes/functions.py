################ imports ################
import numpy as np
from constants import *
import copy
import time
#########################################

def prepare_dataset(method):
    """

    :param font_type:
    :param method:
    :return: data_inputs --> (21,63), data_targets --> (21,7)
    """

    data_inputs = []
    data_targets = []

    letters_list_bipolar = [A_FONT_1_BIPOLAR, B_FONT_1_BIPOLAR, C_FONT_1_BIPOLAR, D_FONT_1_BIPOLAR, E_FONT_1_BIPOLAR,
                            J_FONT_1_BIPOLAR, K_FONT_1_BIPOLAR, A_FONT_1_BIPOLAR, B_FONT_1_BIPOLAR, C_FONT_1_BIPOLAR,
                            D_FONT_1_BIPOLAR, E_FONT_1_BIPOLAR, J_FONT_1_BIPOLAR, K_FONT_1_BIPOLAR, A_FONT_1_BIPOLAR,
                            B_FONT_1_BIPOLAR, C_FONT_1_BIPOLAR, D_FONT_1_BIPOLAR, E_FONT_1_BIPOLAR, J_FONT_1_BIPOLAR,
                            K_FONT_1_BIPOLAR]

    targets_list_bipolar = [A_TARGET_BIPOLAR, B_TARGET_BIPOLAR, C_TARGET_BIPOLAR, D_TARGET_BIPOLAR, E_TARGET_BIPOLAR,
                            J_TARGET_BIPOLAR, K_TARGET_BIPOLAR, A_TARGET_BIPOLAR, B_TARGET_BIPOLAR, C_TARGET_BIPOLAR,
                            D_TARGET_BIPOLAR, E_TARGET_BIPOLAR, J_TARGET_BIPOLAR, K_TARGET_BIPOLAR, A_TARGET_BIPOLAR,
                            B_TARGET_BIPOLAR, C_TARGET_BIPOLAR, D_TARGET_BIPOLAR, E_TARGET_BIPOLAR, J_TARGET_BIPOLAR,
                            K_TARGET_BIPOLAR]

    letters_list_bipolar = [A_FONT_1_BINARY, B_FONT_1_BINARY, C_FONT_1_BINARY, D_FONT_1_BINARY, E_FONT_1_BINARY,
                            J_FONT_1_BINARY, K_FONT_1_BINARY, A_FONT_1_BINARY, B_FONT_1_BINARY, C_FONT_1_BINARY,
                            D_FONT_1_BINARY, E_FONT_1_BINARY, J_FONT_1_BINARY, K_FONT_1_BINARY, A_FONT_1_BINARY,
                            B_FONT_1_BINARY, C_FONT_1_BINARY, D_FONT_1_BINARY, E_FONT_1_BINARY, J_FONT_1_BINARY,
                            K_FONT_1_BINARY]

    targets_list_binary = [A_TARGET_BINARY, B_TARGET_BINARY, C_TARGET_BINARY, D_TARGET_BINARY, E_TARGET_BINARY,
                           J_TARGET_BINARY, K_TARGET_BINARY, A_TARGET_BINARY, B_TARGET_BINARY, C_TARGET_BINARY,
                           D_TARGET_BINARY, E_TARGET_BINARY, J_TARGET_BINARY, K_TARGET_BINARY, A_TARGET_BINARY,
                           B_TARGET_BINARY, C_TARGET_BINARY, D_TARGET_BINARY, E_TARGET_BINARY, J_TARGET_BINARY,
                           K_TARGET_BINARY]


    if (method == Encode_methods.BIPOLAR):

        letters_list = letters_list_bipolar
        targets_list = targets_list_bipolar

        for letter in letters_list:
            data_inputs.append(letter)

        for target in targets_list:
            data_targets.append(target)

        data_inputs = np.array(data_inputs)
        data_targets = np.array(data_targets)

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

    return data_inputs, data_targets



def activate_bipolar(net):
    """

    :param net:
    :return:
    """
    if (net < 0):
        activated_net = -1
    elif (net > 0):
        activated_net = 1
    else:
        activated_net = 0
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

    assert np.ndim(V1) == 2, "X"
    assert np.ndim(V2) == 2, "X"
    assert V1.shape[0] == V2.shape[1], "X" #TODO: fill in this message
    assert V1.shape[1] == V2.shape[0], "X"

    result = np.dot(V1, V2)
    return result



def  update_weights_and_biases(learning_rule, learning_rate, bj_old, wj_old, target_j, input_vector):
    """

    :param Learning_rules:
    :param learning_rate:
    :param bj_old: sclaer
    :param wj_old: vector (63,)
    :param target_vector: (7,)
    :param input_vector: (1,63)
    :return:
    """

    if (learning_rule == Learning_rules.PERCEPTRON):
        bj_new = bj_old + learning_rate * target_j
        wj_new = wj_old + learning_rate * target_j * input_vector
    else:
        bj_new = bj_old
        wj_new = wj_old

    return bj_new, wj_new

def compute_net(input_vector, wj, bj):
    input_vector = input_vector.reshape(1,-1)
    wj = wj.reshape(-1,1)
    net = bj + multiply_vectors(input_vector, wj)
    return net

def compare_vectors(V1, V2):
    counter = 0
    length = (np.array(V1)).shape[0]
    for idx in range(length):
        if (V1[idx] != V2[idx]):
            break
        else:
            counter += 1
    if (counter == length):
        return True
    else:
        return False

def compare_matrices(M1, M2):
    counter = 0
    l = (np.array(M1)).shape[0]
    c = (np.array(M1)).shape[1]
    number_of_elements = l*c
    for l_idx in range(l):
        for c_idx in range(c):
            if (M1[l_idx][c_idx] != M2[l_idx][c_idx]):
                break
            else:
                counter += 1

    if (counter == number_of_elements):
        return True
    else:
        return False

def train_neural_network(learning_rule, encoding_method, data_inputs, data_targets, learning_rate=0.1):
    """

    :param learning_rule:
    :param epochs:
    :param data_inputs: shape --> (21,63) 21 = total number of letters and 63 = number of bits per letter
    :param data_targets: shape --> (21,7) 21 = total number of letters and 7 = number of classes
    :param learning_rate:
    :return:
    """

    weights = np.zeros((NUMBER_OF_BITS_PER_LETTER, NUMBER_OF_CLASSES)) # (63, 7)
    biases = np.zeros((NUMBER_OF_CLASSES, 1))

    old_weights = copy.copy(weights)
    old_biases = copy.copy(biases)
    stop_condition = False
    epochs = 0

    training_start_time = time.time()
    while (stop_condition == False):
        epochs += 1
        for sample_letter_idx in range(DATASET_TOTAL_NUMBER_OF_LETTERS):

            target_vector = data_targets[sample_letter_idx]
            activated_output = [] # shape (7,)
            input_vector = data_inputs[sample_letter_idx].reshape(1, 63) #TODO: change with constants

            # Compute activation for each output unit
            for output_idx in range(NUMBER_OF_CLASSES):
                wj = weights[:, output_idx] # shape (63,1)
                bj = biases[output_idx]  # scaler
                net = compute_net(input_vector, wj, bj) #implement compute net
                if (encoding_method == Encode_methods.BIPOLAR):
                    yj = activate_bipolar(net)
                elif (encoding_method == Encode_methods.BINARY):
                    yj = activate_binary(net)
                activated_output.append(yj)

            # Update biases and weights
            for output_i in range(NUMBER_OF_CLASSES):
                if (activated_output[output_i] != target_vector[output_i]):
                    wj_old = weights[:, output_i] # vector (63,)
                    bj_old = biases[output_i] # scaler
                    target_j = target_vector[output_i]
                    bj_new, wj_new = update_weights_and_biases(Learning_rules.PERCEPTRON, learning_rate, bj_old, wj_old, target_j, input_vector)
                else:
                    continue

                weights[:,output_i] = wj_new
                biases[output_i] = bj_new

            # stop condition
            if (compare_matrices(weights, old_weights) == True):
                stop_condition = True
            else:
                old_weights = copy.copy(weights)
                old_biases = copy.copy(biases)
    training_end_time = time.time()
    training_duration = training_end_time - training_start_time
    return weights, biases, epochs, training_duration

def evaluate_model(weights, biases, epochs, data_inputs, training_duration, encoding_method):
    """

    :param weights:
    :param biases:
    :param epochs:
    :return: general binary accuracy, binary accuracy for every letter,
    """
    if (encoding_method == Encode_methods.BIPOLAR):

        test_dataset_inputs = [A_FONT_1_BIPOLAR, B_FONT_1_BIPOLAR, C_FONT_1_BIPOLAR, D_FONT_1_BIPOLAR, E_FONT_1_BIPOLAR,
                            J_FONT_1_BIPOLAR, K_FONT_1_BIPOLAR, A_FONT_1_BIPOLAR, B_FONT_1_BIPOLAR, C_FONT_1_BIPOLAR,
                            D_FONT_1_BIPOLAR, E_FONT_1_BIPOLAR, J_FONT_1_BIPOLAR, K_FONT_1_BIPOLAR, A_FONT_1_BIPOLAR,
                            B_FONT_1_BIPOLAR, C_FONT_1_BIPOLAR, D_FONT_1_BIPOLAR, E_FONT_1_BIPOLAR, J_FONT_1_BIPOLAR,
                            K_FONT_1_BIPOLAR]

        test_dataset_targets = [A_TARGET_BIPOLAR, B_TARGET_BIPOLAR, C_TARGET_BIPOLAR, D_TARGET_BIPOLAR, E_TARGET_BIPOLAR,
                            J_TARGET_BIPOLAR, K_TARGET_BIPOLAR, A_TARGET_BIPOLAR, B_TARGET_BIPOLAR, C_TARGET_BIPOLAR,
                            D_TARGET_BIPOLAR, E_TARGET_BIPOLAR, J_TARGET_BIPOLAR, K_TARGET_BIPOLAR, A_TARGET_BIPOLAR,
                            B_TARGET_BIPOLAR, C_TARGET_BIPOLAR, D_TARGET_BIPOLAR, E_TARGET_BIPOLAR, J_TARGET_BIPOLAR,
                            K_TARGET_BIPOLAR]

    elif (encoding_method == Encode_methods.BINARY):

        test_dataset_inputs = [A_FONT_1_BINARY, B_FONT_1_BINARY, C_FONT_1_BINARY, D_FONT_1_BINARY, E_FONT_1_BINARY,
                            J_FONT_1_BINARY, K_FONT_1_BINARY, A_FONT_1_BINARY, B_FONT_1_BINARY, C_FONT_1_BINARY,
                            D_FONT_1_BINARY, E_FONT_1_BINARY, J_FONT_1_BINARY, K_FONT_1_BINARY, A_FONT_1_BINARY,
                            B_FONT_1_BINARY, C_FONT_1_BINARY, D_FONT_1_BINARY, E_FONT_1_BINARY, J_FONT_1_BINARY,
                            K_FONT_1_BINARY]

        test_dataset_targets = [A_TARGET_BINARY, B_TARGET_BINARY, C_TARGET_BINARY, D_TARGET_BINARY, E_TARGET_BINARY,
                           J_TARGET_BINARY, K_TARGET_BINARY, A_TARGET_BINARY, B_TARGET_BINARY, C_TARGET_BINARY,
                           D_TARGET_BINARY, E_TARGET_BINARY, J_TARGET_BINARY, K_TARGET_BINARY, A_TARGET_BINARY,
                           B_TARGET_BINARY, C_TARGET_BINARY, D_TARGET_BINARY, E_TARGET_BINARY, J_TARGET_BINARY,
                           K_TARGET_BINARY]

    model_predictions = []
    class_A_false_positives = 0
    class_A_false_negatives = 0
    class_A_true_positives = 0
    class_A_true_negatives = 0

    class_B_false_positives = 0
    class_B_false_negatives = 0
    class_B_true_positives = 0
    class_B_true_negatives = 0

    class_C_false_positives = 0
    class_C_false_negatives = 0
    class_C_true_positives = 0
    class_C_true_negatives = 0

    class_D_false_positives = 0
    class_D_false_negatives = 0
    class_D_true_positives = 0
    class_D_true_negatives = 0

    class_E_false_positives = 0
    class_E_false_negatives = 0
    class_E_true_positives = 0
    class_E_true_negatives = 0

    class_J_false_positives = 0
    class_J_false_negatives = 0
    class_J_true_positives = 0
    class_J_true_negatives = 0

    class_K_false_positives = 0
    class_K_false_negatives = 0
    class_K_true_positives = 0
    class_K_true_negatives = 0

    for input_vector_idx in range(TEST_DATASET_TOTAL_NUMBER_OF_LETTERS):
        prediction_all_classes = get_prediction(weights, biases, test_dataset_inputs[input_vector_idx])
        model_predictions.append(prediction_all_classes)

    model_predictions = np.array(model_predictions)

    # class A evaluation
    for prediction_idx in range(TEST_DATASET_TOTAL_NUMBER_OF_LETTERS):
        if (model_predictions[prediction_idx][CLASS_A_INDEX] == 1):
            class_A_true_positives +=1


    # class B evaluation
    # class C evaluation
    # class D evaluation
    # class E evaluation
    # class J evaluation
    # class K evaluation






def get_prediction(weight, biases, input_vector):
    preds = []
    input_vector = np.array(input_vector)
    for output_idx in range(NUMBER_OF_CLASSES):
        net = compute_net(input_vector, weight[:,output_idx], biases[output_idx])
        prediction = activate_bipolar(net)
        preds.append(prediction)
    return preds








