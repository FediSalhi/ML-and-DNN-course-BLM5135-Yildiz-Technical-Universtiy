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
                            J_FONT_1_BIPOLAR, K_FONT_1_BIPOLAR, A_FONT_2_BIPOLAR, B_FONT_2_BIPOLAR, C_FONT_2_BIPOLAR,
                            D_FONT_2_BIPOLAR, E_FONT_2_BIPOLAR, J_FONT_2_BIPOLAR, K_FONT_2_BIPOLAR, A_FONT_3_BIPOLAR,
                            B_FONT_3_BIPOLAR, C_FONT_3_BIPOLAR, D_FONT_3_BIPOLAR, E_FONT_3_BIPOLAR, J_FONT_3_BIPOLAR,
                            K_FONT_3_BIPOLAR]

    targets_list_bipolar = [A_TARGET_BIPOLAR, B_TARGET_BIPOLAR, C_TARGET_BIPOLAR, D_TARGET_BIPOLAR, E_TARGET_BIPOLAR,
                            J_TARGET_BIPOLAR, K_TARGET_BIPOLAR, A_TARGET_BIPOLAR, B_TARGET_BIPOLAR, C_TARGET_BIPOLAR,
                            D_TARGET_BIPOLAR, E_TARGET_BIPOLAR, J_TARGET_BIPOLAR, K_TARGET_BIPOLAR, A_TARGET_BIPOLAR,
                            B_TARGET_BIPOLAR, C_TARGET_BIPOLAR, D_TARGET_BIPOLAR, E_TARGET_BIPOLAR, J_TARGET_BIPOLAR,
                            K_TARGET_BIPOLAR]

    letters_list_binary = [A_FONT_1_BINARY, B_FONT_1_BINARY, C_FONT_1_BINARY, D_FONT_1_BINARY, E_FONT_1_BINARY,
                            J_FONT_1_BINARY, K_FONT_1_BINARY, A_FONT_2_BINARY, B_FONT_2_BINARY, C_FONT_2_BINARY,
                            D_FONT_2_BINARY, E_FONT_2_BINARY, J_FONT_2_BINARY, K_FONT_2_BINARY, A_FONT_3_BINARY,
                            B_FONT_3_BINARY, C_FONT_3_BINARY, D_FONT_3_BINARY, E_FONT_3_BINARY, J_FONT_3_BINARY,
                            K_FONT_3_BINARY]

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


    elif (method == Encode_methods.BINARY):

        letters_list = letters_list_binary
        targets_list = targets_list_binary

        for letter in letters_list:
            data_inputs.append(letter)

        for target in targets_list:
            data_targets.append(target)

    data_inputs = np.array(data_inputs)
    data_targets = np.array(data_targets)

    return data_inputs, data_targets



def activate_bipolar(net, threshold):
    """

    :param net:
    :return:
    """
    if (net < threshold):
        activated_net = -1
    elif (net > threshold):
        activated_net = 1
    else:
        activated_net = 0
    return activated_net


def activate_binary(net, threshold):
    """

    :param net:
    :return:
    """
    if (net <= threshold):
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

    c = V1.shape[1]
    l = c
    sum = 0

    for i in range(l):
        res = V1[0,i] * V2[i,0]
        sum += res

    result = sum
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

def train_neural_network(learning_rule, encoding_method, data_inputs, data_targets, learning_rate):
    """

    :param learning_rule:
    :param epochs:
    :param data_inputs: shape --> (21,63) 21 = total number of letters and 63 = number of bits per letter
    :param data_targets: shape --> (21,7) 21 = total number of letters and 7 = number of classes
    :param learning_rate:
    :return:
    """

    weights = np.ones((NUMBER_OF_BITS_PER_LETTER, NUMBER_OF_CLASSES)) * WEIGHTS_INITIAL_VALUES_COEF # (63, 7)
    biases = np.ones((NUMBER_OF_CLASSES, 1)) * BIASES_INITIAL_VALUES_COEF

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
            input_vector = data_inputs[sample_letter_idx].reshape(1, NUMBER_OF_BITS_PER_LETTER)

            # Compute activation for each output unit
            for output_idx in range(NUMBER_OF_CLASSES):
                wj = weights[:, output_idx] # shape (63,1)
                bj = biases[output_idx]  # scaler
                net = compute_net(input_vector, wj, bj) #implement compute net
                if (encoding_method == Encode_methods.BIPOLAR):
                    yj = activate_bipolar(net, ACTIVATION_FUNCTION_THRESHOLD)
                elif (encoding_method == Encode_methods.BINARY):
                    yj = activate_binary(net, ACTIVATION_FUNCTION_THRESHOLD)
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

            # if (epochs == 100):
            #     stop_condition = True

        else:
            old_weights = copy.copy(weights)
            old_biases = copy.copy(biases)

    training_end_time = time.time()
    training_duration = training_end_time - training_start_time
    print(epochs)
    return weights, biases, epochs, training_duration

def evaluate_model(weights, biases, epochs, data_inputs, training_duration, encoding_method):
    """

    :param weights:
    :param biases:
    :param epochs:
    :return: general binary accuracy, binary accuracy for every letter,
    """

    if (encoding_method == Encode_methods.BIPOLAR):
        encoding = 'BIPOLAR'
    elif (encoding_method == Encode_methods.BINARY):
        encoding = 'BINARY'

    if (encoding == 'BIPOLAR'):

        test_dataset_inputs = [A_FONT_1_BIPOLAR, B_FONT_1_BIPOLAR, C_FONT_1_BIPOLAR, D_FONT_1_BIPOLAR, E_FONT_1_BIPOLAR,
                            J_FONT_1_BIPOLAR, K_FONT_1_BIPOLAR, A_FONT_2_BIPOLAR, B_FONT_2_BIPOLAR, C_FONT_2_BIPOLAR,
                            D_FONT_2_BIPOLAR, E_FONT_2_BIPOLAR, J_FONT_2_BIPOLAR, K_FONT_2_BIPOLAR, A_FONT_3_BIPOLAR,
                            B_FONT_3_BIPOLAR, C_FONT_3_BIPOLAR, D_FONT_3_BIPOLAR, E_FONT_3_BIPOLAR, J_FONT_3_BIPOLAR,
                            K_FONT_3_BIPOLAR]

        test_dataset_targets = [A_TARGET_BIPOLAR, B_TARGET_BIPOLAR, C_TARGET_BIPOLAR, D_TARGET_BIPOLAR, E_TARGET_BIPOLAR,
                            J_TARGET_BIPOLAR, K_TARGET_BIPOLAR, A_TARGET_BIPOLAR, B_TARGET_BIPOLAR, C_TARGET_BIPOLAR,
                            D_TARGET_BIPOLAR, E_TARGET_BIPOLAR, J_TARGET_BIPOLAR, K_TARGET_BIPOLAR, A_TARGET_BIPOLAR,
                            B_TARGET_BIPOLAR, C_TARGET_BIPOLAR, D_TARGET_BIPOLAR, E_TARGET_BIPOLAR, J_TARGET_BIPOLAR,
                            K_TARGET_BIPOLAR]

    elif (encoding == 'BINARY'):

        test_dataset_inputs = [A_FONT_1_BINARY, B_FONT_1_BINARY, C_FONT_1_BINARY, D_FONT_1_BINARY, E_FONT_1_BINARY,
                            J_FONT_1_BINARY, K_FONT_1_BINARY, A_FONT_2_BINARY, B_FONT_2_BINARY, C_FONT_2_BINARY,
                            D_FONT_2_BINARY, E_FONT_2_BINARY, J_FONT_2_BINARY, K_FONT_2_BINARY, A_FONT_3_BINARY,
                            B_FONT_3_BINARY, C_FONT_3_BINARY, D_FONT_3_BINARY, E_FONT_3_BINARY, J_FONT_3_BINARY,
                            K_FONT_3_BINARY]

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
        if (encoding_method == Encode_methods.BIPOLAR):
            prediction_all_classes = get_prediction_bipolar(weights, biases, test_dataset_inputs[input_vector_idx])
        elif (encoding_method == Encode_methods.BINARY):
            prediction_all_classes = get_prediction_binary(weights, biases, test_dataset_inputs[input_vector_idx])
        model_predictions.append(prediction_all_classes)

    model_predictions = np.array(model_predictions)

    # class A evaluation
    for prediction_idx in range(TEST_DATASET_TOTAL_NUMBER_OF_LETTERS):
        if (model_predictions[prediction_idx][CLASS_A_INDEX] == 1 and (prediction_idx == 0 or prediction_idx == 7 or prediction_idx == 14)):
            class_A_true_positives +=1
        elif (model_predictions[prediction_idx][CLASS_A_INDEX] != 1 and (prediction_idx != 0 or prediction_idx != 7 or prediction_idx != 14)):
            class_A_true_negatives += 1
        elif (model_predictions[prediction_idx][CLASS_A_INDEX] != 1 and (prediction_idx == 0 or prediction_idx == 7 or prediction_idx == 14)):
            class_A_false_negatives +=1
        elif (model_predictions[prediction_idx][CLASS_A_INDEX] == 1 and (prediction_idx != 0 or prediction_idx != 7 or prediction_idx != 14)):
            class_A_false_positives +=1

    # class B evaluation
    for prediction_idx in range(TEST_DATASET_TOTAL_NUMBER_OF_LETTERS):
        if (model_predictions[prediction_idx][CLASS_B_INDEX] == 1 and (prediction_idx == 1 or prediction_idx == 8 or prediction_idx == 15)):
            class_B_true_positives +=1
        elif (model_predictions[prediction_idx][CLASS_B_INDEX] != 1 and (prediction_idx != 1 or prediction_idx != 8 or prediction_idx != 15)):
            class_B_true_negatives += 1
        elif (model_predictions[prediction_idx][CLASS_B_INDEX] != 1 and (prediction_idx == 1 or prediction_idx == 8 or prediction_idx == 15)):
            class_B_false_negatives +=1
        elif (model_predictions[prediction_idx][CLASS_B_INDEX] == 1 and (prediction_idx != 1 or prediction_idx != 8 or prediction_idx != 15)):
            class_B_false_positives +=1

    # class C evaluation
    for prediction_idx in range(TEST_DATASET_TOTAL_NUMBER_OF_LETTERS):
        if (model_predictions[prediction_idx][CLASS_C_INDEX] == 1 and (prediction_idx == 2 or prediction_idx == 9 or prediction_idx == 16)):
            class_C_true_positives +=1
        elif (model_predictions[prediction_idx][CLASS_C_INDEX] != 1 and (prediction_idx != 2 or prediction_idx != 9 or prediction_idx != 16)):
            class_C_true_negatives += 1
        elif (model_predictions[prediction_idx][CLASS_C_INDEX] != 1 and (prediction_idx == 2 or prediction_idx == 9 or prediction_idx == 16)):
            class_C_false_negatives +=1
        elif (model_predictions[prediction_idx][CLASS_C_INDEX] == 1 and (prediction_idx != 2 or prediction_idx != 9 or prediction_idx != 16)):
            class_C_false_positives +=1

    # class D evaluation
    for prediction_idx in range(TEST_DATASET_TOTAL_NUMBER_OF_LETTERS):
        if (model_predictions[prediction_idx][CLASS_D_INDEX] == 1 and (prediction_idx == 3 or prediction_idx == 10 or prediction_idx == 17)):
            class_D_true_positives +=1
        elif (model_predictions[prediction_idx][CLASS_D_INDEX] != 1 and (prediction_idx != 3 or prediction_idx != 10 or prediction_idx != 17)):
            class_D_true_negatives += 1
        elif (model_predictions[prediction_idx][CLASS_D_INDEX] != 1 and (prediction_idx == 3 or prediction_idx == 10 or prediction_idx == 17)):
            class_D_false_negatives +=1
        elif (model_predictions[prediction_idx][CLASS_D_INDEX] == 1 and (prediction_idx != 3 or prediction_idx != 10 or prediction_idx != 17)):
            class_D_false_positives +=1

    # class E evaluation
    for prediction_idx in range(TEST_DATASET_TOTAL_NUMBER_OF_LETTERS):
        if (model_predictions[prediction_idx][CLASS_E_INDEX] == 1 and (prediction_idx == 4 or prediction_idx == 11 or prediction_idx == 18)):
            class_E_true_positives +=1
        elif (model_predictions[prediction_idx][CLASS_E_INDEX] != 1 and (prediction_idx != 4 or prediction_idx != 11 or prediction_idx != 18)):
            class_E_true_negatives += 1
        elif (model_predictions[prediction_idx][CLASS_E_INDEX] != 1 and (prediction_idx == 4 or prediction_idx == 11 or prediction_idx == 18)):
            class_E_false_negatives +=1
        elif (model_predictions[prediction_idx][CLASS_E_INDEX] == 1 and (prediction_idx != 4 or prediction_idx != 11 or prediction_idx != 18)):
            class_E_false_positives +=1

    # class J evaluation
    for prediction_idx in range(TEST_DATASET_TOTAL_NUMBER_OF_LETTERS):
        if (model_predictions[prediction_idx][CLASS_J_INDEX] == 1 and (prediction_idx == 5 or prediction_idx == 12 or prediction_idx == 19)):
            class_J_true_positives +=1
        elif (model_predictions[prediction_idx][CLASS_J_INDEX] != 1 and (prediction_idx != 5 or prediction_idx != 12 or prediction_idx != 19)):
            class_J_true_negatives += 1
        elif (model_predictions[prediction_idx][CLASS_J_INDEX] != 1 and (prediction_idx == 5 or prediction_idx == 12 or prediction_idx == 19)):
            class_J_false_negatives +=1
        elif (model_predictions[prediction_idx][CLASS_J_INDEX] == 1 and (prediction_idx != 5 or prediction_idx != 12 or prediction_idx != 19)):
            class_J_false_positives +=1

    # class K evaluation
    for prediction_idx in range(TEST_DATASET_TOTAL_NUMBER_OF_LETTERS):
        if (model_predictions[prediction_idx][CLASS_K_INDEX] == 1 and (prediction_idx == 6 or prediction_idx == 13 or prediction_idx == 20)):
            class_K_true_positives +=1
        elif (model_predictions[prediction_idx][CLASS_K_INDEX] != 1 and (prediction_idx != 6 or prediction_idx != 13 or prediction_idx != 20)):
            class_K_true_negatives += 1
        elif (model_predictions[prediction_idx][CLASS_K_INDEX] != 1 and (prediction_idx == 6 or prediction_idx == 13 or prediction_idx == 20)):
            class_K_false_negatives +=1
        elif (model_predictions[prediction_idx][CLASS_K_INDEX] == 1 and (prediction_idx != 6 or prediction_idx != 13 or prediction_idx != 20)):
            class_K_false_positives +=1

    # compute accuracies
    class_A_accuracy = (class_A_true_positives + class_A_true_negatives) * 100 / (TEST_DATASET_TOTAL_NUMBER_OF_LETTERS)
    class_B_accuracy = (class_B_true_positives + class_B_true_negatives) * 100 / (TEST_DATASET_TOTAL_NUMBER_OF_LETTERS)
    class_C_accuracy = (class_C_true_positives + class_C_true_negatives) * 100 / (TEST_DATASET_TOTAL_NUMBER_OF_LETTERS)
    class_D_accuracy = (class_D_true_positives + class_D_true_negatives) * 100 / (TEST_DATASET_TOTAL_NUMBER_OF_LETTERS)
    class_E_accuracy = (class_E_true_positives + class_E_true_negatives) * 100 / (TEST_DATASET_TOTAL_NUMBER_OF_LETTERS)
    class_J_accuracy = (class_J_true_positives + class_J_true_negatives) * 100 / (TEST_DATASET_TOTAL_NUMBER_OF_LETTERS)
    class_K_accuracy = (class_K_true_positives + class_K_true_negatives) * 100 / (TEST_DATASET_TOTAL_NUMBER_OF_LETTERS)

    training_duration_str = str(training_duration)[0:4]

    # print(class_A_accuracy)
    # print(class_B_accuracy)
    # print(class_C_accuracy)
    # print(class_D_accuracy)
    # print(class_E_accuracy)
    # print(class_J_accuracy)
    # print(class_K_accuracy)

    print("""
    *********************************** Model Evaluation ***********************************
    Data Encoding Method:     {}
    Activation Function:      {} (threshold = {})
    Learning Rule:            {}
    Learning Rate:            {}
    Size of Training Dataset: {} samples (letter), {} bits/sample
    Training Time :           {} seconds
    Number of Epochs:         {}
    ________________________________________________________________________________________
    Class A Accuracy :        {} %
    Class A True Positives:   {} 
    Class A True Negatives:   {} 
    Class A False Positive :  {}
    Class A False Negative:   {}
    ________________________________________________________________________________________
    Class B Accuracy :        {} %
    Class B True Positives:   {}
    Class B True Negatives:   {}
    Class B False Positive :  {}
    Class B False Negative:   {}
    ________________________________________________________________________________________
    Class C Accuracy :        {} %
    Class C True Positives:   {}
    Class C True Negatives:   {}
    Class C False Positive :  {}
    Class C False Negative:   {}
    ________________________________________________________________________________________
    Class D Accuracy :        {} %
    Class D True Positives:   {}
    Class D True Negatives:   {}
    Class D False Positive :  {}
    Class D False Negative:   {}
    ________________________________________________________________________________________
    Class E Accuracy :        {} %
    Class E True Positives:   {}
    Class E True Negatives:   {}
    Class E False Positive :  {}
    Class E False Negative:   {}
    ________________________________________________________________________________________
    Class J Accuracy :        {} %
    Class J True Positives:   {}
    Class J True Negatives:   {}
    Class J False Positive :  {}
    Class J False Negative:   {}
    ________________________________________________________________________________________
    Class K Accuracy :        {} %
    Class K True Positives:   {}
    Class K True Negatives:   {}
    Class K False Positive :  {}
    Class K False Negative:   {}
    
    """.format(encoding, encoding, ACTIVATION_FUNCTION_THRESHOLD,'Perceptron', LEARNING_RATE, TEST_DATASET_TOTAL_NUMBER_OF_LETTERS,
               NUMBER_OF_BITS_PER_LETTER, training_duration_str, epochs,
               class_A_accuracy, class_A_true_positives, class_A_true_negatives, class_A_false_positives, class_A_false_negatives,
               class_B_accuracy, class_B_true_positives, class_B_true_negatives, class_B_false_positives, class_B_false_negatives,
               class_C_accuracy, class_C_true_positives, class_C_true_negatives, class_C_false_positives, class_C_false_negatives,
               class_D_accuracy, class_D_true_positives, class_D_true_negatives, class_D_false_positives, class_D_false_negatives,
               class_E_accuracy, class_E_true_positives, class_E_true_negatives, class_E_false_positives, class_E_false_negatives,
               class_J_accuracy, class_J_true_positives, class_J_true_negatives, class_J_false_positives, class_J_false_negatives,
               class_K_accuracy, class_K_true_positives, class_K_true_negatives, class_K_false_positives, class_K_false_negatives))


def get_prediction_bipolar(weight, biases, input_vector):
    preds = []
    input_vector = np.array(input_vector)
    for output_idx in range(NUMBER_OF_CLASSES):
        net = compute_net(input_vector, weight[:,output_idx], biases[output_idx])
        prediction = activate_bipolar(net, ACTIVATION_FUNCTION_THRESHOLD)
        preds.append(prediction)
    return preds

def get_prediction_binary(weight, biases, input_vector):
    preds = []
    input_vector = np.array(input_vector)
    for output_idx in range(NUMBER_OF_CLASSES):
        net = compute_net(input_vector, weight[:,output_idx], biases[output_idx])
        prediction = activate_binary(net, ACTIVATION_FUNCTION_THRESHOLD)
        preds.append(prediction)
    return preds








