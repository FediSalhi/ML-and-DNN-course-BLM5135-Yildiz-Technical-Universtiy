################ imports ################
import numpy as np
from global_vars import *
#########################################

def encode_target(letter, method):
    """

    :param letter:
    :param method:
    :return:
    """
    if (method == Target_encode_methods.BINARY):
        if (letter == 'A'):
            encoded_target = [1, 0, 0, 0, 0, 0, 0]

        if (letter == 'A'):
            encoded_target = [0, 1, 0, 0, 0, 0, 0]

        if (letter == 'A'):
            encoded_target = [0, 0, 1, 0, 0, 0, 0]

        if (letter == 'A'):
            encoded_target = [0, 0, 0, 1, 0, 0, 0]

        if (letter == 'A'):
            encoded_target = [0, 0, 0, 0, 1, 0, 0]

        if (letter == 'A'):
            encoded_target = [0, 0, 0, 0, 0, 1, 0]

        if (letter == 'A'):
            encoded_target = [0, 0, 0, 0, 0, 0, 1]

    if (method == Target_encode_methods.BIPOLAR):
        if (letter == 'A'):
            encoded_target = [1, -1, -1, -1, -1, -1, -1]

        if (letter == 'A'):
            encoded_target = [-1, 1, -1, -1, -1, -1, -1]

        if (letter == 'A'):
            encoded_target = [-1, -1, 1, -1, -1, -1, -1]

        if (letter == 'A'):
            encoded_target = [-1, -1, -1, 1, -1, -1, -1]

        if (letter == 'A'):
            encoded_target = [-1, -1, -1, -1, 1, -1, -1]

        if (letter == 'A'):
            encoded_target = [-1, -1, -1, -1, -1, 1, -1]

        if (letter == 'A'):
            encoded_target = [-1, -1, -1, -1, -1, -1, 1]

    return encoded_target

def letter_to_binary(letter, font_type):
    """
    :param letter:
    :param font_type:
    :return:
    """
    if (font_type == Font_types.FONT_1):
        if (letter == 'A'):
            A_font_1 = np.array([0, 0, 1, 1, 0, 0, -0,
                                 0, 0, 0, 1, 0, 0, 0,
                                 0, 0, 0, 1, 0, 0, 0,
                                 0, 0, 1, 0, 1, 0, 0,
                                 0, 0, 1, 0, 1, 0, 0,
                                 0, 1, 1, 1, 1, 1, 0,
                                 0, 1, 0, 0, 0, 1, 0,
                                 0, 1, 0, 0, 0, 1, 0,
                                 1, 1, 1, 0, 1, 1 ,1])

    #     if (letter == 'B'):
    #         # TODO: kitap sayfa 72
    #     if (letter == 'C'):
    #     if (letter == 'D'):
    #     if (letter == 'E'):
    #     if (letter == 'J'):
    #     if (letter == 'K'):
    #
    # if (font_type == font_types.FONT_1):
    #     if (letter == 'A'):
    #     if (letter == 'B'):
    #             # TODO: kitap sayfa 72
    #     if (letter == 'C'):
    #     if (letter == 'D'):
    #     if (letter == 'E'):
    #     if (letter == 'J'):
    #     if (letter == 'K'):
    #
    # if (font_type == font_types.FONT_1):
    #     if (letter == 'A'):
    #     if (letter == 'B'):
    #             # TODO: kitap sayfa 72
    #     if (letter == 'C'):
    #     if (letter == 'D'):
    #     if (letter == 'E'):
    #     if (letter == 'J'):
    #     if (letter == 'K'):


def letter_to_bipolar(letter, font_type):
    """
    :param letter:
    :param font_type:
    :return:
    """
    if (font_type == Font_types.FONT_1):
        if (letter == 'A'):
            A_font_1 = np.array([-1, -1, 1, 1, -1, -1, -1,
                                 -1, -1, -1, 1, -1, -1, -1,
                                 -1, -1, -1, 1, -1, -1, -1,
                                 -1, -1, 1, -1, 1, -1, -1,
                                 -1, -1, 1, -1, 1, -1, -1,
                                 -1, 1, 1, 1, 1, 1, -1,
                                 -1, 1, -1, -1, -1, 1, -1,
                                 -1, 1, -1, -1, -1, 1, -1,
                                 1, 1, 1, -1, 1, 1 ,1])

    #     if (letter == 'B'):
    #         # TODO: kitap sayfa 72
    #     if (letter == 'C'):
    #     if (letter == 'D'):
    #     if (letter == 'E'):
    #     if (letter == 'J'):
    #     if (letter == 'K'):
    #
    # if (font_type == font_types.FONT_1):
    #     if (letter == 'A'):
    #     if (letter == 'B'):
    #             # TODO: kitap sayfa 72
    #     if (letter == 'C'):
    #     if (letter == 'D'):
    #     if (letter == 'E'):
    #     if (letter == 'J'):
    #     if (letter == 'K'):
    #
    # if (font_type == font_types.FONT_1):
    #     if (letter == 'A'):
    #     if (letter == 'B'):
    #             # TODO: kitap sayfa 72
    #     if (letter == 'C'):
    #     if (letter == 'D'):
    #     if (letter == 'E'):
    #     if (letter == 'J'):
    #     if (letter == 'K'):


def encoding_letter(method, letter, font_type):
    """

    :param method:
    :param letter:
    :param font_type:
    :return:
    """
    if (method == Target_encode_methods.BINARY):
        encoded_letter = letter_to_binary(letter, font_type)

    if (method == Target_encode_methods.BIPOLAR):
        encoded_letter = letter_to_bipolar(letter, font_type)

    return encoded_letter

def prepare_data_set(letters_list, method):
    data_x = [] #inputs
    data_t = [] #targets
    fonts_list = [ Font_types.FONT_1,  Font_types.FONT_2,  Font_types.FONT_2]

    # FONT_1


    for font in fonts_list:
        for letter in letters_list:
            encoded_letter = encoding_letter(method, letter, font)
            encoded_t = encode_target(letter, method)

            data_x.append(encoded_letter)
            data_t.append(encoded_t)

    return data_x, data_t


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



def update_weights_and_biases:
    #TODO: from here



def train_neural_network(epochs, data_x, data_t, learning_rate=1):
    #TODO: add shuffling option
    #TODO: add other learning rules

    weights = np.zeros((NUMBER_OF_BITS_PER_LETTER,))
    biases = np.zeros((NUMBER_OF_CLASSES, ))

    old_weights = np.zeros((NUMBER_OF_BITS_PER_LETTER,))
    old_biases = np.zeros((NUMBER_OF_CLASSES,))

    for epoch in range(epochs):

        for pattern_idx in range(NUMBER_OF_PATTERNS):
            net = biases + np.dot(weights, data_x[pattern_idx])
            y_out = bipolar_activation(net)

            if (y_out != data_t[pattern_idx]):
                weights, biases = update_weights_and_biases(weights, biases, learning_rate, data_t[pattern_idx], data_x[pattern_idx])
            else:
                continue

        if (weights == old_weights and biases == old_biases):
            break


