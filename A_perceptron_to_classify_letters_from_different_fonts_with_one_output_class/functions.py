################ imports ################
import numpy as np
from global_vars import *
#########################################


def letter_to_binary(letter, font_type):
    """
    :param letter:
    :param font_type:
    :return:
    """
    if (font_type == font_types.FONT_1):
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

        if (letter == 'B'):
            # TODO: kitap sayfa 72
        if (letter == 'C'):
        if (letter == 'D'):
        if (letter == 'E'):
        if (letter == 'J'):
        if (letter == 'K'):

    if (font_type == font_types.FONT_1):
        if (letter == 'A'):
        if (letter == 'B'):
                # TODO: kitap sayfa 72
        if (letter == 'C'):
        if (letter == 'D'):
        if (letter == 'E'):
        if (letter == 'J'):
        if (letter == 'K'):

    if (font_type == font_types.FONT_1):
        if (letter == 'A'):
        if (letter == 'B'):
                # TODO: kitap sayfa 72
        if (letter == 'C'):
        if (letter == 'D'):
        if (letter == 'E'):
        if (letter == 'J'):
        if (letter == 'K'):

def prepare_data_set(letters_list):
    data_x = [] #inputs
    data_t = [] #targets
    fonts_list = [ Font_types.FONT_1,  Font_types.FONT_2,  Font_types.FONT_2]

    # FONT_1


    for font in fonts_list:
        for letter in letters_list:
            binary_letter = letter_to_binary(letter, font)
            data_x.append(binary_letter)
            if (letter == 'A'):
                data_t.append(1)
            else:
                data_t.append(-1)

    return data_x, data_t

def train_neural_network(lerning_rule, epochs, dataset):
