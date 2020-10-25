################ imports ################
from enum import Enum
#########################################


class Font_types(Enum):
    FONT_1 = 1
    FONT_2 = 2
    FONT_3 = 3

class Target_encode_methods(Enum):
    BINARY  = 1
    BIPOLAR = 2

class Learning_rules(Enum):
    PERCEPTRON = 1
    DELTA = 2
    #TODO: add other learning rules here


NUMBER_OF_LETTERS_PER_FONT = 7
NUMBER_OF_FONTS = 3
NUMBER_OF_PATTERNS = NUMBER_OF_LETTERS_PER_FONT * NUMBER_OF_FONTS
NUMBER_OF_BITS_PER_LETTER = 63
NUMBER_OF_CLASSES = 7 # A, B, C, D, E, J, K classes