################ imports ################
from enum import Enum
#########################################


class Font_types(Enum):
    FONT_1 = 1
    FONT_2 = 2
    FONT_3 = 3

NUMBER_OF_LETTERS_PER_FONT = 7
NUMBER_OF_FONTS = 3
NUMBER_OF_PATTERNS = NUMBER_OF_LETTERS_PER_FONT * NUMBER_OF_FONTS