################################################## imports #############################################################
from enum import Enum

################################################## enumerations ########################################################

class Font_types(Enum):
    FONT_1 = 1
    FONT_2 = 2
    FONT_3 = 3

class Encode_methods(Enum):
    BINARY  = 1
    BIPOLAR = 2

class Learning_rules(Enum):
    PERCEPTRON = 1
    DELTA = 2 # not implemented yet

################################################## general parameters ##################################################
NUMBER_OF_LETTERS_PER_FONT = 7
NUMBER_OF_FONTS = 3
DATASET_TOTAL_NUMBER_OF_LETTERS = NUMBER_OF_LETTERS_PER_FONT * NUMBER_OF_FONTS
NUMBER_OF_BITS_PER_LETTER = 63
NUMBER_OF_CLASSES = 7 # A, B, C, D, E, J, K classes

CLASS_A_INDEX = 0
CLASS_B_INDEX = 1
CLASS_C_INDEX = 2
CLASS_D_INDEX = 3
CLASS_E_INDEX = 4
CLASS_J_INDEX = 5
CLASS_K_INDEX = 6

################################################## test constants ######################################################
TEST_DATASET_TOTAL_NUMBER_OF_LETTERS = 21


################################################## training parameters #################################################
LEARNING_RATE = 1
WEIGHTS_INITIAL_VALUES_COEF = 0
BIASES_INITIAL_VALUES_COEF = 0
# LEARNING_RULE = Learning_rules.PERCEPTRON
LEARNING_RULE = Learning_rules.DELTA
LETTERS_ENCODING_METHOD = Encode_methods.BIPOLAR
ACTIVATION_FUNCTION_THRESHOLD = 0

############################################## characters definition (inputs) ##########################################

#------------- BIPOLAR FONT1 -------------

A_FONT_1_BIPOLAR = [-1,-1,1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1,1,1,1,1,-1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,1,1,1,-1,1,1,1]
B_FONT_1_BIPOLAR = [1,1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,1,1,1,1,1,1,-1]
C_FONT_1_BIPOLAR = [-1,-1,1,1,1,1,1,-1,1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,1,1,1,1,-1]
D_FONT_1_BIPOLAR = [1,1,1,1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,1,1,1,1,1,-1,-1]
E_FONT_1_BIPOLAR = [1,1,1,1,1,1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,1,1,1,1,1,1,1]
J_FONT_1_BIPOLAR = [-1,-1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1]
K_FONT_1_BIPOLAR = [1,1,1,-1,-1,1,1,-1,1,-1,-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,1,1,1,-1,-1,1,1]

#------------- BIPOLAR FONT2 -------------

A_FONT_2_BIPOLAR = [-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,1,1,1,1,1,-1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1]
B_FONT_2_BIPOLAR = [1,1,1,1,1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,-1]
C_FONT_2_BIPOLAR = [-1,-1,1,1,1,-1,-1,-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1]
D_FONT_2_BIPOLAR = [1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,1,-1,1,1,1,1,1,-1,-1]
E_FONT_2_BIPOLAR = [1,1,1,1,1,1,1, 1,-1,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1,1,1,1,1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1,1,1,1,1,1,1]
J_FONT_2_BIPOLAR = [-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1]
K_FONT_2_BIPOLAR = [1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,-1,1,-1]

#------------- BIPOLAR FONT3 -------------

A_FONT_3_BIPOLAR = [-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,1,1,1,1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,1,-1,-1,-1,1,1]
B_FONT_3_BIPOLAR = [1,1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,1,1,1,1,1,1,-1]
C_FONT_3_BIPOLAR = [-1,-1,1,1,1,-1,1,-1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1]
D_FONT_3_BIPOLAR = [1,1,1,1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,1,1,1,1,1,-1,-1]
E_FONT_3_BIPOLAR = [1,1,1,1,1,1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,1,-1,-1,-1,1,1,1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,1,1,1,1,1,1,1]
J_FONT_3_BIPOLAR = [-1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1]
K_FONT_3_BIPOLAR = [1,1,1,-1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,1,1,1,-1,-1,1,1]


#------------- BINARY FONT1 -------------

A_FONT_1_BINARY = [0,0,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,1,1,1,1,1,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,1,1,1,0,1,1,1]
B_FONT_1_BINARY = [1,1,1,1,1,1,0,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,1,1,1,1,0,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,1,1,1,1,1,1,0]
C_FONT_1_BINARY = [0,0,1,1,1,1,1,0,1,0,0,0,0,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,1,1,1,1,0]
D_FONT_1_BINARY = [1,1,1,1,1,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,1,0,1,1,1,1,1,0,0]
E_FONT_1_BINARY = [1,1,1,1,1,1,1,0,1,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,1,0,0,0,0,1,1,1,0,0,0,0,1,0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,1,1,1,1,1,1,1,1]
J_FONT_1_BINARY = [0,0,0,1,1,1,1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,0,1,1,1,0,0]
K_FONT_1_BINARY = [1,1,1,0,0,1,1,0,1,0,0,1,0,0,0,1,0,1,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,0,1,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,1,0,1,1,1,0,0,1,1]

#------------- BINARY FONT2 -------------

A_FONT_2_BINARY = [0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,1,0,0,0,1,0,0,1,1,1,1,1,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0]
B_FONT_2_BINARY = [1,1,1,1,1,1,0,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,1,1,1,1,1,0,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,1,1,1,1,1,0]
C_FONT_2_BINARY = [0,0,1,1,1,0,0,0,1,0,0,0,1,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,1,0,0,0,1,1,1,0,0]
D_FONT_2_BINARY = [1,1,1,1,1,0,0,1,0,0,0,0,1,0,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,1,0,1,1,1,1,1,0,0]
E_FONT_2_BINARY = [1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,1,1,1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,1,1,1,1,1]
J_FONT_2_BINARY = [0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,0,1,1,1,0,0]
K_FONT_2_BINARY = [1,0,0,0,0,1,0,1,0,0,0,1,0,0,1,0,0,1,0,0,0,1,0,1,0,0,0,0,1,1,0,0,0,0,0,1,0,1,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,1,0]

#------------- BINARY FONT3 -------------

A_FONT_3_BINARY = [0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,1,0,0,0,1,0,0,1,1,1,1,1,0,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,1,0,0,0,1,1]
B_FONT_3_BINARY = [1,1,1,1,1,1,0,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,1,1,1,1,0,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,1,1,1,1,1,1,0]
C_FONT_3_BINARY = [0,0,1,1,1,0,1,0,1,0,0,0,1,1,1,0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,1,0,0,0,1,1,1,0,0]
D_FONT_3_BINARY = [1,1,1,1,1,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,1,0,1,1,1,1,1,0,0]
E_FONT_3_BINARY = [1,1,1,1,1,1,1,0,1,0,0,0,0,1,0,1,0,0,1,0,0,0,1,1,1,1,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,1,1,1,1,1,1,1,1]
J_FONT_3_BINARY = [0,0,0,0,1,1,1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,1,1,1,0,0]
K_FONT_3_BINARY = [1,1,1,0,0,1,1,0,1,0,0,0,1,0,0,1,0,0,1,0,0,0,1,0,1,0,0,0,0,1,1,0,0,0,0,0,1,0,1,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,1,0,1,1,1,0,0,1,1]

############################################## targets definition (outputs) ############################################

A_TARGET_BIPOLAR = [1, -1, -1, -1, -1, -1, -1]
A_TARGET_BINARY  = [1, 0, 0, 0, 0, 0, 0]

B_TARGET_BIPOLAR = [-1, 1, -1, -1, -1, -1, -1]
B_TARGET_BINARY  = [0, 1, 0, 0, 0, 0, 0]

C_TARGET_BIPOLAR = [-1, -1, 1, -1, -1, -1, -1]
C_TARGET_BINARY  = [0, 0, 1, 0, 0, 0, 0]

D_TARGET_BIPOLAR = [-1, -1, -1, 1, -1, -1, -1]
D_TARGET_BINARY  = [0, 0, 0, 1, 0, 0, 0]

E_TARGET_BIPOLAR = [-1, -1, -1, -1, 1, -1, -1]
E_TARGET_BINARY  = [0, 0, 0, 0, 1, 0, 0]

J_TARGET_BIPOLAR = [-1, -1, -1, -1, -1, 1, -1]
J_TARGET_BINARY  = [0, 0, 0, 0, 0, 1, 0]

K_TARGET_BIPOLAR = [-1, -1, -1, -1, -1, -1, 1]
K_TARGET_BINARY  = [0, 0, 0, 0, 0, 0, 1]



######################################### noisy characters definition (inputs) #########################################

#------------- BIPOLAR FONT1 -------------

A_FONT_1_NOISY_BIPOLAR = [-1,-1,1,1,-1,-1,-1,
                    -1,-1,1,1,-1,-1,-1,
                    -1,-1,-1,1,-1,-1,-1,
                    -1,-1,1,-1,1,1,-1,
                    -1,-1,1,-1,1,-1,-1,
                    -1,1,1,1,1,1,-1,
                    -1,1,-1,1,-1,1,-1,
                    -1,1,-1,-1,-1,1,-1,
                    1,-1,1,-1,1,-1,1]

B_FONT_1_NOISY_BIPOLAR = [1,1,1,1,1,1,-1,
                    -1,-1,-1,-1,-1,-1,1,
                    -1,1,-1,-1,-1,-1,1,
                    -1,1,1,-1,-1,-1,1,
                    -1,1,1,1,1,1,-1,
                    -1,1,-1,1,-1,-1,1,
                    -1,1,1,-1,-1,-1,1,
                    -1,1,-1,-1,-1,-1,1,
                    1,1,1,1,1,1,-1]

C_FONT_1_NOISY_BIPOLAR = [-1,-1,1,1,1,1,1,
                    1,1,-1,-1,1,-1,1,
                    1,-1,-1,-1,-1,-1,-1,
                    1,1,-1,-1,-1,-1,-1,
                    -1,-1,-1,-1,-1,-1,-1,
                    1,-1,-1,-1,-1,-1,-1,
                    1,-1,-1,-1,-1,-1,-1,
                    -1,1,-1,-1,1,-1,1,
                    -1,-1,1,1,1,1,-1]

D_FONT_1_NOISY_BIPOLAR = [1,1,1,1,1,-1,-1,
                    -1,1,-1,1,-1,1,-1,
                    -1,1,-1,-1,-1,-1,-1,
                    -1,1,1,-1,-1,-1,1,
                    -1,1,-1,-1,-1,-1,-1,
                    -1,1,-1,-1,-1,-1,1,
                    -1,1,-1,-1,-1,-1,1,
                    -1,1,1,-1,-1,1,-1,
                    1,1,1,1,1,-1,-1]

E_FONT_1_NOISY_BIPOLAR = [1,1,-1,1,1,1,1,
                    -1,1,-1,-1,-1,-1,1,
                    -1,1,-1,-1,-1,-1,-1,
                    -1,1,-1,1,-1,-1,-1,
                    -1,1,-1,1,1,-1,-1,
                    -1,1,-1,1,-1,-1,-1,
                    -1,1,-1,-1,-1,-1,-1,
                    1,1,-1,-1,-1,-1,1,
                    1,1,1,1,1,1,1]

J_FONT_1_NOISY_BIPOLAR = [-1,-1,-1,1,1,-1,1,
                    -1,-1,-1,-1,-1,1,-1,
                    -1,-1,-1,-1,-1,1,-1,
                    -1,-1,-1,-1,-1,1,-1,
                    -1,-1,-1,-1,-1,1,-1,
                    -1,-1,-1,-1,-1,-1,-1,
                    -1,1,-1,-1,-1,1,-1,
                    -1,-1,-1,-1,1,1,-1,
                    -1,-1,1,1,1,-1,-1]

K_FONT_1_NOISY_BIPOLAR = [1,-1,1,-1,-1,1,1,
                    -1,1,-1,-1,1,-1,-1,
                    -1,1,-1,1,-1,-1,-1,
                    -1,1,1,-1,-1,-1,-1,
                    -1,1,1,-1,-1,-1,-1,
                    -1,1,-1,1,-1,-1,-1,
                    -1,1,-1,-1,1,1,-1,
                    -1,1,1,-1,-1,1,-1,
                    1,1,-1,-1,-1,1,1]

#------------- BIPOLAR FONT2 -------------

A_FONT_2_BIPOLAR = [-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,1,1,1,1,1,-1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1]
B_FONT_2_BIPOLAR = [1,1,1,1,1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,-1]
C_FONT_2_BIPOLAR = [-1,-1,1,1,1,-1,-1,-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1]
D_FONT_2_BIPOLAR = [1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,1,-1,1,1,1,1,1,-1,-1]
E_FONT_2_BIPOLAR = [1,1,1,1,1,1,1, 1,-1,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1,1,1,1,1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1,1,1,1,1,1,1]
J_FONT_2_BIPOLAR = [-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1]
K_FONT_2_BIPOLAR = [1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,-1,1,-1]

#------------- BIPOLAR FONT3 -------------

A_FONT_3_BIPOLAR = [-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,1,1,1,1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,1,-1,-1,-1,1,1]
B_FONT_3_BIPOLAR = [1,1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,1,1,1,1,1,1,-1]
C_FONT_3_BIPOLAR = [-1,-1,1,1,1,-1,1,-1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1]
D_FONT_3_BIPOLAR = [1,1,1,1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,1,1,1,1,1,-1,-1]
E_FONT_3_BIPOLAR = [1,1,1,1,1,1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,1,-1,-1,-1,1,1,1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,1,1,1,1,1,1,1]
J_FONT_3_BIPOLAR = [-1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1]
K_FONT_3_BIPOLAR = [1,1,1,-1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,1,1,1,-1,-1,1,1]