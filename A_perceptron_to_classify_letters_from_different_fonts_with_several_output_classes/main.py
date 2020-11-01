################################################## imports #############################################################
from constants import *
from functions import *
########################################################################################################################

if __name__ == '__main__':

    data_inputs, data_targets = prepare_dataset(LETTERS_ENCODING_METHOD)
    weights, biases = train_neural_network(LEARNING_RULE, data_inputs, data_targets, LEARNING_RATE)
    prediction = get_prediction(weights, biases, np.array(K_FONT_1_BIPOLAR))
    print(prediction)