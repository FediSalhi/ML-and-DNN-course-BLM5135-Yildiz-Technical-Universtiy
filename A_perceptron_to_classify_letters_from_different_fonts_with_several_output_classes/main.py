################################################## imports #############################################################
from constants import *
from functions import *
from main_screen import *
########################################################################################################################

if __name__ == '__main__':

    show_main_screen()
    show_user_guide()

    encoding_method_command_is_valid = False
    train_model_command_is_valid = False
    evaluate_model_command_is_valid = False
    test_model_command_is_valid = True

    while (encoding_method_command_is_valid == False):
        encoding_method_command = input("Which encoding method do you want to use? (BINARY/BIPOLAR):")
        if (encoding_method_command == "BINARY"):
            encoding_method = Encode_methods.BINARY
            encoding_method_command_is_valid = True
        elif (encoding_method_command == "BIPOLAR"):
            encoding_method = Encode_methods.BIPOLAR
            encoding_method_command_is_valid = True
        else:
            print("You didn't choose a valid encoding method, please try agian!")




    while (train_model_command_is_valid == False):
        start_training_confirmation_command = input("{} will be used to prepare the dataset and start training, continue ? (Yes/No)".format(str(encoding_method)[15:-1]))
        if (start_training_confirmation_command == 'Yes'):
            train_model_command_is_valid = True
            print("training model ...")
            data_inputs, data_targets = prepare_dataset(encoding_method)
            weights, biases, epochs, training_duration = train_neural_network(LEARNING_RULE, encoding_method, data_inputs, data_targets, LEARNING_RATE)
            print("The model is successfully trained.")
        elif (start_training_confirmation_command == 'No'):
            train_model_command_is_valid = True
            print("Well, see you soon ...")
            exit()
        else:
            print("You didn't choose a valid encoding method, please try agian !")


    # get a detailed evaluation of the trained model
    evaluate_model(weights, biases, epochs, data_inputs, training_duration, encoding_method)

    # use this if you want get a prediction for a single character
    # prediction = get_prediction_bipolar(weights, biases, np.array(A_FONT_1_BIPOLAR))
