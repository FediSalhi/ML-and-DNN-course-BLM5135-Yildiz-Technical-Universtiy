# ML-and-DNN-course-BLM5135-Yildiz-Technical-Universtiy
This repository contains codes related to Machine Learning and Deep Neural Networks course tutored by Assoc. Prof. Sırma YAVUZ in Computer Engineering Department at Yildiz Technical University, Istanbul.

## Assignment 3: A Perceptron To Classify Letters From Different Fonts With Several Output Classes

### 1.Introduction

The purpose of this assignment is to implement a one-layer neural network to classify characters from different fonts with several output classes.

The neural network will be trained to recognize 21 input patterns, with seven characters from three types of font. Characters will be encoded using both &quot;Binary&quot; and &quot;Bipolar&quot; methods, which will be compared in terms of performance and accuracy.

Furthermore, during the training phase, weights and biases will be updated using two different learning rules, &quot;Perceptron&quot; and &quot;Delta&quot;, separately, and results will be presented for different learning rates and data representing methods.

During the testing phase, the model will be firstly evaluated on the train dataset i.e. the data used in the training phase. Then, a noisy test dataset will be created and used to provide a more accurate evaluation of the neural network. This dataset will be only used during the test phase, i.e. weights and biases will not be updated according to this data while the model is being trained.

### 2. Material

Algorithms are implemented using Python 3.7. None of machine learning or deep learning libraries or frameworks (e.g. Tensorflow, Keras, ScikitLearn, …) were used. Numpy library was only used to facilitate the manipulation of multidimensional arrays representing vectors and matrices. Note that all algebraic and logical operations such as vector multiplication and comparison were manually implemented i.e. without using any other non-standard Python library.

### 3. Method

We trained the neural network to classify input vectors as belonging or not belonging to seven classes representing A, B, C, D, E, J, and K characters. The target value for each pattern is either 1 for belonging or -1 or 0 for not belonging. This value is only one component of the target vector consisting of 7 components, one for each category.

During the training phase, the neural network is fed with a total of 21 characters belonging to 3 different types of font, with 63 data points per each character (Figure 2). These data points are either 1, 0, or -1 depending on the dataset encoding method.

### 4. Method Implementation and Architecture

Training and test algorithms are implemented in 4 Python (with a .py extension) code files. The diagram shown in Figure 3 illustrates the general architecture of the source code files.

| Source code file name | Task |
| --- | --- |
| main.py | Interacts with the user to choose an encoding method (bipolar/binary), start training and evaluate the model. |
| main\_screen.py | Contains general information and user guide. |
| functions.py | Contains all functions performing main tasks. |
| constants.py | Contains all constants such as learning rate. |

### 5. Conclusions
#### 5.1 Effect of data representation: Binary – Bipolar

This parameter obviously had a critical effect on the overall binary accuracy of each class. With bipolar representation, the model has achieved 100% accuracy for each class using the training dataset and 100% accuracy for A, B, C, J, and D classes, 95.23% for E class, and 80.95% for K class using noisy test dataset, which means that model is still able to classify most of the noisy characters. On the other side, with binary representation, the model failed to classify even the characters used in the training phase. In fact, 0 values will tend to deactivate the neurons which may lead to a wrong classification. Representing &quot;off&quot; data points with -1&#39;s instead of 0&#39;s will head off this case.

#### 5.2 Effect of learning rate: 0.1 – 1

For this problem, neither the accuracy nor the performance (training time) of the model has changed for different learning rates. The effect of this parameter may be more visible on larger datasets.

#### 5.3 Effect of activation function threshold: 0 – 3

Although bias values were used, setting the activation function threshold to 3 was beneficial for the binary dataset and higher accuracies were obtained compared to the results obtained when setting this value to 0.

#### 5.4 Effect of learning rule: Perceptron – Delta

As we know, theoretically speaking, the Delta learning rule must be more effective compared to simple perceptron learning rule since it moves the training parameters i.e. weights and biases depending on the amount of error. But for this simple problem with limited parameters and network size, we didn&#39;t detect any difference when the perceptron training rule is replaced with the Delta training rule.

### 6. References

[1][2][3][4][5][6] Laurene Fausett, &quot;Fundamentals of neural networks: architectures, algorithms, and applications&quot; July 1994.
