# ML-and-DNN-course-BLM5135-Yildiz-Technical-Universtiy
This repository contains codes related to Machine Learning and Deep Neural Networks course tutored by Assoc. Prof. Sırma YAVUZ in Computer Engineering Department at Yildiz Technical University, Istanbul.

## Assignment 3: A Perceptron To Classify Letters From Different Fonts With Several Output Classes

### Introduction

The purpose of this assignment is to implement a one-layer neural network to classify characters from different fonts with several output classes.

The neural network will be trained to recognize 21 input patterns, with seven characters from three types of font. Characters will be encoded using both &quot;Binary&quot; and &quot;Bipolar&quot; methods, which will be compared in terms of performance and accuracy.

Furthermore, during the training phase, weights and biases will be updated using two different learning rules, &quot;Perceptron&quot; and &quot;Delta&quot;, separately, and results will be presented for different learning rates and data representing methods.

During the testing phase, the model will be firstly evaluated on the train dataset i.e. the data used in the training phase. Then, a noisy test dataset will be created and used to provide a more accurate evaluation of the neural network. This dataset will be only used during the test phase, i.e. weights and biases will not be updated according to this data while the model is being trained.

### 2. Material

Algorithms are implemented using Python 3.7. None of machine learning or deep learning libraries or frameworks (e.g. Tensorflow, Keras, ScikitLearn, …) were used. Numpy library was only used to facilitate the manipulation of multidimensional arrays representing vectors and matrices. Note that all algebraic and logical operations such as vector multiplication and comparison were manually implemented i.e. without using any other non-standard Python library.

### 3. Method

We trained the neural network to classify input vectors as belonging or not belonging to seven classes representing A, B, C, D, E, J, and K characters. The target value for each pattern is either 1 for belonging or -1 or 0 for not belonging. This value is only one component of the target vector consisting of 7 components, one for each category.

During the training phase, the neural network is fed with a total of 21 characters belonging to 3 different types of font, with 63 data points per each character (Figure 2). These data points are either 1, 0, or -1 depending on the dataset encoding method.


![](RackMultipart20201113-4-hzwqx5_html_4e1776b484f175a3.png)

Step 4. Compute activation of each output unit, j = 1, 2, .., 7

![](RackMultipart20201113-4-hzwqx5_html_eb89a56ca4df31ea.png)

Step 5. Update biases and weight, j = 1, 2, ..,7; i= 1,2, …, 63

If tj ≠ yj, then

![](RackMultipart20201113-4-hzwqx5_html_b90970d61a434e7.png)

Else, biases and weights remain unchanged.

Step 6. Test for stopping condition:

If no weights changes in occurred in Step 2. Stop, otherwise, continue.

# 4. Method Implementation and Architecture

Training and test algorithms are implemented in 4 Python (with a .py extension) code files. The diagram shown in Figure 3 illustrates the general architecture of the source code files.

![](RackMultipart20201113-4-hzwqx5_html_ed7d35ec2ad4ce44.png)

_Figure 3 Source code general architecture_

Details of each file are given in Table 2.

_Table 2 Source code files details_

| Source code file name | Task |
| --- | --- |
| main.py | Interacts with the user to choose an encoding method (bipolar/binary), start training and evaluate the model. |
| main\_screen.py | Contains general information and user guide. |
| functions.py | Contains all functions performing main tasks. |
| constants.py | Contains all constants such as learning rate. |

# 5. Results

## 5.1 Learning Rule: Perceptron

### 5.1.1 Evaluation Dataset: Training Dataset

Three key parameters were changed one at a time during the training phase. These parameters are given in Table 3. Afterward, the neural network model was evaluated in terms of performance, measuring the training duration and total epochs and in terms of binary accuracy separately for each class. The results are given in Table 4 in which values were represented in blue and green colors for binary and bipolar respectively.

_Table 3 Training parameters_

| Trainingparameter | Values |
| --- | --- |
| Encoding method |
- Binary
- Bipolar
 |
| Learning rate |
- 0.1
- 1
 |
| Activation function threshold |
- 0
- 3
 |

An example of the main.py script output is shown below in Figures 4, 5, 6, and 7.

![](RackMultipart20201113-4-hzwqx5_html_8855447f7cd293c4.png)

_Figure 4 main.py script ouput: main screen_

![](RackMultipart20201113-4-hzwqx5_html_5c1491ffb3d7789a.png)

_Figure 5 main.py script output: model training_

![](RackMultipart20201113-4-hzwqx5_html_fd49caab6aec6b82.png)

_Figure 6 main.py script output: Training parameters_

![](RackMultipart20201113-4-hzwqx5_html_9f99e1e3c2d1f452.png)

_Figure 7 main.py script output: model evaluation – Perceptron_

_Table 4 Results on training dataset corresponding to different parameters – Perceptron learning rule_

| **Training parameters** | **Evaluation metrics** |
| --- | --- |
| **Learning rate** | **Data encoding / Activation function** | **Activation function threshold** | **Binary accuracies** | **Total epochs** | **Training duration (s)** |
| 0.1 | Bipolar | 0 | 100% accuracy for each category | 4 | 0.16 |
| 1 | Bipolar | 0 | 100% accuracy for each category | 4 | 0.17 |
| 0.1 | Binary | 0 | 14.29% accuracy for each category | 1 | 0.04 |
| 1 | Binary | 0 | 14.29% accuracy for each category | 1 | 0.04 |
| 0.1 | Bipolar | 3 | 100% accuracy for each category | 4 | 0.17 |
| 1 | Bipolar | 3 | 100% accuracy for each category | 5 | 0.21 |
| 0.1 | Binary | 3 | 71.43% for A and B categories.95,24% for C category80,95% for D category66,67% for E category100% for J category57,14% for K category | 2 | 0.09 |
| 1 | Binary | 3 | Class A:14.29 % for A, B, C, D, E, and K categories.19,05 % for J category | 2 | 0.09 |

### 5.1.2 Evaluation Dataset: Noisy Test Dataset

To evaluate the neural network more accurately, we have tested it with a dataset consisting of input patterns with a few of its pixels changed. The pixels where the input pattern differs from the training pattern are indicated by @ for a pixel that is &quot;on&quot; now but was &quot;off&quot; in the training patter, and O for a pixel that is &quot;off&quot; now but was originally &quot;on&quot; (Figure 8)[4]. Results corresponding to different parameters are presented in Table 5.

![](RackMultipart20201113-4-hzwqx5_html_2712bfc5ef8bcc84.png)

_Figure 8. Noisy test dataset [5]_

_Table 5 Results on noisy test dataset corresponding to different parameters – Perceptron learning rule_

| **Training parameters** | **Evaluation metrics** |
| --- | --- |
| **Learning rate** | **Data encoding / Activation function** | **Activation function threshold** | **Binary accuracies** | **Total epochs** | **Training duration (s)** |
| 0.1 | Bipolar | 0 | 100% accuracy for A, B, C, J, and D categories.95.23% for E category80.95% for K category | 4 | 0.17 |
| 1 | Bipolar | 0 | 100% accuracy for A, B, C, J, and D categories.95.23% for E category80.95% for K category | 4 | 0.16 |
| 0.1 | Binary | 0 | 14.29% accuracy for each category | 1 | 0.04 |
| 1 | Binary | 0 | 14.29% accuracy for each category | 1 | 0.04 |
| 0.1 | Bipolar | 3 | 100% accuracy for each category | 4 | 0.16 |
| 1 | Bipolar | 3 | 100% accuracy for A, B, C, D, J, and K categories.90.47% for E category. | 5 | 0.21 |
| 0.1 | Binary | 3 | 71.43% for A and B categories.95,24% for C category80,95% for D category66,67% for E category100% for J category57,14% for K category | 2 | 0.09 |
| 1 | Binary | 3 | Class A:14.29 % for A, B, C, D, E, and K categories.19,05 % for J category | 2 | 0.09 |

## 5.2 Learning Rule: Delta

The same algorithm was used to train the neural network but this time, instead of Perceptron, weights and biases were updated using Delta learning rule defined by [6]

![](RackMultipart20201113-4-hzwqx5_html_216c699b0845b396.png):

### 5.2.1 Evaluation Dataset: Training Dataset

The program output for learning rate = 1, activation threshold = 0, and encoding method = bipolar parameters is shown in Figures 9-10. All results are summarized in Table 6 for training datset.

![](RackMultipart20201113-4-hzwqx5_html_27aa4304feb7dd8c.png)

_Figure 9 main.py script output for Delta training rule_

![](RackMultipart20201113-4-hzwqx5_html_1a2539277c46b10d.png)

_Figure 10 main.py script output: model evaluation – training dataset - Delta_

_Table 6. Results on training dataset corresponding to different parameters – Delta learning rule_

| **Training parameters** | **Evaluation metrics** |
| --- | --- |
| **Learning rate** | **Data encoding / Activation function** | **Activation function threshold** | **Binary accuracies** | **Total epochs** | **Training duration (s)** |
| 0.1 | Bipolar | 0 | 100% accuracy for each class.
 | 4 | 0.17 |
| 1 | Bipolar | 0 | 100% accuracy for each class
 | 4 | 0.16 |
| 0.1 | Binary | 0 | 14.29% accuracy for each category | 1 | 0.04 |
| 1 | Binary | 0 | 14.29% accuracy for each category | 1 | 0.04 |
| 0.1 | Bipolar | 3 | 100% accuracy for each category | 4 | 0.18 |
| 1 | Bipolar | 3 | 100% accuracy for each class
 | 5 | 0.22 |
| 0.1 | Binary | 3 | 71.43% for A and B categories.95,24% for C category80,95% for D category66,67% for E category100% for J category57,14% for K category | 2 | 0.09 |
| 1 | Binary | 3 | Class A:14.29 % for A, B, C, D, E, and K categories.19,05 % for J category | 2 | 0.09 |

### 5.1.2 Evaluation dataset: Noisy Test Dataset

All results are summarized in Table 7 for noisy test datset.

_Table 7 Results on noisy test dataset corresponding to different parameters – Delta learning rule_

| **Training parameters** | **Evaluation metrics** |
| --- | --- |
| **Learning rate** | **Data encoding / Activation function** | **Activation function threshold** | **Binary accuracies** | **Total epochs** | **Training duration (s)** |
| 0.1 | Bipolar | 0 | 100% accuracy for A, B, C, J, and D categories.95.23% for E category80.95% for K category | 4 | 0.17 |
| 1 | Bipolar | 0 | 100% accuracy for A, B, C, J, and D categories.95.23% for E category80.95% for K category | 4 | 0.16 |
| 0.1 | Binary | 0 | 14.29% accuracy for each category | 1 | 0.04 |
| 1 | Binary | 0 | 14.29% accuracy for each category | 1 | 0.04 |
| 0.1 | Bipolar | 3 | 100% accuracy for each category | 4 | 0.16 |
| 1 | Bipolar | 3 | 100% accuracy for A, B, C, D, J, and K categories.90.47% for E category. | 5 | 0.21 |
| 0.1 | Binary | 3 | 71.43% for A and B categories.95,24% for C category80,95% for D category66,67% for E category100% for J category57,14% for K category | 2 | 0.09 |
| 1 | Binary | 3 | Class A:14.29 % for A, B, C, D, E, and K categories.19,05 % for J category | 2 | 0.09 |

_Figure 11 Comparison of binary and bipolar data representation using sperately training and noisy test dataset. Comparison metric is binary accuracy for learning rule = perceptron, learning rate = 1, activation function threshold = 0_

**6 ![](RackMultipart20201113-4-hzwqx5_html_9fe5886bb6d6c4d1.png) . Conclusion**

From Table 4, 5, 6, and 7, we can infer that:

## 6.1 Effect of data representation: Binary – Bipolar

This parameter obviously had a critical effect on the overall binary accuracy of each class. With bipolar representation, the model has achieved 100% accuracy for each class using the training dataset and 100% accuracy for A, B, C, J, and D classes, 95.23% for E class, and 80.95% for K class using noisy test dataset, which means that model is still able to classify most of the noisy characters. On the other side, with binary representation, the model failed to classify even the characters used in the training phase. In fact, 0 values will tend to deactivate the neurons which may lead to a wrong classification. Representing &quot;off&quot; data points with -1&#39;s instead of 0&#39;s will head off this case.

## 6.2 Effect of learning rate: 0.1 – 1

For this problem, neither the accuracy nor the performance (training time) of the model has changed for different learning rates. The effect of this parameter may be more visible on larger datasets.

## 6.3 Effect of activation function threshold: 0 – 3

Although bias values were used, setting the activation function threshold to 3 was beneficial for the binary dataset and higher accuracies were obtained compared to the results obtained when setting this value to 0.

## 6.4 Effect of learning rule: Perceptron – Delta

As we know, theoretically speaking, the Delta learning rule must be more effective compared to simple perceptron learning rule since it moves the training parameters i.e. weights and biases depending on the amount of error. But for this simple problem with limited parameters and network size, we didn&#39;t detect any difference when the perceptron training rule is replaced with the Delta training rule.

# 7. Future Work Suggestions

Adding a hidden layer to the neural network is supposed to enhance its binary accuracy for test data not used during the training phase. This hypothesis can be verified in future work.

# 8. References

[1][2][3][4][5][6] Laurene Fausett, &quot;Fundamentals of neural networks: architectures, algorithms, and applications&quot; July 1994.
