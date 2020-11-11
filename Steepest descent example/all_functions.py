################ imports ################
import numpy as np
from constants import *
#########################################

def f_function(x1, x2):
    f = 2*x1*x1 + x1*x2 + x2*x2
    return f

def f_function_grad(x1, x2):
    grad_f = [4*x1 + x2, x1 + 2*x2]
    return grad_f

def compute_norm(X):
    norm = np.sqrt(X[0]*X[0] + X[1]*X[1])
    return norm

def verify_min_point_or_not(X0):
    grad = f_function_grad(X0[0], X0[1])
    norm_grad = compute_norm(grad)
    if (norm_grad < EPSILON_1):
        result = True
    else:
        result = False
    return result

def verify_chosen_point(X1, X2):
    diff_points = [X1[0] - X2[0], X1[1] - X2[1]]
    distance_points = compute_norm(diff_points)

    diff_func = f_function(X1[0], X1[1]) - f_function(X2[0], X2[1])
    abs_diff_func = abs(diff_func)
    if (distance < EPSILON_2 and abs_diff_func < EPSILON_2):
        chosen_point_verified = True
    else:
        chosen_point_verified = False

    return chosen_point_verified

def compute_direction(X):
    direction = -f_function_grad(X[0], X[1])
    return direction

def compute_step(X):
    """
    constant step to simplify algorithm
    :param X:
    :return:
    """
    return MOVING_STEP

