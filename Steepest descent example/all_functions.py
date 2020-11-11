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
    norm = np.sqrt(X[0]*X[0] + X[0]*X[0])
    return norm

def verify_min_point_or_not(X0):
    grad = f_function_grad(X0[0], X0[1])
    norm_grad = compute_norm(grad)
    if (norm_grad < EPSILON_1):
        result = True
    else:
        result = False
    return result

