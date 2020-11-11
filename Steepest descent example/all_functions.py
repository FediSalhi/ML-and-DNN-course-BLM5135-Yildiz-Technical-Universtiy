################ imports ################
import numpy as np
#########################################

def f_function(x1, x2):
    f = 2*x1*x1 + x1*x2 + x2*x2
    return f

def f_function_grad(x1, x2):
    grad_f