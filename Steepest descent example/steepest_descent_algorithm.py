################################################## imports #############################################################
import numpy as np
from constants import *
from all_functions import *
########################################################################################################################

# initial values
is_a_valid_point = True
is_a_min_point = False
k = 0
X0 = START_POINT

while (k < MAX_ITERATIONS):
    k += 1
    if (is_a_valid_point):
        is_a_min_point = verify_min_point_or_not(X0)
    if (is_a_min_point):
        print('{} is a minimum of f(x) function'.format(X0))
        break
    else:
        # choose direciton
        direction = compute_direction(X0)
        step = compute_step(X0)
        # move down to next point
        X1 = np.add(X0, np.multiply(step, direction))
        is_a_valid_point = verify_chosen_point(X0, X1)
        X0 = X1
