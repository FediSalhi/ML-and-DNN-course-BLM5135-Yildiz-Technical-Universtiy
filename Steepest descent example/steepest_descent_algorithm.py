################ imports ################
import numpy as np
from constants import *
from all_functions import *
#########################################


k = 0
X0 = BASKAGIC_NOKTASI
while (k < MAX_ITERATIONS):
    is_a_min_point = verify_min_point_or_not(X0)
    if (is_a_min_point):
        print('{} is a minimum'.format(X0))
        break
    else:
        # choose direciton
        direction = compute_direction(X0)
        step = compute_step(X0)
        # choose another point
        X1 = np.add(X0, step*direction)
        is_a_valid_point = verify_chosen_point(X0, X2)
