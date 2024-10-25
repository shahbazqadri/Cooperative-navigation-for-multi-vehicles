'''
Firefly

Function to initialize agent's poses.
Input:
    swarmsize --  swarmsize.
    x -- An array to store  agents' x positions, the length of this array is the same as swarmsize
    y -- An array to store agents' y positions, the length of this array is the same as swarmsize
    theta -- An array to store agents' orientations, the length of this is the same as swarmsize

Usage:
    Usr can configure an agent's initial x, y, theta by modifying the value of the corresponding element in array x, y, and theta.
    For example, initialize agent 0's pose to x = 0, y = 1, theta = 2:
    x[0] = 0
    y[0] = 1
    theta[0] = 2

Constraints to be considered:
    x -- the value should range between -2.5 to 2.5.
    y -- the value should range between -1.5 to 1.5.
    theta -- the value should range between -pi to pi.

    The minimal pairwise inter-agent distance should be greater than 0.12

def init(swarmsize, x, y, theta, a_ids):
    import math
    import random
    for i in range(swarmsize):
        x[i] = (i % 16 ) * 0.11-1
        y[i] = (i / 16 ) * 0.11-1
        a_ids[i] = 0
        theta[i] = 0
        if i==0:
            a_ids[i]=1
        elif i==15:
            a_ids[i]=2
    pass
'''
def init(swarmsize, x, y, theta, a_ids):
    import math
    import random
    import numpy as np
    
    landmarks = np.array([[2., 5.], [2., -5.]])
    pos0 = np.array([[0., 0.3, 0.3, 0.], [0., 0., 0.3, 0.3]])
    theta0 = [0, 0, 0, 0, 0]



    for i in range(swarmsize-2):
        x[i] = pos0[0][i] 
        y[i] = pos0[1][i] 
        a_ids[i] = i
        theta[i] = theta0[i]
        # if i%3==0:
		# 	a_ids[i]=1
		# elif i%3==1:
		# 	a_ids[i]=0
		# else:
		# 	a_ids[i]=2

    for i in range(2):
        x[i+4] = landmarks[0][i]
        y[i+4] = landmarks[1][i]
        a_ids[i] = i
        theta[i] = 0

    return x, y, theta, a_ids