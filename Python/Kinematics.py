import numpy as np
import matplotlib.pyplot as plt

class Kinematics:
    def __init__(self):
        pass

    def discrete_unicycle(self, states, inputs, Delta_t):
        x = states[0,:]
        y = states[1,:]
        theta = states[2,:]
        v = inputs[0,:]
        omega = inputs[1,:]

        mu = v*Delta_t*np.sinc((omega*Delta_t)/(2*np.pi)) # Division by np.pi to obtain unnormalized sinc i.e., sin(x)/x
        x_k1 = x + mu*np.cos(theta + (omega*Delta_t)/2)
        y_k1 = y + mu*np.sin(theta + (omega*Delta_t)/2)
        theta_k1 = theta + omega*Delta_t

        return np.vstack((x_k1, y_k1, theta_k1)).reshape(3,1)


# Unit Test
# K = Kinematics()
# Delta_t = 0.01
# states = np.array([[1.],[0.],[0.02]])
# input = np.array([[1.],[0.01]])
# x = np.zeros((3,100))
# x[:,0:1] = states
#
# for i in range(1,100):
#     x[:,i:i+1] = K.discrete_unicycle(x[:,i-1], input, Delta_t)
#
# plt.plot(x[0,:], x[1,:])
# plt.show()
# plt.plot(x[2,:])
# plt.show()

