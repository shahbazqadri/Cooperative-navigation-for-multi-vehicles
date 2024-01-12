'''
Helper vehicle class for multiagent estimation discussed in
Rutkowski, Adam J., Jamie E. Barnes, and Andrew T. Smith. "Path planning for optimal cooperative navigation." 2016 IEEE/ION Position, Location and Navigation Symposium (PLANS). IEEE, 2016.

Authors: Shahbaz P Qadri Syed, He Bai
'''

import numpy as np

# Agent class
class Agent:
    def __init__(self):
        self.unicycle = self.Unicycle()

    class Unicycle:
        def __init__(self):
            pass
        # Discrete unicycle dynamics
        def discrete_step(self, states, inputs, Delta_t):
            '''Originally implemented in MATLAB by Hao Chen'''

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

        # Controller update
        def update_controller(self, states, target, v, omega_max, Delta_t):
            ''' Part of the following code was originally implemented in MATLAB by Hao Chen'''

            self.vehicle_pos = states[:2, :]
            self.vehicle_theta = states[2:, :]

            rel_pos = target - self.vehicle_pos
            vehicle_vel = np.vstack((v * np.cos(self.vehicle_theta), v * np.sin(self.vehicle_theta)))
            self.vec_cos = np.dot(rel_pos[:, 0], vehicle_vel[:, 0]) / (
                        np.linalg.norm(rel_pos) * np.linalg.norm(vehicle_vel))
            if self.vec_cos < -1:
                self.vec_cos = -1
            elif self.vec_cos > 1:
                self.vec_cos = 1

            beta = np.arccos(self.vec_cos)
            self.cross_product = np.cross(np.vstack((vehicle_vel, 0.))[:, 0], np.vstack((rel_pos, 0.))[:, 0])
            if self.cross_product[2] < 0:
                beta = -beta

            K = 1 / Delta_t
            omega_max = np.deg2rad(omega_max);
            if np.abs(K * beta) <= omega_max:
                self.omega = K * beta
                self.omega_inlimit = True
            elif K * beta < -omega_max:
                self.omega = -omega_max
                self.omega_inlimit = False
            elif K * beta > omega_max:
                self.omega = omega_max
                self.omega_inlimit = False

            x = states[0, 0]
            y = states[1, 0]
            theta = states[2, 0]
            if self.vec_cos < -1 or self.vec_cos > 1:
                jac_omega = 0
            else:
                term1 = (1/np.sqrt(1-self.vec_cos**2))
                term2 = -v * np.cos(self.vehicle_theta)/(np.linalg.norm(rel_pos) * np.linalg.norm(vehicle_vel))
                term3 = np.dot(rel_pos[:, 0], vehicle_vel[:, 0])*rel_pos[0,0]/((np.linalg.norm(rel_pos)**1.5) * np.linalg.norm(vehicle_vel))
                term4 = -v * np.sin(self.vehicle_theta) / (np.linalg.norm(rel_pos) * np.linalg.norm(vehicle_vel))
                term5 = np.dot(rel_pos[:, 0], vehicle_vel[:, 0])*rel_pos[1,0]/((np.linalg.norm(rel_pos)**1.5) * np.linalg.norm(vehicle_vel))
                term6 = v*(-rel_pos[0,0]*np.sin(self.vehicle_theta) + rel_pos[1,0]*np.cos(self.vehicle_theta))/(np.linalg.norm(rel_pos) * np.linalg.norm(vehicle_vel))
                jac_omega = np.hstack((term1*(term2+term3), term1*(term4+term5), term1*term6))

            if self.cross_product[2] < 0:
                jac_omega = -jac_omega

            if self.omega_inlimit is True:
                jac = np.vstack(((np.zeros((1, 3))), (1 / Delta_t) * jac_omega))

            else:
                jac= np.zeros((2, 3))

            return np.array([[v], [self.omega]]), jac

        # Jacobian of X_{k+1} w.r.t X_k
        def dyn_jacobian(self, states, inputs, Delta_t):
            v = inputs[0, 0]
            omega = inputs[1, 0]
            return np.array([[1,0,0],[0,1,0],[-Delta_t*v*np.sin(Delta_t*omega/2 + states[2,0])*np.sinc(0.159154943091895*Delta_t*omega),Delta_t*v*np.cos(Delta_t*omega/2 + states[2,0])*np.sinc(0.159154943091895*Delta_t*omega), 1]]).transpose()

        # Jacobian of X_{k+1} w.r.t U_k
        def u_jacobian(self, states, inputs, Delta_t):
            v = inputs[0, 0]
            omega = inputs[1, 0]
            c = Delta_t/2
            term1 = Delta_t*v*(c*(np.cos(c*omega)/(c*omega) - np.sin(c*omega)/((c*omega)**2))*np.cos(c*omega + states[2,0]) - c*np.sinc(c*omega/np.pi)*np.sin(c*omega + states[2,0]))
            term2 = Delta_t*v*(c*(np.cos(c*omega)/(c*omega) - np.sin(c*omega)/((c*omega)**2))*np.sin(c*omega + states[2,0]) + c*np.sinc(c*omega/np.pi)*np.cos(c*omega + states[2,0]))
            return np.array([[Delta_t*np.cos(c*omega + states[2,0])*np.sinc(c*omega/np.pi), term1],
                                         [Delta_t*np.sin(c*omega + states[2,0])*np.sinc(c*omega/np.pi), term2],
                                         [0, Delta_t]])
