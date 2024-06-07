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
            z = states[2, :]
            theta = states[3,:]
            v = inputs[0,:]
            omega = inputs[1,:]
            z_vel = inputs[2, :]

            mu = v*Delta_t*np.sinc((omega*Delta_t)/(2*np.pi)) # Division by np.pi to obtain unnormalized sinc i.e., sin(x)/x
            x_k1 = x + mu*np.cos(theta + (omega*Delta_t)/2)
            y_k1 = y + mu*np.sin(theta + (omega*Delta_t)/2)
            z_k1 = z + z_vel*Delta_t
            theta_k1 = theta + omega*Delta_t

            return np.vstack((x_k1, y_k1, z_k1, theta_k1)).reshape(4,1)
        
        # given xk, xk1, find corresponding uk
        def find_u(self, x, xp1, Delta_t): 
            e = xp1 - x
            th = x[3,:]
            omega = e[3,:]/Delta_t
            mu = e[0,:] * np.cos(th + omega * Delta_t/2.0) + e[1,:] * np.sin(th + omega * Delta_t/2.0)
            v = mu / Delta_t /np.sinc(1/np.pi*(omega * Delta_t/2.0))
            z_vel = e[2,:]/Delta_t
            inputs = np.array([v, omega, z_vel]).reshape((2,1))
            return inputs
        
        
        def find_u_jacobian(self, x, xp1, Delta_t):
            omega = (xp1[3,:] - x[3,:])/Delta_t
            theta = x[3,:]
            mu = (xp1[0,:] - x[0,:]) * np.cos(theta + (omega*Delta_t)/2) + (xp1[1,:] - x[1,:]) * np.sin(theta + (omega*Delta_t)/2)
            theta1 = xp1[3,:]
            jac1 = np.zeros((3,4))
            jac0 = np.zeros((3,4))
            # partial omega/partial xp1, and partial omega/partial x 
            jac1[1,3] = 1/Delta_t
            jac0[1,3] = -1/Delta_t
            # partial v/ partial xp1
            jac1[0,0] = 1/Delta_t/np.sinc((omega*Delta_t)/(2*np.pi)) * np.cos(theta + (omega*Delta_t)/2)
            jac1[0,1] = 1/Delta_t/np.sinc((omega*Delta_t)/(2*np.pi)) * np.sin(theta + (omega*Delta_t)/2)
            jac1[0,3] = 0.5 * 1/Delta_t/np.sinc((omega*Delta_t)/(2*np.pi)) * ( - (xp1[0,:] - x[0,:]) * np.sin(theta + (omega*Delta_t)/2) + (xp1[1,:] - x[1,:]) * np.cos(theta + (omega*Delta_t)/2)) + mu / Delta_t * (- 0.5 / np.sinc((omega*Delta_t)/(2*np.pi)) ** 2 * (np.cos((omega*Delta_t)/2) - np.sinc((omega*Delta_t)/(2*np.pi)))/(omega * Delta_t/2))
            jac0[0,0] = - 1/Delta_t/np.sinc((omega*Delta_t)/(2*np.pi)) * np.cos(theta + (omega*Delta_t)/2)
            jac0[0,1] = - jac1[0,1]
            jac0[0,3] = 0.5 * 1/Delta_t/np.sinc((omega*Delta_t)/(2*np.pi)) * ( - (xp1[0,:] - x[0,:]) * np.sin(theta + (omega*Delta_t)/2) + (xp1[1,:] - x[1,:]) * np.cos(theta + (omega*Delta_t)/2)) - mu / Delta_t * (- 0.5 / np.sinc((omega*Delta_t)/(2*np.pi)) ** 2 * (np.cos((omega*Delta_t)/2) - np.sinc((omega*Delta_t)/(2*np.pi)))/(omega * Delta_t/2))

            jac1[2, 2] = 1 / Delta_t
            jac0[2, 2] = -1 / Delta_t
            return jac0, jac1

        # Controller update
        def update_controller(self, states, target, v, omega_max, Delta_t):
            ''' Part of the following code was originally implemented in MATLAB by Hao Chen'''

            self.vehicle_pos = states[:2, :]
            self.vehicle_theta = states[3:, :]
            self.z_pos = states[2:3, :]

            # if D(str(time)) % D(str(self.ctrl_intv)) == 0.:
            v = self.v
            rel_pos = self.target_point[:2, :] - self.vehicle_pos
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

            pert = 0  # 0.01 * np.random.normal()

            K = 1 / self.Delta_t
            omega_max = self.omega_max
            if np.abs(K * beta + pert) <= omega_max:
                self.omega = K * beta + pert
                self.omega_inlimit = True
            elif K * beta + pert < -omega_max:
                self.omega = -omega_max
                self.omega_inlimit = False
            elif K * beta + pert > omega_max:
                self.omega = omega_max
                self.omega_inlimit = False

                # w = self.MPC()
                # self.omega = w
            self.z_vel = (self.target_point[2, 0] - self.z_pos[0,0]) / self.Delta_t

            if self.z_vel <= -self.z_vel_max:
                self.z_vel = -self.z_vel_max
                self.z_vel_inlimit = False
            elif self.z_vel >= self.z_vel_max:
                self.z_vel = self.z_vel_max
                self.z_vel_inlimit = False
            else:
                self.z_vel_inlimit = True

            x = states[0, 0]
            y = states[1, 0]
            z = states[2,0]
            theta = states[3, 0]
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
            
            jac = np.zeros((3,4))
            if self.omega_inlimit is True:
                jac[1,:] =  (1 / Delta_t) * jac_omega
            
            if self.z_vel_inlimit is True:
                jac[2, :] = (1 / Delta_t) * np.array([[0,0,-1,0]])

            return np.array([[v], [self.omega],[self.z_vel]]), jac

        # Jacobian of X_{k+1} w.r.t X_k
        def dyn_jacobian(self, states, inputs, Delta_t):
            v = inputs[0, 0]
            omega = inputs[1, 0]
            return np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[-Delta_t*v*np.sin(Delta_t*omega/2 + states[2,0])*np.sinc(0.159154943091895*Delta_t*omega),Delta_t*v*np.cos(Delta_t*omega/2 + states[2,0])*np.sinc(0.159154943091895*Delta_t*omega), 0, 1]]).transpose()

        # Jacobian of X_{k+1} w.r.t U_k
        def u_jacobian(self, states, inputs, Delta_t):
            v = inputs[0, 0]
            omega = inputs[1, 0]
            c = Delta_t/2
            term1 = Delta_t*v*(c*(np.cos(c*omega)/(c*omega) - np.sin(c*omega)/((c*omega)**2))*np.cos(c*omega + states[2,0]) - c*np.sinc(c*omega/np.pi)*np.sin(c*omega + states[2,0]))
            term2 = Delta_t*v*(c*(np.cos(c*omega)/(c*omega) - np.sin(c*omega)/((c*omega)**2))*np.sin(c*omega + states[2,0]) + c*np.sinc(c*omega/np.pi)*np.cos(c*omega + states[2,0]))
            return np.array([[Delta_t*np.cos(c*omega + states[2,0])*np.sinc(c*omega/np.pi), term1,0],
                                         [Delta_t*np.sin(c*omega + states[2,0])*np.sinc(c*omega/np.pi), term2,0],
                                         [0,0,Delta_t],
                                         [0, Delta_t,0]])


    