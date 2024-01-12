'''
Helper vehicle class for multiagent estimation discussed in
Rutkowski, Adam J., Jamie E. Barnes, and Andrew T. Smith. "Path planning for optimal cooperative navigation." 2016 IEEE/ION Position, Location and Navigation Symposium (PLANS). IEEE, 2016.

Original MATLAB implementation by Hao Chen

Python implementation by Shahbaz P Qadri Syed, He Bai
'''

import numpy as np
from agent import Agent
from decimal import Decimal as D

class Vehicle(object):
    wp_idx = 0
    adjacency = None # adjacency matrix: initialized during runtime by swarm class

    def __init__(self, Delta_t, t, v, std_omega, std_v, std_range, f_range, f_odom):
        # system param
        self.Delta_t = Delta_t #step size of discretization
        self.sim_t   = t # time intervals
        self.omega   = 0. # angular velocity
        self.v       = v # linear velocity
        self.std_omega = std_omega # std deviation of ang vel measurement
        self.std_range = std_range # std deviation of range measurement
        self.std_v     = std_v # std deviation of linear vel measurement
        self.f_range   = f_range # frequency of range measurement
        self.f_odom    = f_odom # frequency of odometry measurement

        # controller parameters
        self.ctrl_cmd = np.empty((2,1))
        self.ctrl_cmd_history = np.empty((2,1))

        # states parameters
        self.states  = np.zeros((3,1))
        self.states_est = np.zeros((3,1))
        self.last_states = np.zeros((3,1))
        self.next_states = np.zeros((3,1))
        self.states_history = np.empty((3,1))

        # odom measurements parameters
        self.meas = np.empty((2,1))
        self.meas_history = np.empty((2,1))

        # navigation parameters
        self.target_point = np.array([[3000],[3000]])
        self.waypoints    = []

        # estimator param
        self.use_estimation = False

    def set_initPos(self, initPos):
        # set initial position
        self.states[0:2,:] = initPos

    def set_endPos(self, endPos):
        # set end position
        self.target_point = endPos

    def set_initPose(self, initPose):
        # set initial position
        self.states = initPose

    def set_endPose(self, endPose):
        # set end position
        self.target_point = endPose[:2,:]

    def update_state(self, time):
        # update_state: compute the new vehicle state
        self.update_measurements(time)
        self.update_controller()
        self.update_kinematics()

        self.last_states = self.states
        self.states = self.next_states
        self.states_history = np.hstack((self.states_history, self.states))


    def update_kinematics(self):
        # compute kinematics
        agent = Agent()
        U = agent.unicycle
        states = self.states
        inputs = np.array([[self.v],[self.omega]])

        self.next_states = U.discrete_step(states,inputs,self.Delta_t)
    def update_measurements(self, time):
        # updata sensor measurements: update the odom measurements
        odom_period = 1./self.f_odom
        if D(str(time) )% D(str(odom_period ))== 0.:
            meas_encoder = np.array([[self.v + self.std_v*np.random.randn()],[self.omega + self.std_omega*np.random.randn()]])
            self.meas    = meas_encoder
            self.meas_history= np.hstack((self.meas_history,meas_encoder))


    def update_controller(self):
        if self.use_estimation == True:
            self.vehicle_pos = self.states_est[:2,:]
            self.vehicle_theta = self.states_est[2:,:]
        else:
            self.vehicle_pos = self.states[:2,:]
            self.vehicle_theta = self.states[2:,:]

        v = self.v
        rel_pos = self.target_point - self.vehicle_pos
        vehicle_vel = np.vstack((v*np.cos(self.vehicle_theta),v*np.sin(self.vehicle_theta)))
        self.vec_cos = np.dot(rel_pos[:,0], vehicle_vel[:,0])/(np.linalg.norm(rel_pos)*np.linalg.norm(vehicle_vel))
        if self.vec_cos < -1:
            self.vec_cos = -1
        elif self.vec_cos > 1:
            self.vec_cos = 1

        beta = np.arccos(self.vec_cos)
        self.cross_product = np.cross(np.vstack((vehicle_vel, 0.))[:,0], np.vstack((rel_pos, 0.))[:,0])
        if self.cross_product[2] < 0:
            beta = -beta

        K = 1/self.Delta_t
        omega_max = np.deg2rad(5);
        if np.abs(K*beta) <= omega_max:
            self.omega = K*beta
            self.omega_inlimit = True
        elif K*beta < -omega_max:
            self.omega = -omega_max
            self.omega_inlimit = False
        elif K*beta > omega_max:
            self.omega = omega_max
            self.omega_inlimit = False

        self.ctrl_cmd = np.array([[self.omega],[self.v]])
        self.ctrl_cmd_history = np.hstack((self.ctrl_cmd_history, self.ctrl_cmd))

    def update_waypoints_ctrl(self, time):
        if self.use_estimation is True:
            vehicle_pos = self.states_est[:2,:]
            vehicle_theta = self.states_est[2:,:]
        else:
            vehicle_pos = self.states[:2,:]
            vehicle_theta = self.states[2:,:]

        if np.linalg.norm(self.waypoints[:,Vehicle.wp_idx]- vehicle_pos) < 100:
            Vehicle.wp_idx = Vehicle.wp_idx+1

        self.target_point = self.waypoints[:,Vehicle.wp_idx]
        self.update_controller(time)


