import numpy as np
from Kinematics import Kinematics
class Vehicle(object):
    wp_idx = 0

    #graph param
    adjacency = None

    def __init__(self, Delta_t, t, v, std_omega, std_v, std_range, f_range, f_odom):
        # system param
        self.Delta_t = Delta_t
        self.sim_t   = t
        self.omega   = 0.
        self.v       = v
        self.std_omega = std_omega
        self.std_range = std_range
        self.std_v     = std_v
        self.f_range   = f_range
        self.f_odom    = f_odom

        # controller param
        self.ctrl_cmd = np.empty((2,1))
        self.ctrl_cmd_history = np.empty((2,1))

        # states param
        self.states  = np.zeros((3,1))
        self.states_est = np.zeros((3,1))
        self.last_states = np.zeros((3,1))
        self.next_states = np.zeros((3,1))
        self.states_history = np.empty((3,1))

        # odom measurements param
        self.meas = np.empty((2,1))
        self.meas_history = np.empty((2,1))

        # range measurements param
        self.measRange_history = []

        # navigation param
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

    def update_state(self, time):
        # update_state: compute the new vehicle state
        self.update_measurements(time)
        self.update_controller(time)
        self.update_kinematics(time)

        self.last_states = self.states
        self.states = self.next_states
        self.states_history = np.hstack((self.states_history, self.states))


    def update_kinematics(self, time):
        # compute kinematics
        K = Kinematics()
        states = self.states
        inputs = np.array([[self.v],[self.omega]])

        self.next_states = K.discrete_unicycle(states,inputs,self.Delta_t)

    def update_measurements(self, time):
        # updata sensor measurements: update the odom measurements
        odom_period = 1/self.f_odom
        if time%odom_period == 0:
            meas_encoder = np.array([[self.v + self.std_v*np.random.randn()],[self.omega + self.std_omega*np.random.randn()]])
            self.meas    = meas_encoder
            self.meas_history= np.hstack((self.meas_history,meas_encoder))


    def update_controller(self, time):
        if self.use_estimation == True:
            vehicle_pos = self.states_est[:2,:]
            vehicle_theta = self.states_est[2:,:]
        else:
            vehicle_pos = self.states[:2,:]
            vehicle_theta = self.states[2:,:]

        v = self.v
        rel_pos = self.target_point - vehicle_pos
        vehicle_vel = np.vstack((v*np.cos(vehicle_theta),v*np.sin(vehicle_theta)))
        vec_cos = np.dot(rel_pos[:,0], vehicle_vel[:,0])/(np.linalg.norm(rel_pos)*np.linalg.norm(vehicle_vel))
        if vec_cos < -1:
            vec_cos = -1
        elif vec_cos > 1:
            vec_cos = 1

        beta = np.arccos(vec_cos)
        cross_product = np.cross(np.vstack((vehicle_vel, 0.))[:,0], np.vstack((rel_pos, 0.))[:,0])
        if cross_product[2] < 0:
            beta = -beta

        K = 1/self.Delta_t
        omega_max = np.deg2rad(5);
        if np.abs(K*beta) <= omega_max:
            self.omega = K*beta
        elif K*beta < -omega_max:
            self.omega = -omega_max
        elif K*beta > omega_max:
            self.omega = omega_max

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


