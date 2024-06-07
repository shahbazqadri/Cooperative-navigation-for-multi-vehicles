'''
Helper vehicle class for multiagent estimation discussed in
Rutkowski, Adam J., Jamie E. Barnes, and Andrew T. Smith. "Path planning for optimal cooperative navigation." 2016 IEEE/ION Position, Location and Navigation Symposium (PLANS). IEEE, 2016.

Original MATLAB implementation by Hao Chen

Python implementation by Shahbaz P Qadri Syed, He Bai
'''

import numpy as np
from agent import Agent
from decimal import Decimal as D

def angle_bound_rad(in_angle : float) -> float:
    # Simple check to put the value between -pi and pi
    going_out=in_angle
    if in_angle < -np.pi:
        going_out += 2*np.pi
    if in_angle > np.pi:
        going_out -= 2*np.pi
    return going_out

# Vehicle class
class Vehicle(object):
    wp_idx = 0
    adjacency = None # adjacency matrix: initialized during runtime by swarm class

    def __init__(self, Delta_t, t, v, std_omega, std_v, std_range, std_z_vel, f_range, f_odom, S_Q):
        # system param
        self.Delta_t = Delta_t #step size of discretization
        self.sim_t   = t # time intervals
        self.omega   = 0. # angular velocity
        self.v       = v # linear velocity
        self.z_vel      = 0 #rate of ascent/descent
        self.std_omega = std_omega # std deviation of ang vel measurement
        self.std_range = std_range # std deviation of range measurement
        self.std_v     = std_v # std deviation of linear vel measurement
        self.std_z_vel = std_z_vel # std deviation of rate ascent/descent measurement
        self.f_range   = f_range # frequency of range measurement
        self.f_odom    = f_odom # frequency of odometry measurement
        self.omega_max = np.deg2rad(10);
        self.z_vel_max = 8
        self.nx = 4
        self.nu = 3
        self.S_Q = S_Q

        # controller parameters
        self.ctrl_cmd = np.empty((3,1))
        self.ctrl_cmd_history = np.empty((3,1))
        self.ctrl_intv = 1.0
        self.pred_intv = 1.0
        self.MPC_horizon = 10
        self.M = 5
        self.w_set = self.gen_w()

        # states parameters
        self.states  = np.zeros((4,1))
        self.states_est = np.zeros((4,1))
        self.last_states = np.zeros((4,1))
        self.next_states = np.zeros((4,1))
        self.states_history = np.empty((4,1))
        self.est_err = np.empty((4,1))
        

        # odom measurements parameters
        self.meas = np.empty((3,1))
        self.meas_history = np.empty((3,1))

        # navigation parameters
        self.target_point = np.array([[3000],[3000],[500]])
        self.waypoints    = []

        # estimator param
        self.use_estimation = True
        
    def gen_w(self):
        # only changing the control in the first 3 steps
        M = self.M
        w_range = np.linspace(-self.omega_max, self.omega_max, M)
        x, y, z = np.meshgrid(w_range, w_range, w_range, indexing = 'xy')
        w_set = np.array([x.reshape((M**3, )), y.reshape((M**3, )), z.reshape((M**3, ))]).T
        w_set = np.hstack((w_set, np.zeros((M**3, self.MPC_horizon - 3))))
        return w_set

    # Set initial vehicle position
    def set_initPos(self, initPos):
        # set initial position
        self.states[0:3,:] = initPos

    # Set final vehicle position
    def set_endPos(self, endPos):
        # set end position
        self.target_point = endPos

    # Set initial vehicle pose
    def set_initPose(self, initPose):
        # set initial position
        self.states = initPose

    # Set final vehicle pose
    def set_endPose(self, endPose):
        # set end position
        self.target_point = endPose[:3,:]

    # Update vehicle control, state, and measurement
    def update_state(self, time):
        # update_state: compute the new vehicle state
        self.update_measurements(time)
        #self.update_controller(time)
        self.update_kinematics()
        #self.add_process_noise()

        self.last_states = self.states
        self.states = self.next_states
        self.states_history = np.hstack((self.states_history, self.states))

    # Update kinematics
    def update_kinematics(self):
        # compute kinematics
        agent = Agent()
        U = agent.unicycle
        states = self.states
        inputs = np.array([[self.v],[self.omega],[self.z_vel]])

        self.next_states = U.discrete_step(states,inputs,self.Delta_t)
        out_state = self.next_states + self.S_Q@np.random.randn(4,1)
        out_state[3,0]= angle_bound_rad(out_state[3])
        self.next_states = out_state
        
    def add_process_noise(self, S_Q):
        out_state = self.states + S_Q@np.random.randn(4)
        out_state[3,0]= angle_bound_rad(out_state[3])
        self.states = out_state 

    # Update vehicle measurements
    def update_measurements(self, time):
        # updata sensor measurements: update the odom measurements
        odom_period = 1./self.f_odom
        if D(str(time) )% D(str(odom_period ))== 0.:
            meas_encoder = np.array([[self.v + self.std_v*np.random.randn()],[self.omega + self.std_omega*np.random.randn()],[self.z_vel + self.std_z_vel*np.random.randn()]])
            self.meas    = meas_encoder
            self.meas_history= np.hstack((self.meas_history,meas_encoder))

    # Update vehicle controller
    def update_controller(self):
        if self.use_estimation == True:
            self.vehicle_pos = self.states_est[:2,:]
            self.vehicle_theta = self.states_est[3:,:]
            self.z_pos = self.states_est[2:3,:]
        else:
            self.vehicle_pos = self.states[:2,:]
            self.vehicle_theta = self.states[3:,:]
            self.z_pos = self.states[2:3, :]
        
        #if D(str(time)) % D(str(self.ctrl_intv)) == 0.:
        v = self.v
        rel_pos = self.target_point[:2,:] - self.vehicle_pos
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
        
        pert =  0#0.01 * np.random.normal()
        
        K = 1/self.Delta_t
        omega_max = self.omega_max 
        if np.abs(K*beta + pert) <= omega_max:
            self.omega = K*beta + pert
            self.omega_inlimit = True
        elif K*beta + pert < -omega_max:
            self.omega = -omega_max
            self.omega_inlimit = False
        elif K*beta + pert > omega_max:
            self.omega = omega_max
            self.omega_inlimit = False
                    
            # w = self.MPC()
            # self.omega = w
        self.z_vel = 0.5*(self.target_point[2,0] - self.z_pos[0,0]) / self.Delta_t
        
        if self.z_vel <= -self.z_vel_max:
            self.z_vel = -self.z_vel_max
            self.z_vel_inlimit = False
        elif self.z_vel >= self.z_vel_max:
            self.z_vel = self.z_vel_max
            self.z_vel_inlimit = False
        else:
            self.z_vel_inlimit = True

        self.ctrl_cmd = np.array([[self.omega],[self.v], [self.z_vel]])
        self.ctrl_cmd_history = np.hstack((self.ctrl_cmd_history, self.ctrl_cmd))


    def MPC(self):
        w = 0
        agent = Agent()
        U = agent.unicycle
        states = np.zeros((self.nx, self.MPC_horizon))
        states[0:3,0:1] = self.vehicle_pos
        states[3,0:1] = self.vehicle_theta
        metrics = np.zeros((self.M**3, 1))
        for i in range(self.M**3):
            for j in range(self.MPC_horizon):
                inputs = np.array([[self.w_set[i,j]],[self.v]])
                #propagate the state
                states[:,j+1:j+2] = U.discrete_step(states[:,j:j+1],inputs,self.pred_intv)
            metrics[i,0]=self.compute_metric(states)
        m_max = np.argmin(metrics)
        w = self.w_set[m_max,0]
        return w
    
    def compute_metric(self, states):
        dis_2_target = np.linalg.norm(states[0:3,-1:] - self.target_point)
        return dis_2_target


    def update_waypoints_ctrl(self, time):
        if self.use_estimation is True:
            vehicle_pos = self.states_est[:3,:]
            vehicle_theta = self.states_est[3:,:]
        else:
            vehicle_pos = self.states[:3,:]
            vehicle_theta = self.states[3:,:]

        if np.linalg.norm(self.waypoints[:,Vehicle.wp_idx]- vehicle_pos) < 100:
            Vehicle.wp_idx = Vehicle.wp_idx+1

        self.target_point = self.waypoints[:,Vehicle.wp_idx]
        self.update_controller(time)


