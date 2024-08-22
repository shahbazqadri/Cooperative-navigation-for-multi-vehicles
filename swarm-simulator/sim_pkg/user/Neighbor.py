import numpy as np
from decimal import Decimal as D
from user.agent import Agent

def angle_bound_rad(in_angle : float) -> float:
    # Simple check to put the value between -pi and pi
    going_out=in_angle
    if in_angle < -np.pi:
        going_out += 2*np.pi
    if in_angle > np.pi:
        going_out -= 2*np.pi
    return going_out

class Neighbor():
    def __init__(self, id, Delta_t, t, v, std_omega, std_v, std_range, f_range, f_odom, S_Qom):
        self.id = id
        self.Delta_t = Delta_t #step size of discretization
        self.sim_t   = t # time intervals
        self.omega   = 0. # angular velocity
        self.v       = v # linear velocity
        self.std_omega = std_omega # std deviation of ang vel measurement
        self.std_range = std_range # std deviation of range measurement
        self.std_v     = std_v # std deviation of linear vel measurement
        self.f_range   = f_range # frequency of range measurement
        self.f_odom    = f_odom # frequency of odometry measurement
        self.omega_max = np.deg2rad(10);
        self.nx = 3
        self.nu = 2
        self.S_Q = S_Qom

        self.states_est = np.array([-999, -999, -999])
        self.states_cov = np.zeros((3,3))
        self.est_err = np.empty((3,1))
        # self.data_shared = False
       
        self.meas_history = np.empty((2,1))
        self.measRange_history = np.empty((2,1))

        self.f_odom = f_odom # frequency of odometry measurement



    def get_est(self):
        return self.states_est
    
    def set_est(self, states_est):
        self.states_est = states_est

    def set_measurement(self, meas):
        self.meas = meas

    # Update vehicle measurements
    def update_measurements(self, time):
        # updata sensor measurements: update the odom measurements
        odom_period = 1./self.f_odom
        if D(str(time) )% D(str(odom_period )) == 0.:

            meas_encoder = self.meas
            self.meas_history= np.hstack((self.meas_history,meas_encoder))

    def update_kinematics(self):
        # compute kinematics
        agent = Agent()
        U = agent.unicycle
        states = self.states
        inputs = np.array([[self.v],[self.omega]])

        self.next_states = U.discrete_step(states,inputs,self.Delta_t)
        out_state = self.next_states + self.S_Q@np.random.randn(3,1)
        out_state[2,0]= angle_bound_rad(out_state[2])
        self.next_states = out_state

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