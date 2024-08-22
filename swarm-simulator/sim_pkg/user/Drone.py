'''
Helper swarm class for multiagent estimation discussed in
Rutkowski, Adam J., Jamie E. Barnes, and Andrew T. Smith. "Path planning for optimal cooperative navigation." 2016 IEEE/ION Position, Location and Navigation Symposium (PLANS). IEEE, 2016.

Original MATLAB implementation by Hao Chen

Python implementation by Shahbaz P Qadri Syed, He Bai
'''
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from scipy.optimize import minimize
from itertools import repeat
from user.Vehicle import Vehicle
from user.Neighbor import Neighbor
from user.agent import Agent
# from Vehicle import Vehicle # Delete and move back to above before running sim again
# from Neighbor import Neighbor
# from agent import Agent


# from joblib import Parallel, delayed
# import multiprocessing

# num_cores = multiprocessing.cpu_count()

# Swarm class
class Drone():
    def __init__(self, id, Delta_t, t, v, std_omega, std_v, std_range, f_range, f_odom, S_Q = np.diag([0.1, 0.1, 0.02])):
        self._id = id
        self.neighbors = []
        self.adjacency = []
        self.nb_neighbors = 0
        self.timestamp = []
        self.meas_range = 0
        self.swarm_measRange_history = []
        self.est_X = []
        self.ctrl_intv = 1.0
        self.pred_intv = 1.0
        self.MPC_horizon = 10
        self.M = 5
        
        self.EST_ERR = []   # Had to initialize EST_ERR in __init__ because of informational limits imposed by the nature of this simulation
        self.Delta_t = Delta_t
        self.sim_t = t
        self.omega = 0.
        self.v = v
        self.std_omega = std_omega
        self.std_range = std_range
        self.std_v = std_v
        self.f_range = f_range
        self.f_odom = f_odom
        self.nx = 3
        self.nu = 2
        self.S_Q = S_Q
        
        self.vehicle = Vehicle(Delta_t, t, v, std_omega, std_v, std_range, f_range, f_odom, S_Q)
        self.omega_max = self.vehicle.omega_max
        self.nx = self.vehicle.nx
        self.std_omega = self.vehicle.std_omega # std deviation of ang vel measurement
        self.std_range = self.vehicle.std_range # std deviation of range measurement
        self.std_v = self.vehicle.std_v # std deviation of linear vel measurement

    @property
    def id(self):
        return self._id

    def add_neighbor(self, id):
        self.nb_neighbors += 1
        neighbor = Neighbor(id, self.Delta_t, self.sim_t, self.v, self.std_omega, self.std_v, self.std_range, self.f_range, self.f_odom, self.S_Q)
        neighbor.std_range = self.std_range # Remove Possibly
        self.neighbors.append(neighbor)
         
    
    # TODO: Check if this is decentralized, it should only be called on the information had by a single drone
    # generate a set of angular velocities for one agent
    def gen_w(self):
        # only changing the control in the first 3 steps
        M = self.M
        w_range = np.linspace(-self.omega_max, self.omega_max, M)
        x, y, z = np.meshgrid(w_range, w_range, w_range, indexing = 'xy')
        w_set = np.array([x.reshape((M**3, )), y.reshape((M**3, )), z.reshape((M**3, ))]).T
        w_set = np.hstack((w_set, np.zeros((M**3, self.MPC_horizon - 3)))) #np.kron(np.ones((1,self.MPC_horizon-3)), z.reshape((M**3,1)))))
        #w_set = np.kron(np.ones((1,self.MPC_horizon)), w_range.reshape((M,1)))
        return w_set
    

    # TODO: decentralize - does some stuff to 'states' before calling the function 
    #                    - 'compute_SAM_metric' and returning the result 
    def metric(self, w_set, states, optim_agent):
        agent = Agent()
        U = agent.unicycle
        for j in range(self.MPC_horizon):
            # Set control inputs for the drone
            if n == int(optim_agent):
                inputs = np.array([[self.vehicle.v], [w_set[j]]])
            else:
                inputs = np.array([[self.vehicle.v], [self.vehicle.omega]]) 
            
            states[:, j + 1:j + 2, 0] = U.discrete_step(states[:, j:j + 1, 0], inputs, self.pred_intv)
            
            for n in range(self.nb_neighbors + 1):
                neighbor = self.neighbors[n - 1]
                neighbor_states_est = neighbor.get_est() # TODO: MAKE SURE set_est() is called FIRST in each Loop. 
                states[:,j+1:j+2,n] = neighbor_states_est# TODO: Also, Make sure that the data is in the correct form after transmission and is not changed (if it is, change it back)
        return self.compute_SAM_metric(states)
    

    # TODO: decentralize - does some decisions based on ?? and changes Qh and Jh
    #                    - which are calculated in this and returned. Idk what this does
    #       F = Jh @ Jd
    #       M = F.T @ np.linalg.inv(Qh) @ F        
    #       return 1/np.linalg.det(M)
    def compute_obsv_metric(self, states, w_set, optim_agent):
        agent = Agent()
        U = agent.unicycle
        for i in range(self.MPC_horizon):
            state = states[:,i,:].squeeze()
            if i == 0:
                Jh = self.compute_range_jac(state)
                Qh = self.std_range ** 2 * np.eye(Jh.shape[0])
            else:
                jh = self.compute_range_jac(state)
                Jh = sc.linalg.block_diag(Jh, jh)
                Qh = sc.linalg.block_diag(Qh, self.std_range ** 2 * np.eye(jh.shape[0]))
        Jd = np.eye(self.nx * self.nb_agents)
        jdc = Jd
        for j in range(self.MPC_horizon - 1):
            for n in range(self.nb_agents):
                states_j_n = states[:,j,n].squeeze().reshape((self.nx,1))
                if n == optim_agent:
                    inputs = np.array([[self.vehicles[n].v],[w_set[j]]])
                else:
                    inputs = np.array([[self.vehicles[n].v],[self.vehicles[n].omega]])
                jd = U.dyn_jacobian(states_j_n, inputs, self.pred_intv)
                if n == 0:
                    Jdn = jd
                else:
                    Jdn = sc.linalg.block_diag(Jdn,jd)    
            jdc = Jdn @ jdc
            Jd = np.vstack((Jd, jdc))
        F = Jh @ Jd
        M = F.T @ np.linalg.inv(Qh) @ F        
        return 1/np.linalg.det(M)
    

    # TODO: decentralize - propogates states and computes metrics based on the inputs
    #                    - returns the metrics
    # def try_parallel(self, i,optim_agent,states,METRIC,U):
    #     all_inputs = np.zeros((self.vehicles[0].nu, self.MPC_horizon, self.nb_agents))
    #     for j in range(self.MPC_horizon):
    #         for n in range(self.nb_agents):
    #             if n == int(optim_agent):
    #                 inputs = np.array([[self.vehicles[n].v], [self.w_set[i,j]]])
    #             else:
    #                 inputs = np.array([[self.vehicles[n].v], [self.vehicles[n].omega]])
    #             all_inputs[:,j,n] = inputs.reshape((2,))
    #         #propagate the state
    #             states[:,j+1:j+2,n] = U.discrete_step(states[:,j:j+1,n],inputs,self.pred_intv)
    #     if METRIC == 'SAM':
    #         metrics=self.compute_SAM_metric1(states,all_inputs)
    #     elif METRIC == 'obsv':
    #         metrics=self.compute_obsv_metric(states, self.w_set[i,:], optim_agent)
    #     else:
    #         print('metric not implemented')
    #     return metrics
    

    # TODO: decentralize - depending on use_cov, it updates the controller 
    #                    - for all agents or computes the optimal control 
    #                    - input 
    # cooeprative MPC
    def MPC(self, optim_agent, use_cov = False, METRIC = 'obsv'):
        print("MPC CALLED")
        if use_cov == False:
            if optim_agent == None:
                self.vehicle.update_controller()
            else:
                if self._id != optim_agent:
                    self.vehicle.update_controller()
                self.w_set = self.gen_w()
                w = 0
                agent = Agent()
                U = agent.unicycle
                states = np.zeros((self.vehicles[0].nx, self.MPC_horizon))
                states[0:2,0:1] = self.vehicle.states_est[0:2,:]
                states[2,0:1] = self.vehicle.states_est[2,:]
                N_total = self.w_set.shape[0] #M**3
                metrics = np.zeros((N_total, 1))
                
                m_min = np.argmin(metrics)
                w = self.w_set[m_min,0]

                # set the control
                print(f"END OF MPC FOR ID: {self.id} OMEGA: {w}")
                self.vehicle.omega = w
                print(w)
        else:
            # use covariance information
            active_planning = False
            if self.vehicle.states_cov[0,0] +  self.vehicle.states_cov[1,1] > 5: # needs planning
                active_planning = True
            if active_planning == False:
                self.vehicle.update_controller()
            else:
                if optim_agent == None:
                    optim_agent = np.random.randint(0,self.nb_neighbors + 1)
                    # Set the optim agent to one of the neighbors? 
                    # Would be much easier to predefine.
                    
                # repeat for the case use_cov = False and optim_agent not None
                if self._id != optim_agent:
                    self.vehicle.update_controller()
                self.w_set = self.gen_w()
                w = 0
                agent = Agent()
                U = agent.unicycle
                states = np.zeros((self.vehicles[0].nx, self.MPC_horizon))
                states[0:2,0:1] = self.vehicle.states_est[0:2,:]
                states[2,0:1] = self.vehicle.states_est[2,:]
                N_total = self.w_set.shape[0]
                metrics = np.zeros((N_total, 1))
                for i in range(N_total):
                    for j in range(self.MPC_horizon):
                        if self._id == int(optim_agent):
                            inputs = np.array([[self.vehicle.v], [self.w_set[i,j]]])
                        else:
                            inputs = np.array([[self.vehicle.v], [self.vehicle.omega]])
                    #propagate the state
                        states[:,j+1:j+2] = U.discrete_step(states[:,j:j+1],inputs,self.pred_intv)
                    metrics[i,0]=self.compute_SAM_metric(states)
                m_min = np.argmin(metrics)
                w = self.w_set[m_min,0]
                # set the control
                print(f"END OF MPCOPTIM FOR ID: {self.id} OMEGA: {w}")
                self.vehicle.omega = w
    

    def compute_SAM_metric1(self, states, all_inputs):
        for i in range(self.MPC_horizon):
            state = states[:,i,:].squeeze()
            if i == 0:
                Jh = self.compute_range_jac(state)
                Qh = self.std_range ** 2 * np.eye(Jh.shape[0])
            else:
                jh = self.compute_range_jac(state)
                Jh = sc.linalg.block_diag(Jh, jh)
                Qh = sc.linalg.block_diag(Qh, self.std_range ** 2 * np.eye(jh.shape[0]))
        
        for j in range(self.MPC_horizon - 1):
            state_j = states[:,j,:].squeeze()
            inputs = all_inputs[:,j,:].squeeze()
            #state_j1 = states[:,j+1,:].squeeze()
            if j == 0:
                Jd = self.compute_dynamics_jac1(state_j, inputs, j)
            else:
                jd = self.compute_dynamics_jac1(state_j, inputs, j)
                Jd = np.vstack((Jd, jd))
        Qd = np.kron(np.eye((self.nb_neighbors + 1) * (self.MPC_horizon - 1)), self.vehicles[0].S_Q.T @ self.vehicles[0].S_Q)        
        J = np.vstack((Jh, Jd))
        Q = sc.linalg.block_diag(Qh, Qd)
        M = J.T @ np.linalg.inv(Q) @ J        
        return 1/np.linalg.det(M)
    
    def compute_dynamics_jac1(self, statej, inputs, j):
        n_col = self.nx * self.MPC_horizon * (self.nb_neighbors + 1)
        jac_size = 3
        Jd = np.zeros(((self.nb_neighbors + 1) * jac_size, n_col))
        for n in range(self.nb_neighbors + 1):
            j0 = self.compute_individual_jac1(statej[:,n:n+1], inputs[:,n:n+1], self.pred_intv)
            Jd[jac_size * n: jac_size * n + jac_size, j * self.nx * (self.nb_neighbors + 1) + self.nx * n : j * self.nx * (self.nb_neighbors + 1) + self.nx * n + self.nx] = - j0
            Jd[jac_size * n: jac_size * n + jac_size, (j+1) * self.nx * (self.nb_neighbors + 1) + self.nx * n : (j + 1) * self.nx * (self.nb_neighbors + 1) + self.nx * n + self.nx] = np.eye(self.nx)
        return Jd
            
    
    def compute_individual_jac1(self, statej, inputs, Delta_t):
        agent = Agent()
        unicycle = agent.unicycle
        J0 = unicycle.dyn_jacobian(statej, inputs, Delta_t)
        return J0
    

    # def compute_SAM_metric(self, states):
    #     for i in range(self.MPC_horizon):
    #         state = states[:,i,:].squeeze()
    #         if i == 0:
    #             Jh = self.compute_range_jac(state)
    #             Qh = self.std_range ** 2 * np.eye(Jh.shape[0])
    #         else:
    #             jh = self.compute_range_jac(state)
    #             Jh = sc.linalg.block_diag(Jh, jh)
    #             Qh = sc.linalg.block_diag(Qh, self.std_range ** 2 * np.eye(jh.shape[0]))
        
    #     for j in range(self.MPC_horizon - 1):
    #         state_j = states[:,j,:].squeeze()
    #         state_j1 = states[:,j+1,:].squeeze()
    #         if j == 0:
    #             Jd = self.compute_dynamics_jac(state_j, state_j1, j)
    #         else:
    #             jd = self.compute_dynamics_jac(state_j, state_j1, j)
    #             Jd = np.vstack((Jd, jd))
    #     Qd = np.kron(np.eye((self.nb_neighbors + 1) * (self.MPC_horizon - 1)), np.diag([self.std_v**2, self.std_omega**2]))        
    #     J = np.vstack((Jh, Jd))
    #     Q = sc.linalg.block_diag(Qh, Qd)
    #     M = J.T @ np.linalg.inv(Q) @ J        
    #     return 1/np.linalg.det(M)
    

    # def compute_dynamics_jac(self, statej, statej1, j):
    #     n_col = self.nx * self.MPC_horizon * (self.nb_neighbors + 1)
    #     col_start = j * (self.nb_neighbors + 1) * self.nx
    #     jac_size = 2
    #     Jd = np.zeros(((self.nb_neighbors + 1) * jac_size, n_col))
    #     for n in range(self.nb_neighbors + 1):
    #         j0, j1 = self.compute_individual_jac(statej[:,n:n+1], statej1[:,n:n+1], self.pred_intv)
    #         Jd[jac_size * n: jac_size * n + jac_size, j * self.nx * (self.nb_neighbors + 1) + self.nx * n : j * self.nx * (self.nb_neighbors + 1) + self.nx * n + self.nx] = j0
    #         Jd[jac_size * n: jac_size * n + jac_size, (j+1) * self.nx * (self.nb_neighbors + 1) + self.nx * n : (j + 1) * self.nx * (self.nb_neighbors + 1) + self.nx * n + self.nx] = j1
    #     return Jd
            
        
    def compute_individual_jac(self, statej, statej1, Delta_t):
        agent = Agent()
        unicycle = agent.unicycle
        J0, J1 = unicycle.find_u_jacobian(statej, statej1, Delta_t)
        return J0, J1
    
    def compute_range_jac(self, state):
        adjacency = self.vehicles[0].adjacency
        nx = self.nx
        nb_agents = self.nb_agents
        for j in range(nb_agents):
            ego_idx = j
            vehicle_pos = state[0:2,j:j+1]
            neighbor_idx_set = np.nonzero(adjacency[j, :])[0]
            for n in range(neighbor_idx_set.shape[0]):
                jac = np.zeros((1, nx * nb_agents))
                neighbor_idx = neighbor_idx_set[n]
                neighbor_pos = state[0:2,neighbor_idx].reshape(2, 1)
                range_ = np.linalg.norm(vehicle_pos - neighbor_pos)
                jac[:,ego_idx*nx:((ego_idx+1)*nx)-1] = -(neighbor_pos - vehicle_pos).transpose()
                jac[:,neighbor_idx*nx:((neighbor_idx+1)*nx)-1] = -(-neighbor_pos + vehicle_pos).transpose()

                if j == 0 and n == 0:
                    if range_ != 0:
                        jacobians = (1. / range_) * jac
                    else:
                        jacobians = np.zeros((1, nx * nb_agents))
                else:
                    if range_!= 0:
                        jacobians = np.vstack((jacobians,(1. / range_) * jac))
                    else:
                        jacobians = np.vstack((jacobians, np.zeros((1, nx * nb_agents))))
        return jacobians


    # TODO: decentralize - sets estimation flags and estimates states for 
    #                    - the first two vehicles
    # def addToLoop_estimator(self):
    #     self.vehicles[0].use_estimation = True
    #     self.vehicles[1].use_estimation = True

    #     #GTSAM
    #     self.vehicles[0].states_est = self.est_X[:3,:]
    #     self.vehicles[1].states_est = self.est_X[3:,:]


    # Update drone's adjacency array from adjacency matrix
    def update_adjacency(self, adjacency):
        self.adjacency = adjacency[self._id]
        self.vehicle.adjacency = adjacency[self._id]
        self.vehicle.measRange_history = np.empty((self.nb_neighbors + 1,0))
        self.vehicle.count = 0


    # Generate range measurements for drone and immediate neighbors
    def update_measRange(self):
        #update range sensor measurements
        measRange_list = []
        vehicle_pos = self.vehicle.states[:2,:]
        for neighbor in self.neighbors:
            neighbor_pos = neighbor.get_est()[:2].reshape(2,1)
            meas_range  =  np.linalg.norm(vehicle_pos - neighbor_pos) + self.vehicle.std_range*np.random.randn()
            measRange_list.append(meas_range)

        measRange_array = np.array(measRange_list).reshape(-1,1)
        #         # Debugging prints to understand the shapes
        # print(f"Shape of measRange_array: {measRange_array.shape}")
        # print(f"Shape of self.vehicle.measRange_history: {self.vehicle.measRange_history.shape}")

        if self.vehicle.measRange_history.shape[0] != measRange_array.shape[0]:
            self.vehicle.measRange_history = np.resize(self.vehicle.measRange_history, (measRange_array.shape[0], self.vehicle.measRange_history.shape[1]))

        self.vehicle.measRange_history = np.hstack((self.vehicle.measRange_history, measRange_array))


    # Generate swarm states history
    def  get_drone_states_history_(self):
        self.vehicle.states_history = np.delete(self.vehicle.states_history, 0, 1)
        self.get_drone_states_history = self.vehicle.states_history
    

    def  get_swarm_est_err_(self):
        self.vehicle.est_err = np.delete(self.vehicle.est_err, 0, 1)
        self.EST_ERR = self.vehicle.est_err


    # # FIXME: REMOVE? - This is already done by the simulation
    # # Plot swarm states history
    # def plot_swarm_traj(self):
    #     for i in range(self.nb_agents):
    #         states = self.get_swarm_states_history[i]
    #         time = self.vehicles[i].sim_t
    #         plt.plot(states[0,:], states[1,:], label= 'Vehicle '+str(i))
    #         # plt.quiver(states[0,:], states[1,:], np.cos(states[2,:]), np.sin(states[2,:]), scale= 20)
    #     plt.legend()
    #     plt.title('Vehicle trajectories')
    #     plt.xlabel('x (m)')
    #     plt.ylabel('y (m)')
    #     plt.show()

    # # Plot swarm heading
    # def plot_swarm_heading(self):
    #     for i in range(self.nb_agents):
    #         states = self.get_swarm_states_history[i]
    #         time = self.vehicles[i].sim_t
    #         plt.plot(time, states[2,1:], label= 'Vehicle '+str(i))
    #     plt.legend()
    #     plt.title('Vehicle headings')
    #     plt.xlabel('time (s)')
    #     plt.ylabel('$\\theta$ (rad)')
    #     plt.show()