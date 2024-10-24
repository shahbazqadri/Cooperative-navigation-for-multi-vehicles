'''
Helper swarm class for multiagent estimation discussed in
Rutkowski, Adam J., Jamie E. Barnes, and Andrew T. Smith. "Path planning for optimal cooperative navigation." 2016 IEEE/ION Position, Location and Navigation Symposium (PLANS). IEEE, 2016.

Original MATLAB implementation by Hao Chen

Python implementation by Shahbaz P Qadri Syed, He Bai
'''
import matplotlib.pyplot as plt
import numpy as np
from user.Vehicle import Vehicle
from user.agent import Agent
import scipy as sc
from scipy.optimize import minimize
from scipy.optimize import dual_annealing as anneal
from itertools import repeat
import time

# from joblib import Parallel, delayed
# import multiprocessing

# num_cores = multiprocessing.cpu_count()

# Swarm class
class Swarm():
    def __init__(self):
        self.vehicles = []
        self.nb_agents = 0
        self.timestamp = []
        self.meas_range = 0
        self.swarm_measRange_history = []
        self.est_X = []
        self.ctrl_intv = 1.0
        self.pred_intv = 1.0
        self.MPC_horizon = 10
        self.M = 5
        self.mu = -10 # smoothing parameter
        self.finalP = []
        self.landmarks = []
        self.nb_landmarks = 0
        self.neighbor_ids = []
    # add landmarks
    def add_landmarks(self, lm_locs):
        self.landmarks = lm_locs
        self.nb_landmarks = lm_locs.shape[1]
        # for j in range(self.nb_agents):
        #     self.vehicles[j].lm_meas_history = []
    def agent_to_landmarks(self, agent_idx = None):
        if agent_idx == None:
            self.landmark_agents = [i for i in range(self.nb_agents)]
        else:
            self.landmark_agents = agent_idx

    # Create vehicle object
    def add_vehicle(self, id, Delta_t, t, v, std_omega, std_v, std_range, f_range, f_odom, S_Q = np.diag([0.1, 0.1, 0.02])):
        self.nb_agents += 1
        vehicle = Vehicle(id, Delta_t, t, v, std_omega, std_v, std_range, f_range, f_odom, S_Q)
        self.omega_max = vehicle.omega_max
        self.Delta_t = Delta_t
        self.S_Q = S_Q
        self.nx = vehicle.nx
        self.nu = vehicle.nu
        self.std_omega = vehicle.std_omega # std deviation of ang vel measurement
        self.std_range = vehicle.std_range # std deviation of range measurement
        self.std_v     = vehicle.std_v # std deviation of linear vel measurement
        self.vehicles.append(vehicle)

    # Add multiple vehicles to swarm object
    def add_n_vehicles(self, vehicle, n):
        for i in range(n):
            self.add_vehicle(vehicle)
        
    
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
    
    def metric(self, w_set, states, optim_agent, METRIC):
        all_inputs = np.zeros((self.vehicles[0].nu, self.MPC_horizon, self.nb_agents))
        agent = Agent()
        U = agent.unicycle
        for j in range(self.MPC_horizon-1):
            for n in range(self.nb_agents):
                if n == int(optim_agent):
                    inputs = np.array([[self.vehicles[n].v], [w_set[j]]])
                else:
                    if j == 0:
                        inputs = np.array([[self.vehicles[n].v], [self.vehicles[n].omega]])
                    else:
                        inputs = np.array([[self.vehicles[n].v], [0.]])
                all_inputs[:, j, n] = inputs.reshape((2,))
            #propagate the state
                states[:,j+1:j+2,n] = U.discrete_step(states[:,j:j+1,n],inputs,self.pred_intv)
        if METRIC == 'SAM':
            # print("Using SAM")
            metrics = self.compute_SAM_metric2(states, all_inputs, METRIC)
        elif METRIC == 'min_eig_SAM':
            # print("Using SAM")
            metrics = -self.compute_SAM_metric2(states, all_inputs, METRIC)
        elif METRIC == 'last_SAM_cov':
            # print("Using SAM")
            metrics = self.compute_SAM_metric2(states, all_inputs, METRIC)
        elif METRIC == 'cond_SAM':
            metrics = self.compute_SAM_metric2(states, all_inputs, METRIC)
        elif METRIC == 'min_eig_SAM_appr':
            metrics = -self.compute_SAM_metric2(states, all_inputs, METRIC)
        elif METRIC == 'Trace_inv_SAM':
            # print("Using SAM")
            metrics = self.compute_SAM_metric2(states, all_inputs, METRIC)
        elif METRIC == 'obsv':
            # print("Using obsv")
            metrics = self.compute_obsv_metric(states, np.expand_dims(w_set,axis=1), optim_agent)
        elif METRIC == 'min_eig_inv_cov':
            # print("Using inv_cov")
            metrics = -self.compute_inv_cov_metric(states, all_inputs, U, METRIC)
        elif METRIC == 'min_eig_inv_cov_appr':
            # print("Using inv_cov")
            metrics = -self.compute_inv_cov_metric(states, all_inputs, U, METRIC)
        elif METRIC == 'det_cov':
            # print("Using inv_cov")
            metrics = self.compute_inv_cov_metric(states, all_inputs, U, METRIC)
        elif METRIC == 'Trace_cov':
            metrics = self.compute_inv_cov_metric(states, all_inputs, U, METRIC)
        else:
            print('metric not implemented')

        return metrics

    # def metric(self, w_set, states, optim_agent):
    #     agent = Agent()
    #     U = agent.unicycle
    #     for j in range(self.MPC_horizon):
    #         for n in range(self.nb_agents):
    #             if n == int(optim_agent):
    #                 inputs = np.array([[self.vehicles[n].v], [w_set[j]]])
    #             else:
    #                 inputs = np.array([[self.vehicles[n].v], [self.vehicles[n].omega]])
    #         #propagate the state
    #             states[:,j+1:j+2,n] = U.discrete_step(states[:,j:j+1,n],inputs,self.pred_intv)
    #     return self.compute_SAM_metric(states)
    
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
        # print(M[0][0])
        # time.sleep(0.05)       
        return 1/np.linalg.det(M)   # FIXME:    Occationally returns np.linalg.det(M) as 0
                                    #           resulting in divide by 0 error.
    
    def try_parallel(self, i,optim_agent,states,METRIC,U):
        all_inputs = np.zeros((self.vehicles[0].nu, self.MPC_horizon, self.nb_agents))
        for j in range(self.MPC_horizon):
            for n in range(self.nb_agents):
                if n == int(optim_agent):
                    inputs = np.array([[self.vehicles[n].v], [self.w_set[i,j]]])
                else:
                    inputs = np.array([[self.vehicles[n].v], [self.vehicles[n].omega]])
                all_inputs[:,j,n] = inputs.reshape((2,))
            #propagate the state
                states[:,j+1:j+2,n] = U.discrete_step(states[:,j:j+1,n],inputs,self.pred_intv)
        if METRIC == 'SAM':
            metrics=self.compute_SAM_metric1(states,all_inputs)
        elif METRIC == 'obsv':
            metrics=self.compute_obsv_metric(states, self.w_set[i,:], optim_agent)
        elif METRIC == 'min_eig_inv_cov':
            # print("Using inv_cov")
            metrics = self.compute_inv_cov_metric(states, all_inputs, U, METRIC)
        elif METRIC == 'det_inv_cov':
            # print("Using inv_cov")
            metrics = self.compute_inv_cov_metric(states, all_inputs, U, METRIC)
        else:
            print('metric not implemented')
        return metrics
    
    def Trajoptim(self, optim_agent, METRIC='obsv'):

        if optim_agent == None:
            for n in range(self.nb_agents):
                self.vehicles[n].update_controller()
        else:
            for n in range(self.nb_agents):
                if n != optim_agent:
                    self.vehicles[n].update_controller()
            w = 0
            agent = Agent()
            U = agent.unicycle
            states = np.zeros((self.vehicles[0].nx, self.MPC_horizon, self.nb_agents))
            inputs = np.zeros((self.vehicles[0].nu, self.MPC_horizon, self.nb_agents))
            for n in range(self.nb_agents):
                states[0:2, 0:1, n] = self.vehicles[n].states_est[0:2, :]
                states[2, 0:1, n] = self.vehicles[n].states_est[2, :]
            # numerical optimization
            res_metric = []
            plan = []
            NN = 1
            ws = np.linspace(-self.omega_max, self.omega_max, NN)
            for ind in range(NN):
                #w_set = np.random.randn((self.MPC_horizon)) #* 0
                #w_set = np.clip(w_set, -self.omega_max, self.omega_max)
                w_set = 0*np.ones(self.MPC_horizon-1) * ws[ind]

                #w_set = np.random.uniform(-self.omega_max, self.omega_max, self.MPC_horizon)
                bnds = tuple(repeat((-self.omega_max, self.omega_max), self.MPC_horizon-1))
                # if METRIC == 'SAM':
                #     print('use pre-planned traj from SAM')
                #     w_set = np.array(([-0.17453292, -0.17448428, -0.17448249, 0.01821928, 0.02025531,
                #                        0.01748351, 0.01445456, 0.01174429, 0.00953385, 0.00780189,
                #                        0.00650251, 0.00559879, 0.00507023, 0.00490365, 0.00509926,
                #                        0.00566233, 0.0066109, 0.00796162, 0.00973551, 0.01193295,
                #                        0.01454627, 0.01753458, 0.02084326, 0.02444794, 0.02833526,
                #                        0.03195438, 0.03406161, 0.03220831, 0.02050405])) # optimized SAM
                # elif METRIC == 'det_cov':
                #     print('use pre-planned traj from det_cov')
                #     w_set = np.array([0.17451861, 0.04903283, 0.12632797, 0.05915758, 0.01967685,
                #                       0.01545765, 0.0116318, 0.00837392, 0.00612913, 0.00479835,
                #                       0.00398573, 0.00323476, 0.00335547, 0.00335733, 0.00313042,
                #                       0.00316596, 0.0030394, 0.00341439, 0.00472231, 0.00675415,
                #                       0.00899736, 0.00965029, 0.00639418, -0.00151485, -0.0076188,
                #                       -0.00313437, 0.0059965, -0.01254097,
                #                       0.09864577])  # optimal from determinant of EKF cov
                # else:
                #     pass
                # w_set = np.array([0.17451861, 0.04903283, 0.12632797, 0.05915758, 0.01967685,
                #                   0.01545765, 0.0116318, 0.00837392, 0.00612913, 0.00479835,
                #                   0.00398573, 0.00323476, 0.00335547, 0.00335733, 0.00313042,
                #                   0.00316596, 0.0030394, 0.00341439, 0.00472231, 0.00675415,
                #                   0.00899736, 0.00965029, 0.00639418, -0.00151485, -0.0076188,
                #                   -0.00313437, 0.0059965, -0.01254097,
                #                   0.09864577])  # optimal from determinant of EKF cov
                # metric1= self.metric(w_set, states, optim_agent, METRIC)

                #res = minimize(self.metric, w_set, args=(states, optim_agent, METRIC), bounds=bnds, method='Powell')
                # print("Made it")

                res = anneal(self.metric, bounds = bnds, args=(states, optim_agent, METRIC))
                print("Made it")

                #print(res.message)
                res_metric.append(res.fun)
                plan.append(res.x)
            self.w_plan = plan[np.argmin(np.array(res_metric))]
            #self.w_plan = w_set

            metric0 = self.metric(self.w_plan, states, optim_agent, METRIC)
            self.metric0 = metric0
        #     self.w_plan = np.array(([-0.17453292, -0.17448428, -0.17448249,  0.01821928,  0.02025531,
        # 0.01748351,  0.01445456,  0.01174429,  0.00953385,  0.00780189,
        # 0.00650251,  0.00559879,  0.00507023,  0.00490365,  0.00509926,
        # 0.00566233,  0.0066109 ,  0.00796162,  0.00973551,  0.01193295,
        # 0.01454627,  0.01753458,  0.02084326,  0.02444794,  0.02833526,
        # 0.03195438,  0.03406161,  0.03220831,  0.02050405]))

            # set the control
            # self.vehicles[int(optim_agent)].omega = w

    # cooeprative MPC
    def MPC(self, optim_agent, use_cov = False, METRIC = 'obsv'):
        if use_cov == False:
            if optim_agent == None:
                for n in range(self.nb_agents):             
                    self.vehicles[n].update_controller()
            else:
                for n in range(self.nb_agents):                
                    if n != optim_agent:
                        self.vehicles[n].update_controller()
                self.w_set = self.gen_w()
                w = 0
                agent = Agent()
                U = agent.unicycle
                states = np.zeros((self.vehicles[0].nx, self.MPC_horizon, self.nb_agents))
                for n in range(self.nb_agents):
                    states[0:2,0:1,n] = self.vehicles[n].states_est[0:2,:]
                    states[2,0:1,n] = self.vehicles[n].states_est[2,:]
                # N_total = self.w_set.shape[0] #M**3
                # metrics = np.zeros((N_total, 1))
                # metrics = Parallel(n_jobs=num_cores)(delayed(self.try_parallel)(i, optim_agent, states, METRIC,U)for i in range(N_total))
                # metrics = [self.try_parallel(i, optim_agent, states, METRIC, U) for i in range(N_total)]
                # FIXME: Why is this running so many times?
                # for i in range(N_total):
                #     for j in range(self.MPC_horizon):
                #         for n in range(self.nb_agents):
                #             if n == int(optim_agent):
                #                 inputs = np.array([[self.vehicles[n].v], [self.w_set[i,j]]])
                #             else:
                #                 inputs = np.array([[self.vehicles[n].v], [self.vehicles[n].omega]])
                #         #propagate the state
                #             states[:,j+1:j+2,n] = U.discrete_step(states[:,j:j+1,n],inputs,self.pred_intv)
                #     if METRIC == 'SAM':
                #         metrics[i,0]=self.compute_SAM_metric(states)
                #     elif METRIC == 'obsv':
                #         metrics[i,0]=self.compute_obsv_metric(states, self.w_set[i,:], optim_agent)
                #     else:
                #         print('metric not implemented')
                #         pass
                # m_min = np.argmin(metrics)
                # w = self.w_set[m_min,0]

                #numerical optimization
                w_set = np.random.randn((self.MPC_horizon))# * 0
                w_set = np.clip(w_set, -self.omega_max, self.omega_max)
                bnds = tuple(repeat((-self.omega_max, self.omega_max), self.MPC_horizon))
                options = {'maxiter': 100}
                res = minimize(self.metric, w_set, args = (states, optim_agent), bounds = bnds, method='Powell', options=options)
                w = res.x[0]
                
                # PSO Needs to write the vectorized metric function
                # w_max = self.omega_max * np.ones(self.MPC_horizon)
                # w_min = - w_max
                # bounds = (w_min, w_max)
                # options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
                # optimizer = GlobalBestPSO(n_particles=10, dimensions=self.MPC_horizon, options=options, bounds=bounds)
                # cost, pos = optimizer.optimize(self.metric, iters=1000, states = states, optim_agent = optim_agent)

                # set the control
                self.vehicles[int(optim_agent)].omega = w
        else:
            # use covariance information
            agent = Agent()
            U = agent.unicycle
            active_planning = False
            for n in range(self.nb_agents):
                if self.vehicles[n].states_cov[0,0] +  self.vehicles[n].states_cov[1,1] > 5: # needs planning
                    active_planning = True
                    break
                else:
                    pass
            if active_planning == False:
                for n in range(self.nb_agents):             
                    self.vehicles[n].update_controller()
            else:
                if optim_agent == None:
                    optim_agent = np.random.randint(0,self.nb_agents)
                    
                # repeat for the case use_cov = False and optim_agent not None
                for n in range(self.nb_agents):                
                    if n != optim_agent:
                        self.vehicles[n].update_controller()
                self.w_set = self.gen_w()
                w = 0
                agent = Agent()
                U = agent.unicycle
                states = np.zeros((self.vehicles[0].nx, self.MPC_horizon, self.nb_agents))
                for n in range(self.nb_agents):
                    states[0:2,0:1,n] = self.vehicles[n].states_est[0:2,:]
                    states[2,0:1,n] = self.vehicles[n].states_est[2,:]
                N_total = self.w_set.shape[0]
                metrics = np.zeros((N_total, 1))
                for i in range(N_total):
                    for j in range(self.MPC_horizon):
                        for n in range(self.nb_agents):
                            if n == int(optim_agent):
                                inputs = np.array([[self.vehicles[n].v], [self.w_set[i,j]]])
                            else:
                                inputs = np.array([[self.vehicles[n].v], [self.vehicles[n].omega]])
                        #propagate the state
                            states[:,j+1:j+2,n] = U.discrete_step(states[:,j:j+1,n],inputs,self.pred_intv)
                    metrics[i,0]=self.compute_SAM_metric(states)
                m_min = np.argmin(metrics)
                w = self.w_set[m_min,0]
                # set the control
                self.vehicles[int(optim_agent)].omega = w
        # print(self.vehicles[int(optim_agent)].omega)
    
    def compute_SAM_metric2(self, states, all_inputs, METRIC):
        adjacency = self.vehicles[0].adjacency
        range_meas_dim = 0
        for j in range(self.nb_agents):
            neighbor_idx_set = np.nonzero(adjacency[j, :])[0]
            range_meas_dim  += neighbor_idx_set.shape[0]

        if self.nb_landmarks > 0:
            nb = len(self.landmark_agents)
            lm_meas_dim = nb * self.nb_landmarks
            Jl = np.zeros(((self.MPC_horizon-1) * lm_meas_dim, (self.MPC_horizon) * self.nx * self.nb_agents))
        # self.MPC_horizon = (K+1) in the writeup

        Jd = np.eye((self.MPC_horizon)*self.nx*self.nb_agents)
        Jh = np.zeros(((self.MPC_horizon-1) * range_meas_dim, (self.MPC_horizon) * self.nx * self.nb_agents))

        for i in range(1,self.MPC_horizon):
            state = states[:, i, :].squeeze()
            if i == 1:
                Qh = self.std_range ** 2 * np.eye(range_meas_dim)
            else:
                Qh = sc.linalg.block_diag(Qh, self.std_range ** 2 * np.eye(range_meas_dim))
            row_idx1 = (range_meas_dim) * (i - 1)
            row_idx2 = (range_meas_dim) * (i)
            col_idx1 = (self.nx * self.nb_agents) * (i)
            col_idx2 = (self.nx * self.nb_agents) * (i + 1)
            Jh[row_idx1:row_idx2, col_idx1:col_idx2] = self.compute_range_jac(state)

        if self.nb_landmarks > 0:
            for i in range(1,self.MPC_horizon):
                state = states[:, i, :].squeeze()
                if i == 1:
                    Ql = self.std_range ** 2 * np.eye(lm_meas_dim)
                else:
                    Ql = sc.linalg.block_diag(Ql, self.std_range ** 2 * np.eye(lm_meas_dim))
                row_idx1 = (lm_meas_dim) * (i - 1)
                row_idx2 = (lm_meas_dim) * (i)
                col_idx1 = (self.nx * self.nb_agents) * (i)
                col_idx2 = (self.nx * self.nb_agents) * (i + 1)
                Jl[row_idx1:row_idx2, col_idx1:col_idx2] = self.compute_lm_range_jac(state)

        for j in range(0, self.MPC_horizon-1):
            state_j = states[:, j, :].squeeze()
            inputs = all_inputs[:, j, :].squeeze()
            # state_j1 = states[:,j+1,:].squeeze()
            for n in range(self.nb_agents):
                if n == 0:
                    F = self.compute_individual_jac1(state_j[:, n:n + 1], inputs[:, n:n + 1], self.pred_intv)
                else:
                    F = sc.linalg.block_diag(F, self.compute_individual_jac1(state_j[:, n:n + 1], inputs[:, n:n + 1], self.pred_intv))
            if j == 0:
                Qd = (self.pred_intv / self.Delta_t) * np.kron( np.eye(self.nb_agents), self.vehicles[0].S_Q.T @ self.vehicles[0].S_Q)
            else:
                Qd = sc.linalg.block_diag(Qd,(self.pred_intv / self.Delta_t) * np.kron( np.eye(self.nb_agents), self.vehicles[0].S_Q.T @ self.vehicles[0].S_Q))

            row_idx1 = (self.nx*self.nb_agents)*(j+1)
            row_idx2 = (self.nx*self.nb_agents)*(j+2)
            col_idx1 = (self.nx * self.nb_agents) * (j)
            col_idx2 = (self.nx * self.nb_agents) * (j + 1)
            Jd[row_idx1:row_idx2, col_idx1:col_idx2] = - F

        for n in range(self.nb_agents):
            if n == 0:
                P0 = self.vehicles[n].states_cov
            else:
                P0 = sc.linalg.block_diag(P0, self.vehicles[n].states_cov)


        Qd = sc.linalg.block_diag(P0,Qd)
        if self.nb_landmarks > 0:
            J = np.vstack((Jd, Jh, Jl))
            Q = sc.linalg.block_diag(Qd, Qh, Ql)
        else:
            J = np.vstack((Jd, Jh))
            Q = sc.linalg.block_diag(Qd, Qh)
        M = J.T @ np.linalg.inv(Q) @ J
        # J_final = np.vstack((Jd[-1:,:], Jh[-1:,:]))
        # Q_final = sc.linalg.block_diag((self.pred_intv / self.Delta_t) * np.kron(np.eye(self.nb_agents),
        #                                                                          self.vehicles[0].S_Q.T @ self.vehicles[
        #                                                                              0].S_Q),
        #                                self.std_range ** 2 * np.eye(range_meas_dim))
        # M_final = J_final.T @ Q_final @ J_final
        M = 0.5* (M + M.T)
        self.finalP = np.linalg.inv(M)
        if METRIC == 'min_eig_SAM':
            return np.log(np.min(np.linalg.eigvals(M)).item())
        if METRIC == 'min_eig_SAM_appr':
            return np.sum((np.linalg.eigvals(M)) ** self.mu) ** (1.0/self.mu)
        if METRIC == 'SAM':
            self.metric0 = -np.sum(np.log(np.linalg.eigvals(M)))
            return -np.sum(np.log(np.linalg.eigvals(M))) #- np.log(np.linalg.det(M))#np.log(1 / np.linalg.det(M))
        if METRIC == 'Trace_inv_SAM':
            # L = np.linalg.cholesky(M)
            # return np.linalg.norm(np.linalg.inv(L), 'fro') ** 2
            return np.trace(np.linalg.inv(M))#np.sum(1/np.linalg.eigvals(M)) #np.trace(np.linalg.inv(M))
        if METRIC == 'cond_SAM':
            return np.log(np.max(np.linalg.eigvals(M)).item()) - np.log(np.min(np.linalg.eigvals(M)).item())
        if METRIC == 'last_SAM_cov':
            V = np.linalg.inv(M)
            return np.sum(np.log(np.linalg.eigvals(V[- self.nx*self.nb_agents:, - self.nx*self.nb_agents:])))    
    
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
        # print("SAM1 P1")
        for j in range(self.MPC_horizon - 1):
            state_j = states[:,j,:].squeeze()
            inputs = all_inputs[:,j,:].squeeze()
            #state_j1 = states[:,j+1,:].squeeze()
            if j == 0:
                Jd = self.compute_dynamics_jac1(state_j, inputs, j)
            else:
                jd = self.compute_dynamics_jac1(state_j, inputs, j)
                Jd = np.vstack((Jd, jd))
        # print("SAM1 P2")
        Qd = np.kron(np.eye(self.nb_agents * (self.MPC_horizon - 1)), self.vehicles[0].S_Q.T @ self.vehicles[0].S_Q)        
        J = np.vstack((Jh, Jd))
        Q = sc.linalg.block_diag(Qh, Qd)
        M = J.T @ np.linalg.inv(Q) @ J 
        # M = 0.5 * (M + M.T)
        if METRIC == 'min_eig_SAM':
            return np.min(np.real(np.linalg.eigvals(M))).item()
        if METRIC == 'min_eig_SAM_appr':
            return np.sum((np.linalg.eigvals(M)) ** (self.mu)) ** (1.0/self.mu)
        if METRIC == 'SAM':
            return 1/np.linalg.det(M)
        if METRIC == 'Trace_inv_SAM':
            return np.trace(np.linalg.inv(M))


    def compute_dynamics_jac1(self, statej, inputs, j):
        n_col = self.nx * self.MPC_horizon * self.nb_agents
        jac_size = 3
        Jd = np.zeros((self.nb_agents * jac_size, n_col))
        for n in range(self.nb_agents):
            # print("iter")
            j0 = self.compute_individual_jac1(statej[:,n:n+1], inputs[:,n:n+1], self.pred_intv)
            Jd[jac_size * n: jac_size * n + jac_size, j * self.nx * self.nb_agents + self.nx * n : j * self.nx * self.nb_agents + self.nx * n + self.nx] = - j0
            Jd[jac_size * n: jac_size * n + jac_size, (j+1) * self.nx * self.nb_agents + self.nx * n : (j + 1) * self.nx * self.nb_agents + self.nx * n + self.nx] = np.eye(self.nx)
        return Jd
    
    def compute_dynamics_u_jac1(self, statej, inputs, j):
        n_col = self.nu * self.MPC_horizon * self.nb_agents
        jac_size = 3
        Jd = np.zeros((self.nb_agents * jac_size, n_col))
        for n in range(self.nb_agents):
            j0 = self.compute_individual_u_jac1(statej[:,n:n+1], inputs[:,n:n+1], self.pred_intv)
            Jd[jac_size * n: jac_size * n + jac_size, j * self.nu * self.nb_agents + self.nu * n : j * self.nu * self.nb_agents + self.nu * n + self.nu] = - j0
            # Jd[jac_size * n: jac_size * n + jac_size, (j+1) * self.nu * self.nb_agents + self.nu * n : (j + 1) * self.nu * self.nb_agents + self.nu * n + self.nu] = np.eye(self.nx)
        return Jd
        

    def compute_individual_jac1(self, statej, inputs, Delta_t):
        agent = Agent()
        unicycle = agent.unicycle
        J0 = unicycle.dyn_jacobian(statej, inputs, Delta_t)
        return J0

    def compute_individual_u_jac1(self, statej, inputs, Delta_t):
        agent = Agent()
        unicycle = agent.unicycle
        J0 = unicycle.u_jacobian(statej, inputs, Delta_t)
        return J0
    
    def compute_SAM_metric(self, states):
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
            state_j1 = states[:,j+1,:].squeeze()
            if j == 0:
                Jd = self.compute_dynamics_jac(state_j, state_j1, j)
            else:
                jd = self.compute_dynamics_jac(state_j, state_j1, j)
                Jd = np.vstack((Jd, jd))
        Qd = np.kron(np.eye(self.nb_agents * (self.MPC_horizon - 1)), np.diag([self.std_v**2, self.std_omega**2]))        
        J = np.vstack((Jh, Jd))
        Q = sc.linalg.block_diag(Qh, Qd)
        M = J.T @ np.linalg.inv(Q) @ J   
        return 1/np.linalg.det(M)

    def compute_inv_cov_metric(self,states,inputs,U, METRIC):
        # print('Using inverse covariance metric')
        # Uses the minimum eigenvalue of the inverse covariance matrix as discussed in Cognetti et. al, Optimal active sensing with process and measurement noise, ICRA, 2018.
        for i in range(self.MPC_horizon):
            state = states[:, i, :].squeeze()
            input = inputs[:, i, :].squeeze()
            H_rl = self.compute_range_jac(state)
            H_lm = self.compute_lm_range_jac(state)
            for n in range(self.nb_agents):
                if n == 0:
                    if i == 0:
                        P_predict= self.vehicles[n].states_cov
                        S_Q_block = self.S_Q
                    if i > 0:
                        A =  U.dyn_jacobian(states[:,i-1:i,n], inputs[:,i-1:i,n], self.pred_intv)#self.compute_dynamics_jac1(state, input, i)#
                    # B = U.u_jacobian(states[:, i:i + 1, n], inputs[:, i:i + 1, n], self.pred_intv)#self.compute_dynamics_u_jac1(state, input, i)#
                else:
                    if i == 0:
                        P_predict = sc.linalg.block_diag(P_predict, self.vehicles[n].states_cov)
                        S_Q_block = sc.linalg.block_diag(S_Q_block, self.S_Q)
                    if i > 0:
                        A = sc.linalg.block_diag(A, U.dyn_jacobian(states[:,i-1:i,n], inputs[:,i-1:i,n], self.pred_intv))#self.compute_dynamics_jac1(state, input, i))#
                    # B = sc.linalg.block_diag(B, U.u_jacobian(states[:,i:i+1,n], inputs[:,i:i+1,n],self.pred_intv))#self.compute_dynamics_u_jac1(state, input, i))#

            if i > 0:
            #     P_inv = np.linalg.inv(P0)

            # else:
                # A_update = -P_inv @ A - A.T @ P_inv
                # H_update = H.T @ np.linalg.inv(np.kron(np.eye(H.shape[0]),self.std_range)) @ H
                # B_update = -P_inv @ B @ np.linalg.inv(np.kron(np.eye(self.nb_agents), np.diag([self.std_v**2, self.std_omega**2])) )  @ B.T @ P_inv
                # P_inv = P_inv + self.Delta_t*(A_update + H_update + B_update)
                # for i in range(1, T):
                # Prediction
                # x_predict[:, i:i + 1] = phi @ x_filtered[:, i - 1:i] + psi @ u[:, i - 1:i]
                P_predict = A @ P_predict @ A.transpose() + (self.pred_intv/self.Delta_t)* (S_Q_block.T @ S_Q_block)

                # Update
                H = np.vstack((H_rl,H_lm))
                K = P_predict @ H.transpose() @ np.linalg.inv(H @ P_predict @ H.transpose() + np.kron(np.eye(H.shape[0]),self.std_range**2))
                P = (np.eye(self.nx*self.nb_agents) - K @ H) @ P_predict
                P_predict = P
                # print(P)
        self.finalP = P
        if METRIC == 'min_eig_inv_cov':
            return np.min(np.linalg.eigvals(np.linalg.inv(P))).item()
        if METRIC == 'min_eig_inv_cov_appr':
            return np.sum(np.linalg.eigvals(np.linalg.inv(P)) ** (-10)) ** (-1.0/10)
        if METRIC == 'det_cov':
            return np.sum(np.log(np.linalg.eigvals(P))) #1/np.linalg.det(np.linalg.inv(P))
        if METRIC == 'cond_cov':
            return np.log(np.min(np.linalg.eigvals(P)).item()) - np.log(np.max(np.linalg.eigvals(P)).item())
        if METRIC == 'Trace_cov':
            return np.trace(P)

    def compute_dynamics_jac(self, statej, statej1, j):
        n_col = self.nx * self.MPC_horizon * self.nb_agents
        col_start = j * self.nb_agents * self.nx
        jac_size = 2
        Jd = np.zeros((self.nb_agents * jac_size, n_col))
        for n in range(self.nb_agents):
            j0, j1 = self.compute_individual_jac(statej[:,n:n+1], statej1[:,n:n+1], self.pred_intv)
            Jd[jac_size * n: jac_size * n + jac_size, j * self.nx * self.nb_agents + self.nx * n : j * self.nx * self.nb_agents + self.nx * n + self.nx] = j0
            Jd[jac_size * n: jac_size * n + jac_size, (j+1) * self.nx * self.nb_agents + self.nx * n : (j + 1) * self.nx * self.nb_agents + self.nx * n + self.nx] = j1
        return Jd
                     
    def compute_individual_jac(self, statej, statej1, Delta_t):
        agent = Agent()
        unicycle = agent.unicycle
        J0, J1 = unicycle.find_u_jacobian(statej, statej1, Delta_t)
        return J0, J1
    
    def compute_lm_range_jac(self, state):
        nx = self.nx
        nb_agents = self.nb_agents
        nb_lm = self.nb_landmarks
        for ind, ego_idx in enumerate(self.landmark_agents): #range(nb_agents)
            vehicle_pos = state[0:2, ego_idx:ego_idx + 1]
            for n in range(nb_lm):
                jac = np.zeros((1, nx * nb_agents))
                lm_loc = self.landmarks[:,n:n+1]
                range_ = np.linalg.norm(vehicle_pos - lm_loc)
                jac[:, ego_idx * nx:((ego_idx + 1) * nx) - 1] = -(lm_loc - vehicle_pos).transpose()
                if ind == 0 and n == 0:
                    if range_ != 0:
                        jacobians = (1. / range_) * jac
                    else:
                        jacobians = np.zeros((1, nx * nb_agents))
                else:
                    if range_ != 0:
                        jacobians = np.vstack((jacobians, (1. / range_) * jac))
                    else:
                        jacobians = np.vstack((jacobians, np.zeros((1, nx * nb_agents))))
        return jacobians
   
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

    # Set initial swarm position
    def set_swarm_initPos(self, pos):
        for i in range(self.nb_agents):
            self.vehicles[i].set_initPos(pos[:,i:i+1])

    # Set final swarm position
    def set_swarm_endpos(self, pos):
        for i in range(self.nb_agents):
            self.vehicles[i].set_endPos(pos[:,i:i+1])

    # Set initial swarm pose
    def set_swarm_initPose(self, pose):
        for i in range(self.nb_agents):
            self.vehicles[i].set_initPose(pose[:,i:i+1])

    # Set final swarm pose
    def set_swarm_endpose(self, pose):
        for i in range(self.nb_agents):
            self.vehicles[i].set_endPose(pose[:,i:i+1])

    # Set swarm waypoints
    def set_swarm_waypoints(self, waypoints):
        for i in range(self.nb_agents):
            self.vehicles[i].waypoints = waypoints[i]

    # Set swarm end states
    def set_swarm_endStates(self, states):
        for i in range(self.nb_agents):
            self.vehicles[i].set_endPos(states[:,i:i+1])

    # set initial estimates and covariance
    def set_swarm_estimates(self, estimates, cov):
        for i in range(self.nb_agents):
            self.vehicles[i].states_est = estimates[:, i:i+1]
            self.vehicles[i].states_cov = cov[self.nx * i:self.nx * (i+1), self.nx * i:self.nx * (i+1)]
    
    # Update swarm states
    def update_state(self, time):
        for i in range(self.nb_agents):
            self.vehicles[i].update_state(time) #TODO: See function

    def addToLoop_estimator(self):
        self.vehicles[0].use_estimation = True
        self.vehicles[1].use_estimation = True

        #GTSAM
        self.vehicles[0].states_est = self.est_X[:3,:]
        self.vehicles[1].states_est = self.est_X[3:,:]

    # Update swarm adjacency matrix
    def update_adjacency(self, adjacency):
        for i in range(self.nb_agents):
            self.vehicles[i].adjacency = adjacency
            self.vehicles[i].measRange_history = np.empty((self.nb_agents,1))
            self.vehicles[i].count = 0
            if self.nb_landmarks > 0:
                self.vehicles[i].lm_meas_history = np.empty((self.nb_landmarks,1))

    # Generate swarm range measurements
    def update_measRange(self):
        #update range sensor measurements
        measRange_matrix = np.zeros((self.nb_agents, self.nb_agents))
        # TODO: Update the measRange_matrix using range measurements from the simulator
        for j in range(self.nb_agents):
            # print(self.vehicles[j].adjacency)
            for jj in range(self.nb_agents):
                if self.vehicles[j].adjacency[j,jj] == 1:
                    vehicle_pos = self.vehicles[j].states[:2,:]
                    neighbor_pos = self.vehicles[jj].states[:2,:]
                    meas_range  =  np.linalg.norm(vehicle_pos - neighbor_pos) + self.vehicles[j].std_range*np.random.randn()
                    # if measRange_matrix[j, jj] == 0 and measRange_matrix[jj,j] == 0:
                    measRange_matrix[j,jj] = meas_range
        # #

        for j in range(self.nb_agents):
            self.vehicles[j].measRange_history = np.hstack((self.vehicles[j].measRange_history, measRange_matrix[j:j+1,:].transpose()))
        
        if self.nb_landmarks > 0:
            lm_measurements = np.zeros((self.nb_agents, self.nb_landmarks))
            for j in range(self.nb_agents):
                vehicle_pos = self.vehicles[j].states[:2, :]
                for i in range(self.nb_landmarks):
                    lm_loc = self.landmarks[:,i:i+1]
                    lm_range = np.linalg.norm(lm_loc - vehicle_pos) + self.vehicles[j].std_range*np.random.randn()
                    lm_measurements[j,i] = lm_range
                # if len(self.vehicles[j].lm_meas_history) == 0:
                #     self.vehicles[j].lm_meas_history = lm_measurements[j:j+1,:].transpose
                # else:
                self.vehicles[j].lm_meas_history = np.hstack((self.vehicles[j].lm_meas_history, lm_measurements[j:j+1,:].transpose()))

    # Generate swarm states history
    def  get_swarm_states_history_(self):
        self.get_swarm_states_history = []
        for i in range(self.nb_agents):
            self.vehicles[i].states_history = np.delete(self.vehicles[i].states_history, 0, 1)
            self.get_swarm_states_history.append(self.vehicles[i].states_history)
    
    def  get_swarm_est_err_(self):
        self.EST_ERR = []
        for i in range(self.nb_agents):
            self.vehicles[i].est_err = np.delete(self.vehicles[i].est_err, 0, 1)
            self.EST_ERR.append(self.vehicles[i].est_err)

    # Plot swarm states history
    def plot_swarm_traj(self):
        for i in range(self.nb_agents):
            states = self.get_swarm_states_history[i]
            time = self.vehicles[i].sim_t
            plt.plot(states[0,:], states[1,:], label= 'Vehicle '+str(i))
            # plt.quiver(states[0,:], states[1,:], np.cos(states[2,:]), np.sin(states[2,:]), scale= 20)
        plt.legend()
        plt.title('Vehicle trajectories')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.show()

    # Plot swarm heading
    def plot_swarm_heading(self):
        for i in range(self.nb_agents):
            states = self.get_swarm_states_history[i]
            time = self.vehicles[i].sim_t
            plt.plot(time, states[2,1:], label= 'Vehicle '+str(i))
        plt.legend()
        plt.title('Vehicle headings')
        plt.xlabel('time (s)')
        plt.ylabel('$\\theta$ (rad)')
        plt.show()




