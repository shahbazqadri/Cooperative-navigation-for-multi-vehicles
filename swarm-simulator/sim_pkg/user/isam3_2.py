'''
Incremental smoothing and mapping (ISAM2) implementation for multiagent estimation discussed in
Rutkowski, Adam J., Jamie E. Barnes, and Andrew T. Smith. "Path planning for optimal cooperative navigation." 2016 IEEE/ION Position, Location and Navigation Symposium (PLANS). IEEE, 2016.

Authors: Shahbaz P Qadri Syed, He Bai, Ben Sailor
'''
'''
Questions:
is there another function that needs to be called to calculate the velocity?
'''

import numpy as np
import scipy as sc
from user.agent import Agent
from user.Swarm import Swarm
import gtsam
from typing import Optional, List
from functools import partial
import matplotlib.pyplot as plt
from decimal import Decimal as D
import random
from scipy.io import savemat

# from user import robot_movement as rm

my_seed = 10
random.seed(my_seed)
np.random.seed(my_seed)
myMetric = 'SAM'

MC_TOTAL = 20

TRUTH = []
EST = []
ERR = []
COV = []
myM = []
nb_landmarks = 2        # The number of landmarks must be changed with changes made to init_pose.py
nb_agents = 4           # Same with the number of landmarks

landmarks = np.array([[2., 10.], [2., -5.]])    # The number of landmarks must be correlated number of coordinates
                                                # Same with the drone positions
#landmarks = [np.zeros((nb_landmarks, nb_landmarks)) if nb_landmarks > 0 else []]

def usr(robot):
    import time as time
    # diff_drive = robot.DiffDrive()
    # diff_drive.set_limits(0.05,0.05)
    
# for MC_RUN in range(MC_TOTAL):
    # print(MC_RUN)
    id_var = robot.id
    opt_agent = id_var
    adj_type = 'd_ring'
    if id_var != 4 and id_var != 5:

        #print('Initializing agents.........')
        agent = Agent()
        unicycle = agent.unicycle
        nx = 3
        nu = 2
        Delta_t   = 0.1
        T = 20 #s
        # finding the number of decimal places of Delta_t
        precision = abs(D(str(Delta_t)).as_tuple().exponent)
        t         = np.arange(0,T,Delta_t)
        t = np.round(t, precision) # to round off python floating point precision errors
        tinc = 1.0#0.5 #sec
        # vel         = 0.031 #m/s
        vel         = 3 #m/s
        #omega_max = 10 #degrees/s
        std_omega = 0*np.deg2rad(0.57) #degrees/s
        std_v     = 0*0.01 #* 10 #m/s
        std_range = 0.01 * 100 #m
        S_Q = np.diag([0.1, 0.1, 0.01]) * Delta_t

        #TODO: frequency of range and odometry measurements from simulator
        f_range   = 10 #Hz
        f_odom    = 10 #Hz
        # #

        f_waypt  = 1
        estimated_states_history = []
        k_old = 0
        
        if adj_type == 'full':
            adjacency = np.ones((nb_agents, nb_agents)) - np.eye(nb_agents)
        elif adj_type == 'd_ring':
            adjacency = np.array([[0 if i != (j+1) % nb_agents else 1 for i in range(nb_agents)] for j in range(nb_agents)])
            adjacency = np.array([[0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1], [1, 0,  0, 0]])
        #TODO: number of agents in the simulation and the adjacency graph of range measurement
        # nb_agents = 5
        # adjacency = np.array([[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1],[1,0,0,0,0]])
        # #
        #adjacency = np.array([[0,1,1,0,0],[0,0,1,0,0],[0,0,0,1,0],[1,0,0,0,1],[1,0,0,0,0]])#

        #adjacency = (np.ones((nb_agents, nb_agents)) - np.eye(nb_agents))#np.array([[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1],[1,0,0,0,0]])#(np.ones((nb_agents, nb_agents)) - np.eye(nb_agents))  
        # Set initial and end position
        # pos0 = np.array([[0.,10.,10.,0.],[0.,0.,10.,10.]])
        # posf = np.array([[30., 40., 40., 30.],[30., 30., 40., 40.]])

        # TODO: Initial and final positions for simulator
        pos0 = np.array([[0., 0.3, 0.3, 0.], [0., 0., 0.3, 0.3]])
        posf = pos0 + np.ones((2,nb_agents)) * vel * 30 #100*np.array([[30., 40., 40., 30.,50.],[30., 30., 40., 40., 40.]])
        if nb_landmarks > 0:
            landmarks = np.array([[2., -5.], [10., 5.]])
            landmarks = landmarks[:,0:nb_landmarks]

        swarm =  Swarm()
        swarm.add_landmarks(landmarks)
        swarm.update_adjacency(adjacency)
        for i in range(nb_agents):
            swarm.add_vehicle(id_var, Delta_t, t, vel, std_omega, std_v, std_range, f_range, f_odom, S_Q=S_Q)
        swarm.set_swarm_initPos(pos0)
        swarm.set_swarm_endpos(posf)
        swarm.agent_to_landmarks([opt_agent])

        #print('Done.')
        
        #print('Propagating true state and generating measurements........')
        swarm.update_adjacency(adjacency)
        
        # # propagate the swarm system
        # for tt in t:
        #     # update vehicle states and plot
        #     swarm.update_state(tt)
        #     swarm.update_measRange()
        
        # for i in range(nb_agents):
        #     swarm.vehicles[i].meas_history = np.delete(swarm.vehicles[i].meas_history, 0, 1)
        #     swarm.vehicles[i].measRange_history = np.delete(swarm.vehicles[i].measRange_history, 0, 1)
        #     if i == 0:
        #         meas_history = swarm.vehicles[i].meas_history
        #     else:
        #         meas_history = np.vstack((meas_history, swarm.vehicles[i].meas_history))
        
        # swarm.get_swarm_states_history_()
        # swarm.plot_swarm_traj()
        # get_swarm_states_history = swarm.get_swarm_states_history
        
        #print('Initializing factor graph...........')
        S0 = 1e-4*np.eye(nx*nb_agents)
        prior_noise = gtsam.noiseModel.Gaussian.Covariance(S0)
        dynamics_noise = gtsam.noiseModel.Constrained.Sigmas(np.array([std_v*Delta_t, 0., std_omega*Delta_t]*nb_agents).reshape(nx*nb_agents,1)) # in the body frame of each agent
        cov = np.kron(np.eye(nb_agents), np.diag([std_v**2, std_omega**2]))
        input_noise = gtsam.noiseModel.Gaussian.Covariance(cov)
        for j in range(nb_agents):
            if j == 0:
                S = swarm.vehicles[j].S_Q.T @ swarm.vehicles[j].S_Q
            else:
                S = sc.linalg.block_diag(S, swarm.vehicles[j].S_Q.T @ swarm.vehicles[j].S_Q)
        process_noise = gtsam.noiseModel.Gaussian.Covariance(S)
        # Create an empty Gaussian factor graph
        graph = gtsam.NonlinearFactorGraph()
        
        
        
        def parse_result(result, cov, nb_agents, t):
            x_res = []
            for j in range(nb_agents):
                x_sol = np.zeros((len(t), nx))
                for k in range(len(t)):
                    x_sol[k, :] = result.atVector(X[k])[j * nx:(j + 1) * nx]
                swarm.vehicles[j].states_est = x_sol[-1,:].reshape((3,1))
                swarm.vehicles[j].states_cov = cov[j*nx:(j+1)*nx, j*nx:(j+1)*nx]
                swarm.vehicles[j].est_err = np.hstack((swarm.vehicles[j].est_err, swarm.vehicles[j].states_est - swarm.vehicles[j].states))
                #print(swarm.vehicles[j].states_est.shape)
                x_res.append(x_sol.T)
            return x_res
        
        
        def error_dyn1( measurements, this: gtsam.CustomFactor,
                    values: gtsam.Values,
                    jacobians: Optional[List[np.ndarray]]):
        
            key1 = this.keys()[0]
            key2 = this.keys()[1]
        
            X_, Xp1 = values.atVector(key1), values.atVector(key2)
            u = measurements.reshape(nu*nb_agents,1)
            u_cal = np.zeros((nu*nb_agents,1))
            for j in range(nb_agents):
                u_cal[j*nu:(j+1)*nu, :] = unicycle.find_u(X_[j*nx:(j+1)*nx].reshape(nx,1), Xp1[j*nx:(j+1)*nx].reshape(nx,1), Delta_t)
            
            if jacobians is not None:
                for j in range(nb_agents):
                    jac0, jac1 = unicycle.find_u_jacobian(X_[j*nx:(j+1)*nx].reshape(nx,1), Xp1[j*nx:(j+1)*nx].reshape(nx,1), Delta_t)
                    if j == 0:
                        jacobians[0] = jac0
                        jacobians[1] = jac1
                    else:
                        jacobians[0] = sc.linalg.block_diag(jacobians[0], jac0)
                        jacobians[1] = sc.linalg.block_diag(jacobians[1], jac1)

            
            error = (u_cal - u).reshape(nu*nb_agents,)
        
            return error
        # landmark range factor

        # Dynamics factor
        def error_dyn( measurements, this: gtsam.CustomFactor,
                    values: gtsam.Values,
                    jacobians: Optional[List[np.ndarray]]):
        
            key1 = this.keys()[0]
            key2 = this.keys()[1]
        
            X_, Xp1 = values.atVector(key1), values.atVector(key2)
            #m_x = np.zeros((nx*nb_agents,1))
            x = np.zeros((nx*nb_agents,1))
            u = measurements.reshape(nu*nb_agents,1)
            
            # for j in range(nb_agents):
            #     inputs = u[j*nu:(j+1)*nu,:]
            #     v = inputs[0,:]
            #     omega = inputs[1,:]
            #     dth = omega * Delta_t
            #     mu = v * Delta_t * np.sinc(1/np.pi*(dth/2))
            #     m_x[j*nx:(j+1)*nx, :] = np.array([mu * np.cos(dth/2), mu * np.sin(dth/2), dth]).reshape((nx,1))
            #     th = X_[j*nx+2]
            #     rot = np.array([[np.cos(th), -np.sin(th), 0.], [np.sin(th), np.cos(th), 0.], [0.,0.,1.]])
            #     drot = np.array([[-np.sin(th), -np.cos(th), 0.], [np.cos(th), -np.sin(th), 0.], [0.,0.,0.]])
            #     if j == 0:
            #         Rot = rot
            #         dRot = drot
            #     else:
            #         Rot = sc.linalg.block_diag(Rot, rot)
            #         dRot = sc.linalg.block_diag(dRot, drot)
            
            # error = Rot.T @ (Xp1 - X_ ) - m_x.reshape(nx*nb_agents,)
            # derror = Rot @ dRot.T @ (Xp1 - X_).reshape(nx*nb_agents,1)
            # dJac = np.hstack((np.zeros((nx*nb_agents,nx-1)), derror)) * 0
            
            # if jacobians is not None:
            #     jacobians[1] = Rot.T#np.eye(nx*nb_agents)
            #     for j in range(nb_agents):
            #         if j == 0:
            #             jac = - np.eye(nx) + dJac[j*nx:(j+1)*nx,:]
            #         else:
            #             jac = sc.linalg.block_diag(jac, - np.eye(nx) +
            #                                         dJac[j*nx:(j+1)*nx,:]) 
        
            #     jacobians[0] = Rot.T * jac 
        
        
            for j in range(nb_agents):
                x[j*nx:(j+1)*nx, :] = unicycle.discrete_step(X_[j*nx:(j+1)*nx].reshape(nx,1), u[j*nu:(j+1)*nu,:], Delta_t)
                # th = x[j*nx+2,0] # current heading
                # rot = np.array([[np.cos(th), -np.sin(th), 0.], [np.sin(th), np.cos(th), 0.], [0.,0.,1.]])
                # drot = np.array([[-np.sin(th), -np.cos(th), 0.], [np.cos(th), -np.sin(th), 0.], [0.,0.,0.]])
                # if j == 0:
                #     Rot = rot
                #     dRot = drot
                # else:
                #     Rot = sc.linalg.block_diag(Rot, rot)
                #     dRot = sc.linalg.block_diag(dRot, drot)

                    
        
        
            error = (Xp1 - x.reshape(nx*nb_agents,))   
            if jacobians is not None:
                jacobians[1] = np.eye(nx*nb_agents)
                for j in range(nb_agents):
                    if j == 0:
                        jac = unicycle.dyn_jacobian(X_[j*nx:(j+1)*nx].reshape(nx,1), u[j*nu:(j+1)*nu,:], Delta_t)
                    else:
                        jac = sc.linalg.block_diag(jac, unicycle.dyn_jacobian(X_[j*nx:(j+1)*nx].reshape(nx,1), u[j*nu:(j+1)*nu,:], Delta_t))  
                jacobians[0] = - jac 
        
            return error

        def error_range_lm(ego_idx, landmarks, measurement, this: gtsam.CustomFactor,
                        values: gtsam.Values,
                        jacobians: Optional[List[np.ndarray]]):

            key1 = this.keys()[0]

            X_ = values.atVector(key1)
            n = measurement.shape[0]

            range_est = np.zeros((n, 1))
            vehicle_pos = X_[ego_idx * nx:((ego_idx + 1) * nx) - 1].reshape(2, 1)

            for j in range(n):
                jac = np.zeros((1, nx * nb_agents))
                lm_loc = landmarks[:,j:j+1]
                range_ = np.linalg.norm(vehicle_pos - lm_loc)
                range_est[j, :] = range_
                jac[:, ego_idx * nx:((ego_idx + 1) * nx) - 1] = (- lm_loc + vehicle_pos).transpose()

                if jacobians is not None:

                    if j == 0:
                        if range_ != 0:
                            jacobians[0] = (1. / range_) * jac
                        else:
                            jacobians[0] = np.zeros((1, nx * nb_agents))
                    else:
                        if range_ != 0:
                            jacobians[0] = np.vstack((jacobians[0], (1. / range_) * jac))
                        else:
                            jacobians[0] = np.vstack((jacobians[0], np.zeros((1, nx * nb_agents))))

            error = (range_est - measurement.reshape(n, 1)).reshape(n, )

            return error

        # Range factor
        def error_range(ego_idx, neighbor_idx_set, measurement, this: gtsam.CustomFactor,
                    values: gtsam.Values,
                    jacobians: Optional[List[np.ndarray]]):
        
            key1 = this.keys()[0]
        
            X_ = values.atVector(key1)
            n = measurement.shape[0]
        
            range_est = np.zeros((n,1))
            vehicle_pos = X_[ego_idx * nx:((ego_idx + 1) * nx )- 1].reshape(2, 1)
        
            for j in range(n):
                jac = np.zeros((1, nx * nb_agents))
                neighbor_idx = neighbor_idx_set[j]
                neighbor_pos = X_[neighbor_idx*nx:(neighbor_idx+1)*nx-1].reshape(2, 1)
                range_ = np.linalg.norm(vehicle_pos - neighbor_pos)
                range_est[j,:] = range_
                jac[:,ego_idx*nx:((ego_idx+1)*nx)-1] = -(neighbor_pos - vehicle_pos).transpose()
                jac[:,neighbor_idx*nx:((neighbor_idx+1)*nx)-1] = -(-neighbor_pos + vehicle_pos).transpose()
        
                if jacobians is not None:
        
                    if j == 0:
                        if range_ != 0:
                            jacobians[0] = (1. / range_) * jac
                        else:
                            jacobians[0] = np.zeros((1, nx * nb_agents))
                    else:
                        if range_!= 0:
                            jacobians[0] = np.vstack((jacobians[0],(1. / range_) * jac))
                        else:
                            jacobians[0] = np.vstack((jacobians[0], np.zeros((1, nx * nb_agents))))
        
            error = (range_est - measurement.reshape(n,1)).reshape(n,)
        
            return error
        
        # Create the keys corresponding to unknown variables in the factor graph
        X = []
        for k in range(len(t)):
            X.append(gtsam.symbol('x', k))
        v = gtsam.Values()
        
        # set initial state as prior
        X0 = np.zeros((nx, nb_agents))
        #TODO: initial orientation of the agents (at pos0)
        theta0 = [0, 0, 0, 0, 0]

        for j in range(nb_agents):
            X0[:, j:j + 1] = np.vstack((pos0[:, j:j + 1], np.array([[theta0[j]]])))
        
        swarm.set_swarm_estimates(X0, S0)

        X0 = X0.T.flatten().reshape(nx*nb_agents,1)
        graph.add(gtsam.PriorFactorVector(X[0], X0, prior_noise))
        v.insert(X[0], X0)
        count = 0
        X_val = X0
        idx = 0
        isam = gtsam.ISAM2()
        initialized = False
        k = 0
        tt = t[k]
        start_time = robot.get_clock()
        current_time = start_time
        # print("id_vra: ", id_var)

        while k < len(t):
            while True:
                time.sleep(0.1)
                current_pos = robot.get_pose()
                if not current_pos:
                    # print("GOT A FALSE FLAG FROM CURRENT POS")
                    continue
                break

            current_state_est = np.array([current_pos[0], current_pos[1], current_pos[2]])
            data_to_send = f"id:{id_var};state_est:{current_state_est}"

            # FIXME:
            # print(swarm.vehicles[id_var].states_est)
            swarm.vehicles[id_var].set_est(current_state_est)

            # Send to all neighbors
            robot.send_msg(f"{data_to_send}")
            time.sleep(0.1)

            # Wait for responses from all other drones
            for _ in range(len(swarm.vehicles)):
                recv = robot.recv_msg(clear=False)
                if len(recv) > 0:
                    for msg in recv:
                        sender_id = int(msg.split(";")[0].split(":")[1])
                        state_est_str = msg.split(";")[1].split(":")[1]
                        state_est = np.array([float(x) for x in state_est_str.strip("[]").split()])
                        swarm.vehicles[sender_id].set_est(state_est)
                        # print(f"Drone {sender_id} state {state_est}")

                    
            if id_var == 0:
                robot.recv_msg(clear=True)
                # for i in range(len(swarm.vehicles)):
                    # print(swarm.vehicles[i].states_est)

            current_time = robot.get_clock()
            tt = current_time - start_time

            if tt < t[k]:
                # print("tt is < t[k]")
                continue

        
            # print(f"Drone {id_var} state_est {swarm.vehicles[id_var].states_est}")


            # ALL the drones
            tt = t[k]#k * Delta_t
            swarm.update_measRange() #TODO: See function
            swarm.update_state(tt) #TODO: See function 
            
            if k < len(t) - 1:
                # print("STOP POINT 1")
                # Dynamics factor
                odom_period = 1. / f_odom
                if D(str(t[k])) % D(str(odom_period)) == 0.:
                    # print("DYNAMICS FACTOR")
                    idx = D(str(t[k])) // D(str(odom_period))
                    idx_bias = D(str(t[0])) // D(str(odom_period))
                    for i in range(nb_agents):
                        # one drone at a time
                        swarm.vehicles[i].meas_history = np.delete(swarm.vehicles[i].meas_history, 0, 1)
                        if k == 0:
                            swarm.vehicles[i].measRange_history = np.delete(swarm.vehicles[i].measRange_history, 0, 1)
                            swarm.vehicles[i].lm_meas_history = np.delete(swarm.vehicles[i].lm_meas_history, 0, 1)
                        if i == 0:
                            meas_history = swarm.vehicles[i].meas_history[:,-1:]
                        else:
                            meas_history = np.vstack((meas_history, swarm.vehicles[i].meas_history[:,-1:]))
                    
                    gf = gtsam.CustomFactor(process_noise, [X[k], X[(k + 1)]],
                                            partial(error_dyn, meas_history))
                    
                    # gf = gtsam.CustomFactor(input_noise, [X[k], X[(k + 1)]],
                    #                         partial(error_dyn1, meas_history))
                    
                    graph.add(gf)
        
                    # Initial values for optimizer
                    for j in range(nb_agents):
                        X_val[j * nx:(j + 1) * nx, :] = unicycle.discrete_step(X_val[j * nx:(j + 1) * nx, :],
                                                                            meas_history[j * nu:(j + 1) * nu, :].reshape(
                                                                                nu, 1), Delta_t)
                    v.insert(X[k+1], X_val)
        
            # Range measurment factor
            range_period = 1./f_range
            
            if D(str(t[k])) % D(str(range_period)) == 0.:
                
                if t[k] > 0:
                    # print("RANGE MEASURE FACTOR")
                    range_meas = np.zeros((nb_agents, 1))
                    for j in range(nb_agents):
                        idx_set = np.nonzero(adjacency[j, :])[0]
                        range_meas = swarm.vehicles[j].measRange_history[idx_set, k]
                        range_noise = gtsam.noiseModel.Gaussian.Covariance(np.diag([std_range ** 2] * len(idx_set)))

                        gfrange = gtsam.CustomFactor(range_noise, [X[k]],
                                                    partial(error_range, j, idx_set, range_meas))
                        graph.add(gfrange)
                        if swarm.nb_landmarks > 0 and j in swarm.landmark_agents:
                            landmark_noise = gtsam.noiseModel.Gaussian.Covariance(np.diag([std_range ** 2] * swarm.nb_landmarks))
                            landmark_meas = swarm.vehicles[j].lm_meas_history[:, k:k+1]
                            lmrange = gtsam.CustomFactor(landmark_noise, [X[k]],
                                                            partial(error_range_lm, j, landmarks, landmark_meas))
                            graph.add(lmrange)
            if k == 0:
                s = opt_agent
                # swarm.MPC_horizon = int(T)
                # if myMetric == 'basic':
                    # swarm.w_plan = np.zeros(swarm.MPC_horizon)
                    # swarm.vehicles[int(s)].omega = swarm.w_plan[0]
                # else:
                    # swarm.Trajoptim(optim_agent=s, METRIC=myMetric)
                    # swarm.vehicles[int(s)].omega = swarm.w_plan[0]
                # print("made it")

                # prev_plan - swarm.w_plan
            # print("Didn't Make it here")

            # incremental smoothing
            if k > (tinc/Delta_t)-1 and count > (tinc/Delta_t)-1:
                # swarm.vehicles[int(s)].omega = swarm.w_plan[int(k/(tinc / Delta_t))-1]
                for n in range(nb_agents):
                    if n != s:
                        swarm.vehicles[n].omega = 0.
                count = 0
                # print("INCREMENTAL SMOOTHING")
                if not initialized:
                    #Optimize the first batch
                    params = gtsam.LevenbergMarquardtParams()
                    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, v, params)
                    v = optimizer.optimize()
                    initialized = True
        
                # ISAM2 update
                isam.update(graph, v)
                result = isam.calculateEstimate()
                cov = isam.marginalCovariance(X[k])           

                estimated_states_history = parse_result(result, cov, nb_agents, t[:k+1])
                s = np.random.randint(0,swarm.nb_agents)

                # This is where changes need to be made to change 
                # A) if there is an optim agent and 
                # B) what metric is used.
                swarm.MPC(optim_agent = id_var, use_cov = True, METRIC = 'min_eig_inv_cov')

                # for j in range(nb_agents):
                #     if k == (tinc/Delta_t):
                #         estimated_states_history.append(estimated_states[j])
                
                #     else:
                #         estimated_states_history[j] = np.hstack((estimated_states_history[j], estimated_states[j]))
                # k_old = k+1

                # Reset graph and values
                graph = gtsam.NonlinearFactorGraph()
                v = gtsam.Values()

                #Skip reintialization for the final batch
                if k < len(t)-1:
                    X0 = result.atVector(X[k+1]).reshape(nx*nb_agents,1)
                    X_val = X0
                    count = 0
            else:
                count += 1

            # params = gtsam.LevenbergMarquardtParams()
            # optimizer = gtsam.LevenbergMarquardtOptimizer(graph, v, params)
            # result = optimizer.optimize()

            # marginals = gtsam.Marginals(graph, result)
            # if isam.valueExists(X[k]):
            #     print(f"Variable X[{k}] found in the BayesTree.")
            #     cov = isam.marginalCovariance(X[k])
            #     estimated_states_history = parse_result(result, cov, nb_agents, t[:k+1])
            # else:
            #     print(f"Variable X[{k}] not found in the BayesTree.")
            # cov = marginals.marginalCovariance(X[k])
            # estimated_states_history = parse_result(result, cov, nb_agents, t[:k+1])

            radius_of_wheel, dist_between_wheel = 0.015, 0.08
            # print(id_var, swarm.vehicles[id_var].omega)
            # v_L = ((2 * vel) - (swarm.vehicles[id_var].omega * dist_between_wheel)) / (2 * radius_of_wheel)
            # v_R = ((2 * vel) + (swarm.vehicles[id_var].omega * dist_between_wheel)) / (2 * radius_of_wheel)        
            v_L = vel - ((swarm.vehicles[id_var].omega * dist_between_wheel) / 2)
            v_R = vel + ((swarm.vehicles[id_var].omega * dist_between_wheel) / 2)
            # print(f"Robot {id_var} diff_vel ({v_L}, {v_R}) ")
            
            robot.set_vel(v_L, v_R)
            print("Robot ", id_var, " vel: ", v_L, v_R, "Omega: ", swarm.vehicles[id_var].omega)
            k += 1

        print("FINISHED THE LOOP!")
        k -= 1
        swarm.get_swarm_states_history_()
        #swarm.plot_swarm_traj()
        get_swarm_states_history = swarm.get_swarm_states_history
        
        swarm.get_swarm_est_err_()
        EST_ERR = swarm.EST_ERR
        
        #print(graph)
        #print('Done.')
        #print('Performing factor graph optimization........')
        
        isam.update(graph, v)
        result = isam.calculateEstimate()
        cov = isam.marginalCovariance(X[k])            
        estimated_states_history = parse_result(result, cov, nb_agents, t[:k+1])
        
        # for j in range(nb_agents):
        #     if k == (tinc/Delta_t):
        #         estimated_states_history.append(estimated_states[j])
        
        #     else:
        #         estimated_states_history[j] = np.hstack((estimated_states_history[j], estimated_states[j]))
        k_old = k
        print('Done.')
        print('Reshaping results for plotting........')
        x_res = []
        # for j in range(nb_agents):
        #     x_sol = np.zeros((len(t), nx))
        #     for k in range(len(t)):
        #         x_sol[k, :] = result.atVector(X[k])[j*nx:(j+1)*nx]
        
        #     np.savetxt('x'+str(j)+'_'+str(id_var)+'.csv', x_sol, delimiter=',')
        #     x_res.append(x_sol)
        
        print('Done')
        
        TRUTH.append(get_swarm_states_history)
        EST.append(estimated_states_history)
        ERR.append(EST_ERR)
        
        # states = estimated_states_history[id_var]#x_res[j].transpose()
        # states_ = swarm.get_swarm_states_history[id_var]
        # est_dat = (states[0,:], states[1,:])
        # true_dat = (states_[0, 1:], states_[1, 1:])
        # np.savetxt('obsv_x'+str(id_var)+'_est.csv', est_dat, delimiter=',')
        # np.savetxt('obsv_x'+str(id_var)+'_true.csv', true_dat, delimiter=',')

        if id_var == 0:
            for j in range(nb_agents):
                savemat('estimate_history.mat', {'estimate_history': estimated_states_history})

                states = estimated_states_history[j]#x_res[j].transpose()
                states_ = swarm.get_swarm_states_history[j]
                time = t
                plt.plot(states[0, :], states[1, :], label='estimated Vehicle ' + str(j))
                plt.plot(states_[0, 1:], states_[1, 1:], label='true Vehicle ' + str(j))
                plt.legend()
                plt.title('Vehicle trajectories')
                plt.xlabel('x (m)')
                plt.ylabel('y (m)')

            plt.show()
        
        # time = t
        # legends = ['x', 'y', '$\\theta$']
        # for l in range(3):
        #     for j in range(nb_agents):
        #         states = estimated_states_history[j]#x_res[j].transpose()
        #         states_ = get_swarm_states_history[j]
        #         plt.plot(states[l, :] - states_[l, :], label='Vehicle '+str(j))
        #     plt.legend()
        #     plt.title(legends[l]+'-error trajectories' )
        #     plt.xlabel('timesteps')
        #     plt.ylabel(legends[l])
        #     plt.show()