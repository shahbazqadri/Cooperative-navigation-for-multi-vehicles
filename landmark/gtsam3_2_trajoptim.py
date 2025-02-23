'''
Incremental smoothing and mapping (ISAM2) implementation for multiagent estimation discussed in
Rutkowski, Adam J., Jamie E. Barnes, and Andrew T. Smith. "Path planning for optimal cooperative navigation." 2016 IEEE/ION Position, Location and Navigation Symposium (PLANS). IEEE, 2016.

Authors: Shahbaz P Qadri Syed, He Bai
'''

import numpy as np
import scipy as sc
from scipy.signal.windows import lanczos

from agent import Agent
from Swarm_PSO import Swarm
import gtsam
from gtsam.symbol_shorthand import L
from typing import Optional, List
from functools import partial
import matplotlib.pyplot as plt
from decimal import Decimal as D
import random

MC_TOTAL = 20
myMetric = 'Trace_inv_SAM'
adj_type_set = ['full']
nb_agents = 4
nb_landmarks = 0
for adj_type in adj_type_set:
    for opt_agent in range(nb_agents):
        #opt_agent = 2

        my_seed = 100
        random.seed(my_seed)
        np.random.seed(my_seed)


        TRUTH = []
        EST = []
        ERR = []
        COV = []
        myM = []
        landmarks = [np.zeros((nb_landmarks, nb_landmarks)) if nb_landmarks > 0 else []]
        for MC_RUN in range(MC_TOTAL):
            print(MC_RUN)
            cov_mc = []
            # print('Initializing agents.........')
            agent = Agent()
            unicycle = agent.unicycle
            nx = 3
            nu = 2
            Delta_t = 0.1
            T = 10  # s
            # finding the number of decimal places of Delta_t
            precision = abs(D(str(Delta_t)).as_tuple().exponent)
            t = np.arange(0, T, Delta_t)
            t = np.round(t, precision)  # to round off python floating point precision errors
            tinc = 1.0  # 0.5 #sec
            vel = 30  # m/s
            # omega_max = 10 #degrees/s
            std_omega = 0 * np.deg2rad(0.57)  # degrees/s
            std_v = 0 * 0.01  # * 10 #m/s
            std_range = 0.01 * 100  # m
            S_Q = np.diag([0.3, 0.3, 0.03])/3 * Delta_t

            # TODO: frequency of range and odometry measurements from simulator
            f_range = 1  # Hz
            f_odom = 10  # Hz
            # #

            f_waypt = 1
            estimated_states_history = []
            k_old = 0

            # TODO: number of agents in the simulation and the adjacency graph of range measurement

            # adjacency = np.array([[0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 0, 0]]) # cross
            # adjacency = np.array([[0, 1, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 1, 0], [1, 0, 0, 0, 0], [0, 0, 1, 0, 0]])#ring2
            # adjacency = np.array([[0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 1, 0, 0, 0], [1, 0, 1, 0, 0], [0, 0, 1, 0, 0]]) #sparseloop
            #adjacency = np.array([[0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [1, 0, 0, 0, 0]])  # ring
            #adjacency = np.array([[0, 1], [1, 0]])
            #adjacency = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
            if adj_type == 'full':
                adjacency = np.ones((nb_agents, nb_agents)) - np.eye(nb_agents)
            elif adj_type == 'd_ring':
                adjacency = np.array([[0 if i != (j+1) % nb_agents else 1 for i in range(nb_agents)] for j in range(nb_agents)])
                adjacency = np.array([[0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1], [1, 0,  0, 0]])
            # adjacency = (np.ones((nb_agents, nb_agents)) - np.eye(nb_agents))#complete
            # #
            # adjacency = np.array([[0,1,1,0,0],[0,0,1,0,0],[0,0,0,1,0],[1,0,0,0,1],[1,0,0,0,0]])#

            # adjacency = (np.ones((nb_agents, nb_agents)) - np.eye(nb_agents))#np.array([[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1],[1,0,0,0,0]])#(np.ones((nb_agents, nb_agents)) - np.eye(nb_agents))
            # Set initial and end position
            # pos0 = np.array([[0.,10.,10.,0.],[0.,0.,10.,10.]])
            # posf = np.array([[30., 40., 40., 30.],[30., 30., 40., 40.]])

            # TODO: Initial and final positions for simulator
            pos0 = 10 * np.array([[10., 0., 10., 20., 10.], [10., 0., -10., 0., 0.]])
            pos0 = pos0[:nb_agents, :nb_agents]
            posf = pos0 + np.ones(
                (2, nb_agents)) * vel * T * 0.7  # 100*np.array([[30., 40., 40., 30.,50.],[30., 30., 40., 40., 40.]])
            #  #
            # print('Done.')
            if nb_landmarks > 0:
                landmarks = np.array([[200., 500., 0.], [200., -500.0, 0.]])
                landmarks = landmarks[:,0:nb_landmarks]

            swarm = Swarm()
            if nb_landmarks > 0:
                swarm.add_landmarks(landmarks)
            swarm.update_adjacency(adjacency)
            for i in range(nb_agents):
                swarm.add_vehicle(Delta_t, t, vel, std_omega, std_v, std_range, f_range, f_odom, S_Q)
            swarm.set_swarm_initPos(pos0)
            swarm.set_swarm_endpos(posf)
            if nb_landmarks > 0:
                swarm.agent_to_landmarks([opt_agent])

            # print('Done.')

            # print('Propagating true state and generating measurements........')
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

            # print('Initializing factor graph...........')
            S0 = 1e-4 * np.eye(nx * nb_agents)
            prior_noise = gtsam.noiseModel.Gaussian.Covariance(S0)
            dynamics_noise = gtsam.noiseModel.Constrained.Sigmas(
                np.array([std_v * Delta_t, 0., std_omega * Delta_t] * nb_agents).reshape(nx * nb_agents,
                                                                                         1))  # in the body frame of each agent
            cov = np.kron(np.eye(nb_agents), np.diag([std_v ** 2, std_omega ** 2]))
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
                    swarm.vehicles[j].states_est = x_sol[-1, :].reshape((3, 1))
                    swarm.vehicles[j].states_cov = cov[j * nx:(j + 1) * nx, j * nx:(j + 1) * nx]
                    swarm.vehicles[j].est_err = np.hstack(
                        (swarm.vehicles[j].est_err, swarm.vehicles[j].states_est - swarm.vehicles[j].states))
                    # print(swarm.vehicles[j].states_est.shape)
                    x_res.append(x_sol.T)
                return x_res


            def error_dyn1(measurements, this: gtsam.CustomFactor,
                           values: gtsam.Values,
                           jacobians: Optional[List[np.ndarray]]):

                key1 = this.keys()[0]
                key2 = this.keys()[1]

                X_, Xp1 = values.atVector(key1), values.atVector(key2)
                u = measurements.reshape(nu * nb_agents, 1)
                u_cal = np.zeros((nu * nb_agents, 1))
                for j in range(nb_agents):
                    u_cal[j * nu:(j + 1) * nu, :] = unicycle.find_u(X_[j * nx:(j + 1) * nx].reshape(nx, 1),
                                                                    Xp1[j * nx:(j + 1) * nx].reshape(nx, 1), Delta_t)

                if jacobians is not None:
                    for j in range(nb_agents):
                        jac0, jac1 = unicycle.find_u_jacobian(X_[j * nx:(j + 1) * nx].reshape(nx, 1),
                                                              Xp1[j * nx:(j + 1) * nx].reshape(nx, 1), Delta_t)
                        if j == 0:
                            jacobians[0] = jac0
                            jacobians[1] = jac1
                        else:
                            jacobians[0] = sc.linalg.block_diag(jacobians[0], jac0)
                            jacobians[1] = sc.linalg.block_diag(jacobians[1], jac1)

                error = (u_cal - u).reshape(nu * nb_agents, )

                return error


            # Dynamics factor
            def error_dyn(measurements, this: gtsam.CustomFactor,
                          values: gtsam.Values,
                          jacobians: Optional[List[np.ndarray]]):

                key1 = this.keys()[0]
                key2 = this.keys()[1]

                X_, Xp1 = values.atVector(key1), values.atVector(key2)
                # m_x = np.zeros((nx*nb_agents,1))
                x = np.zeros((nx * nb_agents, 1))
                u = measurements.reshape(nu * nb_agents, 1)

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
                    x[j * nx:(j + 1) * nx, :] = unicycle.discrete_step(X_[j * nx:(j + 1) * nx].reshape(nx, 1),
                                                                       u[j * nu:(j + 1) * nu, :], Delta_t)
                    # th = x[j*nx+2,0] # current heading
                    # rot = np.array([[np.cos(th), -np.sin(th), 0.], [np.sin(th), np.cos(th), 0.], [0.,0.,1.]])
                    # drot = np.array([[-np.sin(th), -np.cos(th), 0.], [np.cos(th), -np.sin(th), 0.], [0.,0.,0.]])
                    # if j == 0:
                    #     Rot = rot
                    #     dRot = drot
                    # else:
                    #     Rot = sc.linalg.block_diag(Rot, rot)
                    #     dRot = sc.linalg.block_diag(dRot, drot)

                error = (Xp1 - x.reshape(nx * nb_agents, ))
                if jacobians is not None:
                    jacobians[1] = np.eye(nx * nb_agents)
                    for j in range(nb_agents):
                        if j == 0:
                            jac = unicycle.dyn_jacobian(X_[j * nx:(j + 1) * nx].reshape(nx, 1), u[j * nu:(j + 1) * nu, :],
                                                        Delta_t)
                        else:
                            jac = sc.linalg.block_diag(jac, unicycle.dyn_jacobian(X_[j * nx:(j + 1) * nx].reshape(nx, 1),
                                                                                  u[j * nu:(j + 1) * nu, :], Delta_t))
                    jacobians[0] = - jac

                return error

            # landmark range factor
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

                range_est = np.zeros((n, 1))
                vehicle_pos = X_[ego_idx * nx:((ego_idx + 1) * nx) - 1].reshape(2, 1)

                for j in range(n):
                    jac = np.zeros((1, nx * nb_agents))
                    neighbor_idx = neighbor_idx_set[j]
                    neighbor_pos = X_[neighbor_idx * nx:(neighbor_idx + 1) * nx - 1].reshape(2, 1)
                    range_ = np.linalg.norm(vehicle_pos - neighbor_pos)
                    range_est[j, :] = range_
                    jac[:, ego_idx * nx:((ego_idx + 1) * nx) - 1] = -(neighbor_pos - vehicle_pos).transpose()
                    jac[:, neighbor_idx * nx:((neighbor_idx + 1) * nx) - 1] = -(-neighbor_pos + vehicle_pos).transpose()

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


            # Create the keys corresponding to unknown variables in the factor graph
            X = []
            for k in range(len(t)):
                X.append(gtsam.symbol('x', k))
            v = gtsam.Values()

            # set initial state as prior
            X0 = np.zeros((nx, nb_agents))
            # TODO: initial orientation of the agents (at pos0)
            theta0 = [0, 0, 0, 0, 0]

            for j in range(nb_agents):
                X0[:, j:j + 1] = np.vstack((pos0[:, j:j + 1], np.array([[theta0[j]]])))

            swarm.set_swarm_estimates(X0, S0)

            X0 = X0.T.flatten().reshape(nx * nb_agents, 1)
            graph.add(gtsam.PriorFactorVector(X[0], X0, prior_noise))
            v.insert(X[0], X0)
            count = 0
            X_val = X0
            idx = 0
            # isam = gtsam.ISAM2()
            initialized = False
            k = 0

            for k in range(0, len(t)):
                # ALL the drones
                tt = t[k]  # k * Delta_t
                swarm.update_measRange()  # TODO: See function
                swarm.update_state(tt)  # TODO: See function
                if k < len(t) - 1:
                    # Dynamics factor
                    odom_period = 1. / f_odom
                    if D(str(t[k])) % D(str(odom_period)) == 0.:
                        idx = D(str(t[k])) // D(str(odom_period))
                        idx_bias = D(str(t[0])) // D(str(odom_period))
                        for i in range(nb_agents):
                            # one drone at a time
                            swarm.vehicles[i].meas_history = np.delete(swarm.vehicles[i].meas_history, 0, 1)
                            if k == 0:
                                swarm.vehicles[i].measRange_history = np.delete(swarm.vehicles[i].measRange_history, 0, 1)
                                if nb_landmarks > 0:
                                    swarm.vehicles[i].lm_meas_history = np.delete(swarm.vehicles[i].lm_meas_history, 0, 1)
                            if i == 0:
                                meas_history = swarm.vehicles[i].meas_history[:, -1:]
                            else:
                                meas_history = np.vstack((meas_history, swarm.vehicles[i].meas_history[:, -1:]))

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
                        v.insert(X[k + 1], X_val)

                # Range measurment factor
                range_period = 1. / f_range
                if D(str(t[k])) % D(str(range_period)) == 0.:
                    #range_meas = np.zeros((nb_agents, 1))
                    if t[k] > 0:
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
                # incremental smoothing
                # ####################### works when planning starts at 1 second
                # if k == (tinc / Delta_t) - 1:
                #     params = gtsam.LevenbergMarquardtParams()
                #     optimizer = gtsam.LevenbergMarquardtOptimizer(graph, v, params)
                #     result = optimizer.optimize()
                #
                #     marginals = gtsam.Marginals(graph, result)
                #     cov = marginals.marginalCovariance(X[k])
                #     s = np.random.randint(0, swarm.nb_agents)
                #     s = opt_agent
                #     estimated_states_history = parse_result(result, cov, nb_agents, t[:k + 1])
                #     if MC_RUN == 0:
                #         swarm.Trajoptim(optim_agent=s, METRIC=myMetric)
                #         swarm.vehicles[int(s)].omega = swarm.w_plan[int(k/(tinc / Delta_t))-1]
                #         prev_plan = swarm.w_plan
                #     else:
                #         swarm.w_plan = prev_plan
                #         swarm.vehicles[int(s)].omega = swarm.w_plan[int(k / (tinc / Delta_t)) - 1]
                # if k > (tinc / Delta_t) - 1 and count > (tinc / Delta_t) - 1:
                #     swarm.vehicles[int(s)].omega = swarm.w_plan[int(k/(tinc / Delta_t))-1]
                #     for n in range(nb_agents):
                #         if n != s:
                #             swarm.vehicles[n].omega = 0.
                #     count = 0
                # else:
                #     count += 1
                ##################
                if k == 0:
                    s = opt_agent
                    swarm.MPC_horizon = int(T)
                    if MC_RUN == 0:
                        if myMetric == 'basic':
                            swarm.w_plan = np.zeros(swarm.MPC_horizon)
                            swarm.vehicles[int(s)].omega = swarm.w_plan[0]
                        else:
                            swarm.Trajoptim(optim_agent=s, METRIC=myMetric)
                            swarm.vehicles[int(s)].omega = swarm.w_plan[0]
                        prev_plan = swarm.w_plan
                    else:
                        swarm.w_plan = prev_plan
                        swarm.vehicles[int(s)].omega = swarm.w_plan[0]
                if k >= (tinc / Delta_t) - 1 and count >= (tinc / Delta_t) - 1:
                    swarm.vehicles[int(s)].omega = swarm.w_plan[int(k / (tinc / Delta_t)) - 1]
                    for n in range(nb_agents):
                        if n != s:
                            swarm.vehicles[n].update_controller()
                            #swarm.vehicles[n].omega = 0.
                    count = 0
                else:
                    count += 1
            params = gtsam.LevenbergMarquardtParams()
            #params.setMaxIterations(1000)
            optimizer = gtsam.LevenbergMarquardtOptimizer(graph, v, params)
            result = optimizer.optimize()
            # initialized = True

            # # ISAM2 update
            # isam.update(graph, v)
            # result = isam.calculateEstimate()
            marginals = gtsam.Marginals(graph, result)
            cov = marginals.marginalCovariance(X[k])
            estimated_states_history = parse_result(result, cov, nb_agents, t[:k + 1])

            # # Reset graph and values
            # graph = gtsam.NonlinearFactorGraph()
            # v = gtsam.Values()

            #     # Skip reintialization for the final batch
            #     if k < len(t) - 1:
            #         X0 = result.atVector(X[k + 1]).reshape(nx * nb_agents, 1)
            #         X_val = X0
            #         count = 0
            # else:
            #     count += 1

            swarm.get_swarm_states_history_()
            # swarm.plot_swarm_traj()
            get_swarm_states_history = swarm.get_swarm_states_history

            swarm.get_swarm_est_err_()
            EST_ERR = swarm.EST_ERR

            # isam.update(graph, v)
            # result = isam.calculateEstimate()
            # cov = isam.marginalCovariance(X[k])
            estimated_states_history = parse_result(result, cov, nb_agents, t[:k + 1])


            TRUTH.append(get_swarm_states_history)
            EST.append(estimated_states_history)
            ERR.append(EST_ERR)
            COV.append(cov)
            if MC_RUN == 0:
                myP = swarm.finalP

        data_dict = {"TRUTH": TRUTH, "EST": EST, "ERR": ERR, "ADJ": adjacency, "Metric": myMetric, "COV": COV, "P": myP, "LM": landmarks}
        filename = str(MC_TOTAL) + '-' + str(nb_agents) + '-' + str(T) + '-' + adj_type + '-' + myMetric + '-' + str(opt_agent) + '-' + str(nb_landmarks) + '-once_TrajOpt.mat'
        sc.io.savemat(filename, data_dict)
        # if myMetric == 'basic':
        #     for j in range(1, nb_agents):
        #         filename = str(MC_TOTAL) + '-' + str(nb_agents) + '-' + str(T) + '-' + adj_type + '-' + myMetric + '-' + str(j) + '-once_w20_LM.mat'
        #         sc.io.savemat(filename, data_dict)
        #     break
