'''
Incremental smoothing and mapping (ISAM2) implementation for multiagent estimation discussed in
Rutkowski, Adam J., Jamie E. Barnes, and Andrew T. Smith. "Path planning for optimal cooperative navigation." 2016 IEEE/ION Position, Location and Navigation Symposium (PLANS). IEEE, 2016.

Authors: Shahbaz P Qadri Syed, He Bai
'''

import numpy as np
import scipy as sc
from agent import Agent
from Swarm import Swarm
import gtsam
from typing import Optional, List
from functools import partial
import matplotlib.pyplot as plt
from decimal import Decimal as D
import random
random.seed(10)

print('Initializing agents.........')
agent = Agent()
unicycle = agent.unicycle
nx = 3
nu = 2
Delta_t   = 0.1
T = 150 #s
# finding the number of decimal places of Delta_t
precision = abs(D(str(Delta_t)).as_tuple().exponent)
t         = np.arange(0,T,Delta_t)
t = np.round(t, precision) # to round off python floating point precision errors
tinc = 0.5 #sec
vel         = 30 #m/s
omega_max = 5 #degrees/s
std_omega = np.deg2rad(0.57) #degrees/s
std_v     = 0.01 #m/s
std_range = 0.01 #m
f_range   = 10 #Hz
f_odom    = 10 #Hz
f_waypt  = 1
estimated_states_history = []
k_old = 0

nb_agents = 5
adjacency = np.array([[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1],[1,0,0,0,0]])#(np.ones((nb_agents, nb_agents)) - np.eye(nb_agents))
# Set initial and end position
# pos0 = np.array([[0.,10.,10.,0.],[0.,0.,10.,10.]])
# posf = np.array([[30., 40., 40., 30.],[30., 30., 40., 40.]])
pos0 = 10*np.array([[0.,10.,10.,0.,20.],[0.,0.,10.,10.,10.]])
posf = pos0 + np.ones((2,nb_agents)) * vel * T#100*np.array([[30., 40., 40., 30.,50.],[30., 30., 40., 40., 40.]])
print('Done.')

swarm =  Swarm()
swarm.update_adjacency(adjacency)
for i in range(nb_agents):
    swarm.add_vehicle(Delta_t, t, vel, std_omega, std_v, std_range, f_range, f_odom)
swarm.set_swarm_initPos(pos0)
swarm.set_swarm_endpos(posf)
print('Done.')

print('Propagating true state and generating measurements........')
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

print('Initializing factor graph...........')
S0 = 1e-4*np.eye(nx)
prior_noise = gtsam.noiseModel.Gaussian.Covariance(S0)
dynamics_noise = gtsam.noiseModel.Constrained.Sigmas(np.array([std_v*Delta_t, 0., std_omega*Delta_t]).reshape(nx,1))

# Create an empty Gaussian factor graph
graph = gtsam.NonlinearFactorGraph()

def parse_result(result, nb_agents, t):
    x_res = []
    for j in range(nb_agents):
        x_sol = np.zeros((len(t), nx))
        for k in range(len(t)):
            x_sol[k, :] = result.atVector(X[k * nb_agents + j])
        swarm.vehicles[j].states_est = x_sol.T
        x_res.append(x_sol.T)

    return x_res

# Dynamics factor
def error_dyn( measurements, this: gtsam.CustomFactor,
              values: gtsam.Values,
              jacobians: Optional[List[np.ndarray]]):

    key1 = this.keys()[0]
    key2 = this.keys()[1]

    X_, Xp1 = values.atVector(key1), values.atVector(key2)
    x = np.zeros((nx,1))
    u = measurements.reshape(nu,1)

    # for j in range(nb_agents):
    x= unicycle.discrete_step(X_.reshape(nx,1), u, Delta_t)


    error = Xp1 - x.reshape(nx,)

    if jacobians is not None:
        jacobians[1] = np.eye(nx)
        # for j in range(nb_agents):
        #     if j == 0:
        #         jac = unicycle.dyn_jacobian(X_[j*nx:(j+1)*nx].reshape(nx,1), u[j*nu:(j+1)*nu,:], Delta_t)
            # else:
            #     jac = sc.linalg.block_diag(jac, unicycle.dyn_jacobian(X_[j*nx:(j+1)*nx].reshape(nx,1), u[j*nu:(j+1)*nu,:], Delta_t))

        jacobians[0] = -unicycle.dyn_jacobian(X_.reshape(nx,1), u, Delta_t)

    return error

# Range factor
def error_range( measurement, this: gtsam.CustomFactor,
              values: gtsam.Values,
              jacobians: Optional[List[np.ndarray]]):

    key1 = this.keys()[0]
    key2 = this.keys()[1]

    X_ego, X_neighbor = values.atVector(key1), values.atVector(key2)

    vehicle_pos = X_ego[:2].reshape(2, 1)
    neighbor_pos = X_neighbor[:2].reshape(2, 1)
    range_ = np.linalg.norm(vehicle_pos - neighbor_pos)


    if jacobians is not None:
        if range_ != 0:
            jacobians[0] = (1. / range_) * np.hstack(((-neighbor_pos + vehicle_pos).transpose(), np.array([[0]])))
            jacobians[1] = (1. / range_) * np.hstack((-(-neighbor_pos + vehicle_pos).transpose(), np.array([[0]])))
        else:
            jacobians[0] = np.zeros((1, nx))
            jacobians[1] = np.zeros((1, nx))


    error = (range_ - measurement.reshape(1,)).reshape(1,)

    return error
# def dynamic_noise(X):
#     cov =0
#     return gtsam.noiseModel.Constrained.Covariance(cov)

# Create the keys corresponding to unknown variables in the factor graph
X = []
# for k in range(len(t)):
#     X.append(gtsam.symbol('x', k))
for k in range(len(t)):
    for j in range(nb_agents):
        X.append(gtsam.symbol('x', k*nb_agents+j))
v = gtsam.Values()

# set initial state as prior
X0 = np.zeros((nx, nb_agents))
theta0 = [0, 0, 0, 0, 0]
X_val = []
# set initial state as prior
for j in range(nb_agents):
    X0 = np.vstack((pos0[:,j:j+1],np.array([[0.]])))
    X_val.append(X0)
    graph.add(gtsam.PriorFactorVector(X[j], X0, prior_noise))
    v.insert(X[j], X0)

# X0 = X0.T.flatten().reshape(nx*nb_agents,1)
# graph.add(gtsam.PriorFactorVector(X[0], X0, prior_noise))
# v.insert(X[0], X0)
count = 0
# X_val = X0
idx = 0
isam = gtsam.ISAM2()
initialized = False
k = 0
for k in range(0, len(t)):
    # print(k)
    tt = t[k]#k * Delta_t
    swarm.update_measRange()
    swarm.update_state(tt)
    for j in range(nb_agents):
        if k < len(t) - 1:
            # Dynamics factor
            odom_period = 1. / f_odom
            if D(str(t[k])) % D(str(odom_period)) == 0.:
                idx = D(str(t[k])) // D(str(odom_period))
                idx_bias = D(str(t[0])) // D(str(odom_period))
                # for i in range(nb_agents):

                swarm.vehicles[j].meas_history = np.delete(swarm.vehicles[j].meas_history, 0, 1)
                if k == 0:
                    swarm.vehicles[j].measRange_history = np.delete(swarm.vehicles[j].measRange_history, 0, 1)
                # if j == 0:
                meas_history = swarm.vehicles[j].meas_history[:,-1:]
                # else:
                #     meas_history = np.vstack((meas_history, swarm.vehicles[i].meas_history[:,-1:]))
                # dynamics_noise = gtsam.noiseModel.Constrained.Covariance(
                #     get_cov(X_val))
                gf = gtsam.CustomFactor(dynamics_noise, [X[k * nb_agents + j], X[(k + 1) * nb_agents + j]],
                                        partial(error_dyn, meas_history))
                # gf = gtsam.CustomFactor(partial(dynamics_noise, v.atVector(X[k])), [X[k], X[(k + 1)]],
                #                                                 partial(error_dyn, meas_history))
                graph.add(gf)

                # Initial values for optimizer
                # for j in range(nb_agents):
                X_val[j] = unicycle.discrete_step(X_val[j], meas_history.reshape(nu, 1), Delta_t)
                v.insert(X[(k + 1) * nb_agents + j], X_val[j])

        # Range measurement factor
        # if t[k] > 0:
        range_period = 1./f_range
        if D(str(t[k])) % D(str(range_period)) == 0.:
            # range_meas = np.zeros((nb_agents, 1))
            # for j in range(nb_agents):
            X_idx = [X[k * nb_agents + j]]
            idx_set = np.nonzero(adjacency[j, :])[0]
            range_meas = swarm.vehicles[j].measRange_history[idx_set, k]
            range_noise = gtsam.noiseModel.Gaussian.Covariance(np.diag([std_range ** 2]))
            idx_count = 0
            for idx_n, a in enumerate(idx_set):
                # X_idx.extend(X[k * nb_agents + a])
                gfrange = gtsam.CustomFactor(range_noise, [X[k * nb_agents + j], X[k * nb_agents + a]], partial(error_range, range_meas[idx_n]))
                graph.add(gfrange)

    # incremental smoothing
    if k > (tinc/Delta_t)-1 and count > (tinc/Delta_t)-1:
        if not initialized:
            #Optimize the first batch
            params = gtsam.LevenbergMarquardtParams()
            optimizer = gtsam.LevenbergMarquardtOptimizer(graph, v, params)
            v = optimizer.optimize()
            initialized = True

        # ISAM2 update
        isam.update(graph, v)
        # print(graph)
        result = isam.calculateEstimate()
        for j in range(nb_agents):
            cov = isam.marginalCovariance(X[k * nb_agents + j])
            # marginals = gtsam.Marginals(graph, result)
            # cov = marginals.marginalCovariance(X[k])
            print(cov)
        estimated_states_history = parse_result(result, nb_agents, t[:k+1])
        # for j in range(nb_agents):
        #     if k == (tinc/Delta_t):
        #         estimated_states_history.append(estimated_states[j])
        #
        #     else:
        #         estimated_states_history[j] = np.hstack((estimated_states_history[j], estimated_states[j]))
        # k_old = k+1
        # Reset graph and values
        graph = gtsam.NonlinearFactorGraph()
        v = gtsam.Values()

        #Skip reintialization for the final batch
        if k < len(t)-1:
            for j in range(nb_agents):
                X0 = result.atVector(X[(k+1) * nb_agents + j]).reshape(nx,1)
                X_val[j] = X0
            count = 0
    else:
        count += 1


swarm.get_swarm_states_history_()
swarm.plot_swarm_traj()
get_swarm_states_history = swarm.get_swarm_states_history

print(graph)
print('Done.')
print('Performing factor graph optimization........')

# if graph.size() > 0:
isam.update(graph, v)
result = isam.calculateEstimate()
estimated_states_history = parse_result(result, nb_agents, t[:k+1])
# for j in range(nb_agents):
#     if k == (tinc/Delta_t):
#         estimated_states_history.append(estimated_states[j])
#
#     else:
#         estimated_states_history[j] = np.hstack((estimated_states_history[j], estimated_states[j]))
# k_old = k
print('Done.')
print('Reshaping results for plotting........')
# x_res = []
# for j in range(nb_agents):
#     x_sol = np.zeros((len(t), nx))
#     for k in range(len(t)):
#         x_sol[k, :] = result.atVector(X[k])[j*nx:(j+1)*nx]
#
#     np.savetxt('x'+str(j)+'.csv', x_sol, delimiter=',')
#     x_res.append(x_sol)

print('Done')

for j in range(nb_agents):
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

time = t
legends = ['x', 'y', '$\\theta$']
for l in range(3):
    for j in range(nb_agents):
        states = estimated_states_history[j]#x_res[j].transpose()
        states_ = get_swarm_states_history[j]
        plt.plot(states[l, :] - states_[l, :], label='Vehicle '+str(j))
    plt.legend()
    plt.title(legends[l]+'-error trajectories' )
    plt.xlabel('timesteps')
    plt.ylabel(legends[l])
    plt.show()