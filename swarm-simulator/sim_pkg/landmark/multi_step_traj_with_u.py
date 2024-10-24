'''
Multi-step(interval) trajectory factor graph optimization (GTSAM) for multiagent estimation discussed in
Rutkowski, Adam J., Jamie E. Barnes, and Andrew T. Smith. "Path planning for optimal cooperative navigation." 2016 IEEE/ION Position, Location and Navigation Symposium (PLANS). IEEE, 2016.

Authors: Shahbaz P Qadri Syed, He Bai
Centralized state-control estimator: multi-step(interval) trajectory
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


# Generate measurements from true state
def generate_truth_and_measurements(swarm, nb_agents, pose0, posef, adjacency, t):
    swarm.set_swarm_initPose(pose0)
    swarm.set_swarm_endpose(posef)
    print('Done.')

    print('Propagating true state and generating measurments........')
    swarm.update_adjacency(adjacency)
    # propagate the swarm system
    for tt in t:
        # update vehicle states and plot
        swarm.update_state(tt)
        swarm.update_measRange()
    for i in range(nb_agents):
        swarm.vehicles[i].meas_history = np.delete(swarm.vehicles[i].meas_history, 0, 1)
        swarm.vehicles[i].measRange_history = np.delete(swarm.vehicles[i].measRange_history, 0, 1)
        if i == 0:
            meas_history = swarm.vehicles[i].meas_history
        else:
            meas_history = np.vstack((meas_history, swarm.vehicles[i].meas_history))
    swarm.get_swarm_states_history_()
    # swarm.plot_swarm_traj()
    get_swarm_states_history = swarm.get_swarm_states_history

    return  swarm, meas_history, get_swarm_states_history


# Dynamics custom factor
def error_dyn( measurements, this: gtsam.CustomFactor,
              values: gtsam.Values,
              jacobians: Optional[List[np.ndarray]]):
    key1 = this.keys()[0]
    key2 = this.keys()[1]
    key3 = this.keys()[2]

    X_, U_, Xp1 = values.atVector(key1), values.atVector(key2), values.atVector(key3)
    x = np.zeros((nx*nb_agents,1))
    # u = np.zeros((nu*nb_agents,1))
    for j in range(nb_agents):
        x[j*nx:(j+1)*nx, :] = unicycle.discrete_step(X_[j*nx:(j+1)*nx].reshape(nx,1), U_[j*nu:(j+1)*nu].reshape(nu,1), Delta_t)


    error = Xp1 - x.reshape(nx*nb_agents,)

    if jacobians is not None:
        jacobians[2] = np.eye(nx*nb_agents)
        for j in range(nb_agents):
            if j == 0:
                jac = unicycle.dyn_jacobian(X_[j*nx:(j+1)*nx].reshape(nx,1), U_[j*nu:(j+1)*nu].reshape(nu,1), Delta_t)
                jacu = unicycle.u_jacobian(X_[j*nx:(j+1)*nx].reshape(nx,1), U_[j*nu:(j+1)*nu].reshape(nu,1), Delta_t)
            else:
                jac = sc.linalg.block_diag(jac, unicycle.dyn_jacobian(X_[j*nx:(j+1)*nx].reshape(nx,1), U_[j*nu:(j+1)*nu].reshape(nu,1), Delta_t))
                jacu = sc.linalg.block_diag(jacu, unicycle.u_jacobian(X_[j * nx:(j + 1) * nx].reshape(nx, 1),
                                                                      U_[j * nu:(j + 1) * nu].reshape(nu, 1), Delta_t))
        jacobians[0] = -jac
        jacobians[1] = -jacu

    return error


# Range measurement custom factor
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

# odometry custom factor
def error_odom(measurement, this: gtsam.CustomFactor,
              values: gtsam.Values,
              jacobians: Optional[List[np.ndarray]]):
    key1 = this.keys()[0]

    u1 = values.atVector(key1)
    u1 = u1.reshape(nu*nb_agents, 1)

    error = (u1 - measurement.reshape(nu * nb_agents, 1)).reshape(nu * nb_agents, )
    if jacobians is not None:
        jacobians[0] = np.eye(nu*nb_agents)
    return error

# Single interval trajectory estimation
def estimate_step(t, nx, nu, nb_agents, X0, U0, prior_noise, swarm, meas_history, adjacency, std_omega, std_v, std_range, f_range, f_odom):
    # Noise model
    dynamics_noise = gtsam.noiseModel.Constrained.All(nx*nb_agents)
    odom_noise = gtsam.noiseModel.Gaussian.Covariance(np.diag([std_v ** 2, std_omega ** 2] * nb_agents))

    #Create an empty Gaussian factor graph
    graph = gtsam.NonlinearFactorGraph()

    # Create the keys corresponding to unknown variables in the factor graph
    X = []
    U = []
    for k in range(len(t)):
        X.append(gtsam.symbol('x', k))
        U.append(gtsam.symbol('u', k))

    v = gtsam.Values()

    # set initial state as prior
    graph.add(gtsam.PriorFactorVector(X[0], X0, prior_noise))
    v.insert(X[0], X0)
    v.insert(U[0], U0)
    X_val = X0

    for k in range(len(t)):
        if k < len(t) - 1:
            # dynamics factor
            gf = gtsam.CustomFactor(dynamics_noise, [X[k], U[k], X[(k + 1)]],
                                    partial(error_dyn, np.array([X[k], X[(k + 1)]])))
            graph.add(gf)

        # odometry factor
        odom_period = 1. / f_odom
        if D(str(t[k])) % D(str(odom_period)) == 0.:
            idx = D(str(t[k])) // D(str(odom_period))
            idx_bias = D(str(t[0])) // D(str(odom_period))
            gfodom = gtsam.CustomFactor(odom_noise, [U[k]], partial(error_odom, meas_history[:, int(idx - idx_bias)]))
            graph.add(gfodom)

        if k > 0:
            # Range factor
            range_period = 1./f_range
            if D(str(t[k])) % D(str(range_period)) == 0.:
                for j in range(nb_agents):
                    idx_set = np.nonzero(adjacency[j,:])[0]
                    range_meas = swarm.vehicles[j].measRange_history[idx_set, k]
                    range_noise = gtsam.noiseModel.Gaussian.Covariance(np.diag([std_range ** 2] * len(idx_set)))
                    gfrange = gtsam.CustomFactor(range_noise, [X[k]], partial(error_range, j, idx_set, range_meas))  # np.array([X[k]])
                    graph.add(gfrange)

            for j in range(nb_agents):
                X_val[j*nx:(j+1)*nx,:] = unicycle.discrete_step(X_val[j*nx:(j+1)*nx,:] , meas_history[j*nu:(j+1)*nu, k:k+1].reshape(nu,1), Delta_t)

            # Initial  values for optimizer
            v.insert(X[k], X_val)
            v.insert(U[k], meas_history[:, k:k+1])

    # print(graph)
    print('Done.')
    print('Performing factor graph optimization........')
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, v, params)
    result = optimizer.optimize()

    marginals = gtsam.Marginals(graph, result)
    x_res, u_res = [], []
    for j in range(nb_agents):
        x_sol = np.zeros((len(t), nx))
        u_sol = np.zeros((len(t), nu))
        for k in range(len(t)):
            x_sol[k, :] = result.atVector(X[k])[j * nx:(j + 1) * nx]
            u_sol[k, :] = result.atVector(U[k])[j * nu:(j + 1) * nu]
        x_res.append(x_sol.T)
        u_res.append(u_sol.T)
    return x_res, u_res, marginals.marginalCovariance(X[len(t)-1]), marginals.marginalCovariance(U[len(t)-1])

print('Initializing agents.........')

agent = Agent()
unicycle = agent.unicycle
nb_agents = 4 #Number of agents
Delta_t = 0.01 #stepsize for discretization
T = 100 # Total horizon of estimation in secs
num_sub_traj = 10 #Number of steps/sub-intervals
nx = 3 #state dimension
nu = 2 #control dimension
vel         = 30 #m/s
omega_max = 5 #degrees/s
std_omega = np.deg2rad(0.57) #degrees/s
std_v     = 0.01 #m/s
std_range = 0.01 #m
f_range   = 10 #Hz
f_odom    = 100 #Hz
f_waypt  = 1
adjacency = np.ones((nb_agents, nb_agents)) - np.eye(nb_agents) #Adjacency matrix for range measurements
# Set initial and end pose
pose0 = np.array([[0.,10.,10.,0.],[0.,0.,10.,10.],[0.,0.,0.,0.]])
pose_est0 = np.array([[0.,10.,10.,0.],[0.,0.,10.,10.],[0.,0.,0.,0.]])
posef = np.array([[30., 40., 40., 30.],[30., 30., 40., 40],[0.,0.,0.,0.]])
print('Done.')

S0 = 1e-4*np.eye(nx*nb_agents)
u0 = 1e-4*np.eye(nu*nb_agents)
prior_noise    = gtsam.noiseModel.Gaussian.Covariance(S0)
swarm_states_history = []
estimated_states_history = []
X0 = pose_est0.T.flatten().reshape(nx*nb_agents,1)
U0 = np.ones((nu*nb_agents,1))

for i in range(num_sub_traj):
    print(i)
    tmin = i * (T/num_sub_traj)
    tmax = (i+1) * (T/num_sub_traj)
    # finding the number of decimal places of Delta_t
    precision = abs(D(str(Delta_t)).as_tuple().exponent)
    t         = np.arange(tmin,tmax+Delta_t,Delta_t)
    t = np.round(t, precision) # to round off python floating point precision errors

    # Initialize swarm object
    swarm =  Swarm()
    for ii in range(nb_agents):
        swarm.add_vehicle(Delta_t, t, vel, std_omega, std_v, std_range, f_range, f_odom)

    # Generate measurements from true state
    swarm, meas_history, states_history = generate_truth_and_measurements(swarm, nb_agents, pose0, posef, adjacency, t)

    # Estimate trajectory for one sub-interval
    x_res, u_res, xf_cov, uf_cov =  estimate_step(t, nx, nu, nb_agents, X0, U0, prior_noise, swarm, meas_history, adjacency, std_omega, std_v, std_range, f_range, f_odom)

    for j in range(nb_agents):
        if i == 0:
            swarm_states_history.append(states_history[j])
            estimated_states_history.append(x_res[j])

        else:
            swarm_states_history[j] = np.hstack((swarm_states_history[j], states_history[j]))
            estimated_states_history[j] = np.hstack((estimated_states_history[j], x_res[j]))

        # Re-initialization for next sub-interval
        pose0[:, j:j + 1] = states_history[j][:, -1:]
        pose_est0[:, j:j + 1] = x_res[j][:, -1:]
        U0[j * nu:(j + 1) * nu, :] = u_res[j][:, -1:]

    X0 = pose_est0.T.flatten().reshape(nx * nb_agents, 1)
    prior_noise = gtsam.noiseModel.Gaussian.Covariance(xf_cov)

# Plotting true and estimated trajectories
for j in range(nb_agents):
    states = estimated_states_history[j]
    states_ = swarm_states_history[j]
    time = t
    plt.plot(states[0, :], states[1, :], label='estimated Vehicle ' + str(j))
    plt.plot(states_[0, :], states_[1, :], label='true Vehicle ' + str(j))
    plt.legend()
    plt.title('Vehicle trajectories')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.show()

# Plotting error trajectories
time = t
legends = ['x', 'y', '$\\theta$']
for l in range(3):
    for j in range(nb_agents):
        states = estimated_states_history[j]
        states_ = swarm_states_history[j]
        plt.plot(states[l, :] - states_[l, :], label='Vehicle '+str(j))
    plt.legend()
    plt.title(legends[l]+'-error trajectories' )
    plt.xlabel('timesteps')
    plt.ylabel(legends[l])
    plt.show()