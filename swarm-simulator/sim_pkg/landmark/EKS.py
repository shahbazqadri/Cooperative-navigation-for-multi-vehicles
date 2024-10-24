
import numpy as np
import scipy as sc
from agent import Agent
from Swarm import Swarm
import matplotlib.pyplot as plt
from decimal import Decimal as D
import random
my_seed = 10
random.seed(my_seed)
np.random.seed(my_seed)

MC_TOTAL = 20

TRUTH = []
EST = []
ERR = []


def compute_range_jac(state):
    adjacency = swarm.vehicles[0].adjacency
    nx = nx
    nb_agents = self.nb_agents
    for j in range(nb_agents):
        ego_idx = j
        vehicle_pos = state[0:2, j:j + 1]
        neighbor_idx_set = np.nonzero(adjacency[j, :])[0]
        for n in range(neighbor_idx_set.shape[0]):
            jac = np.zeros((1, nx * nb_agents))
            neighbor_idx = neighbor_idx_set[n]
            neighbor_pos = state[nx * neighbor_idx:nx * neighbor_idx + 2].reshape(2, 1)
            range_ = np.linalg.norm(vehicle_pos - neighbor_pos)
            jac[:, ego_idx * nx:((ego_idx + 1) * nx) - 1] = -(neighbor_pos - vehicle_pos).transpose()
            jac[:, neighbor_idx * nx:((neighbor_idx + 1) * nx) - 1] = -(-neighbor_pos + vehicle_pos).transpose()

            if j == 0 and n == 0:
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
for MC_RUN in range(MC_TOTAL):
    print(MC_RUN)

    #print('Initializing agents.........')
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
    tinc = 1.0#0.5 #sec
    vel         = 30 #m/s
    #omega_max = 10 #degrees/s
    std_omega = 0*np.deg2rad(0.57) #degrees/s
    std_v     = 0*0.01 #* 10 #m/s
    std_range = 0.01 * 100 #m
    S_Q = np.diag([0.1, 0.1, 0.01]) * Delta_t

    f_range = 10  # Hz
    f_odom = 10  # Hz
    # #

    f_waypt = 1
    estimated_states_history = []
    k_old = 0

    nb_agents = 5
    # adjacency = np.array([[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1],[1,0,0,0,0]])
    adjacency = (np.ones((nb_agents, nb_agents)) - np.eye(nb_agents))

    pos0 = 10 * np.array([[0., 10., 10., 0., 20.], [0., 0., 10., 10., 10.]])
    posf = pos0 + np.ones(
        (2, nb_agents)) * vel * 100  # 100*np.array([[30., 40., 40., 30.,50.],[30., 30., 40., 40., 40.]])
    #  #
    # print('Done.')

    swarm = Swarm()
    swarm.update_adjacency(adjacency)
    for i in range(nb_agents):
        swarm.add_vehicle(Delta_t, t, vel, std_omega, std_v, std_range, f_range, f_odom, S_Q)
    swarm.set_swarm_initPos(pos0)
    swarm.set_swarm_endpos(posf)

    swarm.update_adjacency(adjacency)


    prior_noise = 1e-4 * np.eye(nx * nb_agents)
    dynamics_noise =  np.kron(np.eye(nb_agents),np.diag([std_v * Delta_t, 0., std_omega * Delta_t]))  # in the body frame of each agent
    input_noise = np.kron(np.eye(nb_agents), np.diag([std_v ** 2, std_omega ** 2]))
    for j in range(nb_agents):
        if j == 0:
            process_noise = swarm.vehicles[j].S_Q.T @ swarm.vehicles[j].S_Q
        else:
            process_noise = sc.linalg.block_diag(process_noise, swarm.vehicles[j].S_Q.T @ swarm.vehicles[j].S_Q)

    X0 = np.zeros((nx, nb_agents))
    theta0 = [0, 0, 0, 0, 0]
    for j in range(nb_agents):
        X0[:, j:j + 1] = np.vstack((pos0[:, j:j + 1], np.array([[theta0[j]]])))
    X0 = X0.T.flatten().reshape(nx * nb_agents, 1)
    nX = X0.shape[0]
    x_predict = np.zeros((nX, len(t)))
    x_predict[:, 0:1] = X0
    P_predict = np.zeros((nX, nX * len(t)))
    P_predict[:, 0:nX] = prior_noise
    x_filtered = np.zeros((nX, len(t)))
    x_filtered[:, 0:1] = X0
    P_filtered = np.zeros((nX, nX * len(t)))
    P_filtered[:, 0:nX] = prior_noise
    k = 0
    for i in range(1, len(t)):
        for j in range(nb_agents):
            swarm.vehicles[j].states_est = x_filtered[j*nx:(j+1)*nx, i - 1:i].reshape((3, 1))
            swarm.vehicles[j].states_cov = P_filtered[:,(i-1) * nX:(i) * nX][j * nx:(j + 1) * nx, j * nx:(j + 1) * nx]
            swarm.vehicles[j].est_err = np.hstack(
                (swarm.vehicles[j].est_err, swarm.vehicles[j].states_est - swarm.vehicles[j].states))
        s = np.random.randint(0, swarm.nb_agents)
        swarm.MPC(optim_agent=None, use_cov=False, METRIC='obsv')
        u = np.zeros((nu*nb_agents,1))
        for j in range(nb_agents):
            u[j*nu:(j+1)*nu] = np.array([[swarm.vehicles[j].omega],[swarm.vehicles[j].v]])
        H = swarm.compute_range_jac(x_filtered[:, i - 1:i])
        for n in range(nb_agents):
            if n == 0:
                if i == 0:
                    P_predict = swarm.vehicles[n].states_cov
                    S_Q_block = S_Q
                A = agent.dyn_jacobian(x_filtered[:, i - 1:i], u,
                                   Delta_t)  # self.compute_dynamics_jac1(state, input, i)#
                B = agent.u_jacobian(x_filtered[:, i - 1:i], u,
                                   Delta_t)  # self.compute_dynamics_u_jac1(state, input, i)#
            else:
                if i == 0:
                    P_predict = sc.linalg.block_diag(P_predict, swarm.vehicles[n].states_cov)
                    S_Q_block = sc.linalg.block_diag(S_Q_block, S_Q)
                A = sc.linalg.block_diag(A, agent.dyn_jacobian(x_filtered[:, i - 1:i], u,
                                   Delta_t))  # self.compute_dynamics_jac1(state, input, i))#
                B = sc.linalg.block_diag(B, agent.u_jacobian(x_filtered[:, i - 1:i], u,
                                   Delta_t))  # self.compute_dynamics_u_jac1(state, input, i))#

        # Prediction
        x_predict[:, i:i + 1] = A @ x_filtered[:, i - 1:i] + B @ u
        P_predict[:, nX * i:nX * (i + 1)] = A @ P_predict[:,
                                                  nX * (i - 1):nX * i] @ A.transpose() +  process_noise

        # Update
        K = P_predict[:, i * nX:(i + 1) * nX] @ H.transpose() @ np.linalg.inv(
            H @ P_predict[:, i * nX:(i + 1) * nX] @ H.transpose() + np.kron(np.eye(H.shape[0]),std_range**2))
        range_meas = np.zeros((nb_agents, 1))
        for j in range(nb_agents):
            idx_set = np.nonzero(adjacency[j, :])[0]
            range_meas = swarm.vehicles[j].measRange_history[idx_set, i:i + 1]
        x_filtered[:, i:i + 1] = x_predict[:, i:i + 1] + K @ (range_meas - H @ x_predict[:, i:i + 1])
        P_filtered[:, i * nX:(i + 1) * nX] = (np.eye(nX) - K @ H) @ P_predict[:, i * nX:(i + 1) * nX]

    x_smoothed = np.zeros_like(x_filtered)
    x_smoothed[:, -1:] = x_filtered[:, -1:]
    P_smoothed = np.zeros_like(P_filtered)
    P_smoothed[:, -1 * nX:] = P_filtered[:, -1 * nX:]

    for i in range(len(t) - 2, -1, -1):
        A = P_filtered[:, nX * i:nX * (i + 1)] @ A.transpose() @ np.linalg.inv(
            P_predict[:, (i + 1) * nX:(i + 2) * nX])
        x_smoothed[:, i:i + 1] = x_filtered[:, i:i + 1] + A @ (x_smoothed[:, i + 1:i + 2] - x_predict[:, i + 1:i + 2])
        P_smoothed[:, i * nX:(i + 1) * nX] = P_filtered[:, i * nX:(i + 1) * nX] + A @ (
                    P_smoothed[:, (i + 1) * nX:(i + 2) * nX] - P_predict[:, (i + 1) * nX:(i + 2) * nX]) @ A.transpose()

    swarm.get_swarm_states_history_()
    # swarm.plot_swarm_traj()
    get_swarm_states_history = swarm.get_swarm_states_history

    swarm.get_swarm_est_err_()
    EST_ERR = swarm.EST_ERR

    estimated_states_history = []
    for j in range(nb_agents):
        x_sol = np.zeros((len(t), nx))
        for k in range(len(t)):
            x_sol[k, :] = x_smoothed[j * nx:(j + 1) * nx,k:k+1]
        swarm.vehicles[j].states_est = x_sol[j:j+1,k:k+1].reshape((3,1))
        swarm.vehicles[j].states_cov = P_smoothed[:, k*nX:(k+1)*nX][j*nx:(j+1)*nx, j*nx:(j+1)*nx]
        swarm.vehicles[j].est_err = np.hstack((swarm.vehicles[j].est_err, swarm.vehicles[j].states_est - swarm.vehicles[j].states))
        #print(swarm.vehicles[j].states_est.shape)
        estimated_states_history.append(x_sol.T)

    TRUTH.append(get_swarm_states_history)
    EST.append(estimated_states_history)
    ERR.append(EST_ERR)