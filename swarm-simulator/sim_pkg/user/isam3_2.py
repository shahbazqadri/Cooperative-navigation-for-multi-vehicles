import numpy as np
from decimal import Decimal as D
import random
import gtsam
from user.agent import Agent
from user.Drone import Drone
from typing import Optional, List
import scipy as sc
from functools import partial
import time
############################## DEBUGGING ##############################
import sys                                                            #
import pdb                                                            #
import logging                                                        #
#######################################################################
def trace_calls(frame, event, arg):                                   #
    if event != 'call':                                               #
        return                                                        #
    code = frame.f_code                                               #
    func_name = code.co_name                                          #
    file_name = code.co_filename                                      #
    file_names = ('isam3_2.py', 'Vehicle.py', 'agent.py', 'Drone.py', 'Neighbor.py') #
    if any(file_name.endswith(f) for f in file_names):                #
        line_no = frame.f_lineno                                      #
        print(f"Call to {func_name} in {file_name}:{line_no}")        #
    return trace_calls                                                #
#######################################################################
logging.basicConfig(level=logging.DEBUG)                              # 
#######################################################################
def usr(robot):
    # sys.settrace(trace_calls)
    id_var = robot.id
    # Initialize parameters and random seed
    my_seed = 10
    random.seed(my_seed)
    np.random.seed(my_seed)
    TRUTH = []
    EST = []
    ERR = []
    
    agent = Agent()
    unicycle = agent.unicycle
    nx = 3
    nu = 2
    Delta_t = 0.1
    T = 1500  # seconds

    precision = abs(D(str(Delta_t)).as_tuple().exponent)
    t = np.arange(0, T, Delta_t)
    t = np.round(t, precision)
    tinc = 1.0
    vel = 30 # m/s
    std_omega = 0 * np.deg2rad(0.57)
    std_v = 0 * 0.01
    std_range = 0.01 * 100
    S_Q = np.diag([0.1, 0.1, 0.01]) * Delta_t
    f_range = 10  # Hz
    f_odom = 10  # Hz

    f_waypt = 1
    estimated_states_history = []

    print (t)
    drone = Drone(id_var, Delta_t, t, vel, std_omega, std_v, std_range, f_range, f_odom, S_Q)

    adjacency = np.ones((5,5))-np.eye(5)#np.array([[0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [1, 0, 0, 0, 0]])
    # adjacency_bi_dir = adjacency + adjacency.T
    drone.update_adjacency(adjacency)
    for neighbor_id,val in enumerate(drone.adjacency):
        if val == 1:
            drone.add_neighbor(neighbor_id)
    
    print(f"Drone {drone.id} neighbors: {[neighbor.id for neighbor in drone.neighbors]}")

    pos0 = 5 * np.array([[0., 10., 10., 0., -10.], [0., 0., 10., 10., 0.]])
    theta0 = [0, 0, 0, 0, 0]
    posf = pos0 + np.ones(pos0.shape) * vel * 50
    print(f"id: {drone.id}, pos0: {pos0[:,drone.id:drone.id+1]}, posf: {posf[:,drone.id:drone.id+1]}")
    drone.vehicle.set_initPos(pos0[:,drone.id:drone.id+1])
    drone.vehicle.set_endPos(posf[:,drone.id:drone.id+1])
    print("nb_neighbors: ", drone.nb_neighbors)
    # ISAM2 Setup       -       CHANGED nb_agents to (nb_neighbors + 1)
    S0 = 1e-4*np.eye(nx * (drone.nb_neighbors + 1))
    print(S0.shape)
    prior_noise = gtsam.noiseModel.Gaussian.Covariance(S0)
    print("PriorNoise: ", prior_noise)
    print()
    dynamics_noise = gtsam.noiseModel.Constrained.Sigmas(np.array([std_v*Delta_t, 0., std_omega*Delta_t] * (drone.nb_neighbors + 1)).reshape(nx * (drone.nb_neighbors + 1), 1)) # in the body frame of each agent
    cov = np.kron(np.eye(drone.nb_neighbors + 1), np.diag([std_v**2, std_omega**2]))
    input_noise = gtsam.noiseModel.Gaussian.Covariance(cov)
    
    # Initialize the process noise covariance matrix S
    for j in range(drone.nb_neighbors + 1):
        if j == 0:
            # Acces the covariance matrix from the drone itself
            S = drone.vehicle.S_Q.T @ drone.vehicle.S_Q
        else:
            # Access the covariance matrix of the drone's neighbors
            S = sc.linalg.block_diag(S, drone.vehicle.S_Q.T @ drone.vehicle.S_Q)
    process_noise = gtsam.noiseModel.Gaussian.Covariance(S)

    def parse_result(result, cov, nb_agents, t):
        x_res = []
        for j in range(drone.nb_neighbors + 1):
            x_sol = np.zeros((len(t), nx))
            if j == drone.id:
                for k in range(len(t)):
                    x_sol[k, :] = result.atVector(X[k])[j * nx:(j + 1) * nx]
                drone.vehicle.states_est = x_sol[-1, :].reshape((3, 1))
                drone.vehicle.states_cov = cov[j * nx:(j + 1) * nx, j * nx:(j + 1) * nx]
                drone.vehicle.est_err = np.hstack((drone.vehicle.est_err, drone.vehicle.states_est - drone.vehicle.states))
            else:
                for k in range(len(t)):
                    x_sol[k, :] = result.atVector(X[k])[j * nx:(j + 1) * nx]
                drone.neighbors[j].states_est = x_sol[-1, :].reshape((3, 1))
                drone.neighbors[j].states_cov = cov[j * nx:(j + 1) * nx, j * nx:(j + 1) * nx]
                drone.neighbors[j].est_err = np.hstack((drone.vehicle.est_err, drone.vehicle.states_est - drone.vehicle.states))
            x_res.append(x_sol.T)
        return x_res

    def error_dyn(measurements, this: gtsam.CustomFactor, values: gtsam.Values, jacobians: Optional[List[np.ndarray]]):
        key1 = this.keys()[0]
        key2 = this.keys()[1]
        X_, Xp1 = values.atVector(key1), values.atVector(key2)
        u = measurements.reshape(nu * (drone.nb_neighbors + 1), 1)
        u_cal = np.zeros((nu * (drone.nb_neighbors + 1), 1))
        for j in range(drone.nb_neighbors + 1):
            u_cal[j * nu:(j + 1) * nu, :] = unicycle.find_u(X_[j * nx:(j + 1) * nx].reshape(nx, 1), Xp1[j * nx:(j + 1) * nx].reshape(nx, 1), Delta_t)

        if jacobians is not None:
            for j in range(drone.nb_neighbors + 1):
                jac0, jac1 = unicycle.find_u_jacobian(X_[j * nx:(j + 1) * nx].reshape(nx, 1), Xp1[j * nx:(j + 1) * nx].reshape(nx, 1), Delta_t)
                if j == 0:
                    jacobians[0] = jac0
                    jacobians[1] = jac1
                else:
                    jacobians[0] = sc.linalg.block_diag(jacobians[0], jac0)
                    jacobians[1] = sc.linalg.block_diag(jacobians[1], jac1)

        error = (u_cal - u).reshape(nu * (drone.nb_neighbors + 1), )
        return error

    def error_range(ego_idx, neighbor_idx_set, measurement, this: gtsam.CustomFactor, values: gtsam.Values, jacobians: Optional[List[np.ndarray]]):
        key1 = this.keys()[0]
        X_ = values.atVector(key1)
        n = measurement.shape[0]
        range_est = np.zeros((n, 1))
        vehicle_pos = X_[ego_idx * nx:((ego_idx + 1) * nx) - 1].reshape(2, 1)
        for j in range(n):
            jac = np.zeros((1, nx * (drone.nb_neighbors + 1)))
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
                        jacobians[0] = np.zeros((1, nx * (drone.nb_neighbors + 1)))
                else:
                    if range_ != 0:
                        jacobians[0] = np.vstack((jacobians[0], (1. / range_) * jac))
                    else:
                        jacobians[0] = np.vstack((jacobians[0], np.zeros((1, nx * (drone.nb_neighbors + 1)))))

        error = (range_est - measurement.reshape(n, 1)).reshape(n, )
        return error

    graph = gtsam.NonlinearFactorGraph()    
    X = []
    for k in range(len(t)):
        X.append(gtsam.symbol('x', k))
    v = gtsam.Values()

    # Set the first state for this drone and its neghbors
    X0 = np.zeros((nx, drone.nb_neighbors + 1))
    X0[:, 0:1] = np.vstack((pos0[:, drone.id].reshape(2,1), np.array([[theta0[drone.id]]])))
    for j in range(drone.nb_neighbors + 1):
        if j < drone.nb_neighbors:
            X0[:, j:j + 1] = np.vstack((pos0[:, drone.neighbors[j].id:drone.neighbors[j].id + 1], np.array([[theta0[drone.neighbors[j].id]]])))   
        else:
            X0[:, j:j + 1] = np.vstack((pos0[:, drone.id:drone.id + 1], np.array([[theta0[drone.id]]])))
    X0 = X0.T.flatten().reshape(nx * (drone.nb_neighbors + 1), 1)

    

    if (not v.exists(X[0])) and (not graph.exists(X[0])):
        graph.add(gtsam.PriorFactorVector(X[0], X0, prior_noise))
        v.insert(X[0], X0)
    
        
        
    count = 0
    X_val = X0
    # idx = 0
    initialized = False
    first = True
    isam = gtsam.ISAM2()

    current_time = robot.get_clock()
    start_time = current_time
    tt=0.0
    k=0

    
    robot.set_vel(vel, vel)
    # while k < len(t):
    while tt < t[len(t) - 1]:
############################### DATA TRANSFER #########################################################
        while True:
            time.sleep(0.1)
            current_pos = robot.get_pose()
            if not current_pos:
                # print("GOT A FALSE FLAG FROM CURRENT POS")
                continue
            break

        current_time = robot.get_clock()
        tt = current_time - start_time

        # print(current_time)
        # Each drone will send data to all neighbors in the following format
    
        # Initialize variables to track received states and communication status
        received_states = {neighbor.id: False for neighbor in drone.neighbors}

        # Step 1: Drone with ID 0 sends its state first
        current_state_est = np.array([current_pos[0], current_pos[1], current_pos[2]])
        data_to_send = f"id:{drone.id};state_est:{current_state_est}"
        
        # Send to all neighbors
        for neighbor in drone.neighbors:
            robot.send_msg(f"{data_to_send}")
        
        # Wait for responses from all other drones
        for _ in range(len(drone.neighbors)):
            recv = robot.recv_msg(clear=False)
            if len(recv) > 0:
                for msg in recv:
                    sender_id = int(msg.split(";")[0].split(":")[1])
                    if sender_id in received_states:
                        state_est_str = msg.split(";")[1].split(":")[1]
                        state_est = np.array([float(x) for x in state_est_str.strip("[]").split()])
                        for neighbor in drone.neighbors:
                            if neighbor.id == sender_id:
                                neighbor.set_est(state_est)
                        received_states[sender_id] = True
                        # print(f"Drone {drone.id} received states: {received_states}")

        if drone.id == 0:
            robot.recv_msg(clear=True)

        # Update swarm states
        drone.vehicle.update_state(t[k])
        drone.update_measRange()
        
        if tt < t[k]:
            continue

        # if tt < t[len(t) - 1]:
            # Dynamics factor
        odom_period = 1. / f_odom
        if D(str(t[k])) % D(str(odom_period)) == 0.:
            for j in range(drone.nb_neighbors + 1):
                # drone.vehicle.meas_history = np.delete(drone.vehicle.meas_history, 0, 1)            
                if first:
                    drone.vehicle.meas_history = np.delete(drone.vehicle.meas_history, 0, 1)
                    first = False
                if j == 0:
                    meas_history = drone.vehicle.meas_history[:,-1:]
                else:
                    meas_history = np.vstack((meas_history, drone.neighbors[j - 1].meas_history[:, -1:]))
       
            # if not graph.exists(X[k]):
            gf = gtsam.CustomFactor(process_noise, [X[k], X[(k + 1)]], partial(error_dyn, meas_history))
            graph.add(gf)
            # else:
                # print(f"Factor for {X[k]} already exist")
            # Initial values for optimizer
            X_val[:nx, :] = unicycle.discrete_step(X_val[:nx, :], meas_history[:nu, :].reshape(nu, 1), Delta_t)
            for j, neighbor in enumerate(drone.neighbors):
                state_est = neighbor.get_est().reshape(nx, 1)
                X_val[(j + 1) * nx:(j + 2) * nx, :] = state_est
            
            if (not v.exists(X[k + 1])) and (not graph.exists(X[k + 1])):
                v.insert(X[k + 1], X_val)
            # else:
            #     print(f"prior factor for {X[k + 1]} already exists")

        # Range measurement factor
        range_period = 1. / f_range
        if D(str(t[k])) % D(str(range_period)) == 0.:
            # range_meas = np.zeros((drone.nb_neighbors + 1, 1))
            for neighbor in drone.neighbors:
                idx_set = [neighbor.id]
                range_meas = np.linalg.norm(current_pos[:2] - neighbor.get_est()[:2].reshape(2, 1)) + std_range * np.random.randn() #Changed this
                range_noise = gtsam.noiseModel.Gaussian.Covariance(np.diag([std_range ** 2] * len(idx_set)))
                gfrange = gtsam.CustomFactor(range_noise, [X[k]], partial(error_range, drone.id, idx_set, range_meas))
                graph.add(gfrange)

        # Incremental smoothing
        if k > (1.0 / Delta_t) - 1 and count > (1.0 / Delta_t) - 1:
            if not initialized:
                params = gtsam.LevenbergMarquardtParams()
                if drone.id == 0:
                    print("params: ", params)
                    print("graph size: ", graph.size())
                    print("v: ", v)
                    # print("Noise Model Size: ", prior_noise.size())
                optimizer = gtsam.LevenbergMarquardtOptimizer(graph, v, params)
                v = optimizer.optimize()
                initialized = True

            # ISAM2 update
            isam.update(graph, v)
            result = isam.calculateEstimate()
            cov = isam.marginalCovariance(X[k])
            estimated_states_history = parse_result(result, cov, drone.nb_neighbors + 1, t[:k+1])
            drone.MPC(optim_agent=None, use_cov=False, METRIC='obsv')
            graph = gtsam.NonlinearFactorGraph()
            v = gtsam.Values()

            # Send control commands to the robot
            # 1. Assuming its in Meters
            # 2. Need to varify the wheel spacing
            vel_w1 = vel - drone.vehicle.omega * 0.1
            vel_w2 = vel + drone.vehicle.omega * 0.1
            # if drone.id == 0:
                # print(f"Drone {drone.id} Wheel Velocity: {vel_w1, vel_w2}")
            robot.set_vel(vel_w1, vel_w2)

            if k < len(t) - 1:
                X0 = result.atVector(X[k + 1]).reshape(nx * (drone.nb_neighbors + 1), 1)
                X_val = X0
                count = 0
        else:
            count += 1

        k += 1
        current_pos = robot.get_pose()
        
        # robot.velocity = (30, drone.omega)
        # robot.integrate(Delta_t)
    drone.get_drone_states_history_()
    EST_ERR = drone.EST_ERR
    # Debugging print statements to verify contents of graph and v
    
    isam.update(graph, v)
    result = isam.calculateEstimate()
    cov = isam.marginalCovariance(X[k])
    estimated_states_history = parse_result(result, cov, (drone.nb_neighbors + 1), t[:k + 1])
    TRUTH.append(drone.get_drone_states_history)
    EST.append(estimated_states_history)
    ERR.append(EST_ERR)
    k += 1