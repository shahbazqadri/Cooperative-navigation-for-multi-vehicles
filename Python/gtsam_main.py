import numpy as np
from Swarm import Swarm
import sympy as sp
import gtsam
from typing import Optional, List
from functools import partial
import matplotlib.pyplot as plt
from decimal import Decimal as D

print('Initializing agents.........')
Delta_t   = 0.05
# finding the number of decimal places of Delta_t
precision = abs(D(str(Delta_t)).as_tuple().exponent)
t         = np.arange(0,100,Delta_t)
t = np.round(t, precision) # to round off python floating point precision errors
v         = 50 #m/s
std_omega = np.deg2rad(0.57) #rad/s
std_v     = 0.01 #m/s
std_range = 0.01 #m
f_range   = 20 #Hz
f_odom    = 10 #Hz

swarm = Swarm()
nb_agents = 4
for i in range(nb_agents):
    swarm.add_vehicle(Delta_t, t, v, std_omega, std_v, std_range, f_range, f_odom)

adjacency = np.ones((nb_agents, nb_agents)) - np.eye(nb_agents)

swarm.update_adjacency(adjacency)

# Set initial and end position
pos0 = np.array([[0.,10.,10.,0.],[0.,0.,10.,10.]])
posf = np.array([[30., 40., 40., 30.],[30., 30., 40., 40]])
swarm.set_swarm_initPos(pos0)
swarm.set_swarm_endpos(posf)
print('Done.')

print('Propagating true state and generating measurments........')
# propagate the swarm system
for tt in t:
    # update vehicle states and plot
    swarm.update_state(tt)
    swarm.update_measRange()
# swarm.get_swarm_states_history()
# swarm.plot_swarm_traj()

print('Done.')

print('Initializing factor graph...........')
S0 = 1e-4*np.eye(3)
# S0[2,2] = 1.
prior_noise = gtsam.noiseModel.Gaussian.Covariance(S0)
dynamics_noise = gtsam.noiseModel.Gaussian.Covariance(S0)#(std_v*Delta_t)**2, 0., (std_omega*Delta_t)**2]))
range_noise = gtsam.noiseModel.Gaussian.Covariance(np.diag([std_range**2]))
odom_noise = gtsam.noiseModel.Gaussian.Covariance(np.diag([std_v**2, std_omega**2]))
q_noise = gtsam.noiseModel.Gaussian.Information(np.diag([1.,1.,1.]))
qT_noise = gtsam.noiseModel.Gaussian.Information(np.diag([100,100,10]))
# Create an empty Gaussian factor graph
graph = gtsam.NonlinearFactorGraph()

## odometry jacobian
dt,v_, x_, y_, theta_,g1, g2 = sp.symbols('dt,  v_, x, y, theta, g1, g2')
vehicle_pos = sp.Matrix([[x_],[y_]])
rel_pos =  sp.Matrix([[g1],[g2]]) - vehicle_pos
vehicle_vel = sp.Matrix([[v_*sp.cos(theta_)],[v_*sp.sin(theta_)]])
norm_relpos = sp.sqrt(sum(sp.matrices.dense.matrix_multiply_elementwise(rel_pos,rel_pos)))
norm_vehvel = sp.sqrt(sum(sp.matrices.dense.matrix_multiply_elementwise(vehicle_vel,vehicle_vel)))

dot = sum(sp.matrices.dense.matrix_multiply_elementwise(rel_pos,vehicle_vel))
a = sp.acos(dot/(norm_vehvel*norm_relpos))
odom_jac = sp.diff(a, sp.Matrix([[x_],[y_],[theta_]])).simplify().simplify()

def error_dyn(agent_idx, this: gtsam.CustomFactor,
              values: gtsam.Values,
              jacobians: Optional[List[np.ndarray]]):
    key1 = this.keys()[0]
    key2 = this.keys()[1]

    x1, Xp1 = values.atVector(key1), values.atVector(key2)
    x1 = x1.reshape(3,1)
    swarm.vehicles[agent_idx].states = x1
    swarm.vehicles[agent_idx].update_controller()
    swarm.vehicles[agent_idx].update_kinematics()
    # swarm.vehicles[agent_idx].states = swarm.vehicles[agent_idx].next_states
    nextstate = swarm.vehicles[agent_idx].next_states

    v = swarm.vehicles[agent_idx].v
    omega = swarm.vehicles[agent_idx].omega
    error = Xp1 - nextstate.reshape(3,)

    if jacobians is not None:
        jacobians[1] = np.eye(3)
        jacobians[0] = -np.array([[1,0,0],[0,1,0],[-Delta_t*v*np.sin(Delta_t*omega/2 + x1[2,0])*np.sinc(0.159154943091895*Delta_t*omega),Delta_t*v*np.cos(Delta_t*omega/2 + x1[2,0])*np.sinc(0.159154943091895*Delta_t*omega), 1]]).transpose()

    return error


def error_range(measurement, this: gtsam.CustomFactor,
              values: gtsam.Values,
              jacobians: Optional[List[np.ndarray]]):
    key1 = this.keys()[0]
    key2 = this.keys()[1]

    v, n = values.atVector(key1), values.atVector(key2)
    vehicle_pos = v[:2].reshape(2,1)
    neighbor_pos = n[:2].reshape(2,1)
    range = np.linalg.norm(vehicle_pos - neighbor_pos)
    error = ( range - measurement).reshape(1,)

    if jacobians is not None:
        if range != 0:
            jacobians[1] = (1./range)*np.hstack(((neighbor_pos - vehicle_pos).transpose(), np.array([[0.]]))).reshape(1,3)
            jacobians[0] = (1./range)*np.hstack(((-neighbor_pos + vehicle_pos).transpose(), np.array([[0.]]))).reshape(1,3)
        else:
            jacobians[1] = np.zeros((1,3))
            jacobians[0] = np.zeros((1,3))
    return error

def error_odom(time, agent_idx, measurement, this: gtsam.CustomFactor,
              values: gtsam.Values,
              jacobians: Optional[List[np.ndarray]]):
    key1 = this.keys()[0]

    x = values.atVector(key1)
    x = x.reshape(3,1)
    swarm.vehicles[agent_idx].update_controller()
    v = swarm.vehicles[agent_idx].v
    omega = swarm.vehicles[agent_idx].omega
    error = (np.array([[v], [omega]]) - measurement).reshape(2,)

    if jacobians is not None:
        target = swarm.vehicles[agent_idx].target_point
        x = swarm.vehicles[agent_idx].vehicle_pos[0,0]
        y = swarm.vehicles[agent_idx].vehicle_pos[1,0]
        theta = swarm.vehicles[agent_idx].vehicle_theta[0,0]
        if swarm.vehicles[agent_idx].vec_cos < -1 or swarm.vehicles[agent_idx].vec_cos > 1:
            jac_omega = 0
        else:
            jac_omega = odom_jac.evalf(subs={dt:Delta_t, v_:v, x_:x , y_:y, theta_:theta,g1: target[0,0], g2:target[1,0]})
            jac_omega = np.array(jac_omega).astype(np.float64).transpose() #converting sympy matrix to numpy array

        if swarm.vehicles[agent_idx].cross_product[2] < 0:
            jac_omega = -jac_omega

        if swarm.vehicles[agent_idx].omega_inlimit is True:
            jacobians[0] = np.vstack(((np.zeros((1,3))), (1/Delta_t)*jac_omega))

        else:
            jacobians[0] = np.zeros((2,3))
    return error




# Create the keys corresponding to unknown variables in the factor graph
X = []
U = []
for k in range(len(t)):
    for j in range(nb_agents):
        X.append(gtsam.symbol('x', k*nb_agents+j))

v = gtsam.Values()
# set initial state as prior
for j in range(nb_agents):
    X0 = np.vstack((pos0[:,j:j+1],np.array([[0.]])))
    graph.add(gtsam.PriorFactorVector(X[j], X0, prior_noise))
    v.insert(X[j], X0)
    Xf = np.vstack((posf[:, j:j + 1], np.array([[0.]])))
    # graph.add(gtsam.PriorFactorVector(X[j], X0, prior_noise))
    v.insert(X[(len(t)-1) * nb_agents + j], Xf)
for k in range(len(t)):
    print('time = {}'.format(t[k]))
    for j in range(nb_agents):

        if k < len(t) - 1:
            gf = gtsam.CustomFactor(dynamics_noise, [X[k * nb_agents + j], X[(k + 1) * nb_agents + j]],
                                    partial(error_dyn, j))
            graph.add(gf)

        if k > 0:


           range_period = 1./f_range
           if D(str(t[k])) % D(str(range_period))  == 0:
                for jj in range(nb_agents):
                    if swarm.vehicles[j].adjacency[j, jj] == 1:
                        gfrange = gtsam.CustomFactor(range_noise, [X[k*nb_agents+j],X[k*nb_agents+jj]],
                                        partial(error_range, swarm.vehicles[j].measRange_history[jj,k]))  # np.array([X[k]])
                        graph.add(gfrange)

           odom_period = 1. / f_odom
           if D(str(t[k])) % D(str(odom_period)) == 0:
               idx = int(t[k]//odom_period)
               # print(idx)
               gfodom = gtsam.CustomFactor(odom_noise, [X[k * nb_agents + j]],
                                            partial(error_odom, t[k],j, swarm.vehicles[j].meas_history[:,idx:idx+1]))  # np.array([X[k]])
               graph.add(gfodom)

           if k < len(t) - 1:
                v.insert(X[k*nb_agents+j], np.full((3, 1), 0.))
print(graph)
print('Done.')
print('Performing factor graph optimization........')
params = gtsam.GaussNewtonParams()
optimizer = gtsam.GaussNewtonOptimizer(graph, v, params)
result = optimizer.optimize()
print('Done.')
print('Reshaping results for plotting........')
x_res = []
for j in range(nb_agents):
    x_sol = np.zeros((len(t), 3))
    for k in range(len(t)):
        x_sol[k, :] = result.atVector(X[k])
    np.savetxt('x1.csv', x_sol, delimiter= ',')
    x_res.append(x_sol)

print('Done')

marginals = gtsam.Marginals(graph, result)
for j in range(nb_agents):
    print("Final state Covariance on agent {}:\n{}\n".format(j, marginals.marginalCovariance(X[len(t)-1])))
    states = x_res[j].transpose()
    # states_ = swarm.get_swarm_states_history[j]
    time = t
    plt.plot(states[0, :], states[1, :], label='estmated Vehicle ' + str(j))
    # plt.plot(states_[0, :], states_[1, :], label='true Vehicle ' + str(i))
    # plt.quiver(states[0, :], states[1, :], np.cos(states[2, :]), np.sin(states[2, :]), scale=20)
    plt.legend()
    plt.title('Vehicle trajectories')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.show()
for j in range(nb_agents):
    print("Final state Covariance on agent {}:\n{}\n".format(j, marginals.marginalCovariance(X[len(t)-1])))
    states = x_res[j].transpose()
    states_ = swarm.vehicles[j].states_history
    time = t
    plt.plot(states[0, :], states[1, :], label='estmated Vehicle ' + str(j))
    plt.plot(states_[0, :], states_[1, :], label='true Vehicle ' + str(j))
    # plt.quiver(states[0, :], states[1, :], np.cos(states[2, :]), np.sin(states[2, :]), scale=20)
plt.legend()
plt.title('Vehicle trajectories')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.show()

