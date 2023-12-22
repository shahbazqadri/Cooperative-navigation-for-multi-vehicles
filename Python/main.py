import decimal

import numpy as np
from Swarm import Swarm
import gtsam
from typing import Optional, List
from functools import partial

Delta_t = 0.1
# finding the number of decimal places of Delta_t
precision = abs(decimal.Decimal(str(Delta_t)).as_tuple().exponent)
t       = np.arange(0,10,Delta_t)
t = np.round(t, precision) # to round off python floating point precision errors
v       = 30 #m/s
std_omega = np.deg2rad(0.57) #rad/s
std_v     = 0.01 #m/s
std_range = 0.01 #m
f_range   = 10 #Hz
f_odom    = 10 #Hz

swarm = Swarm()
nb_agents = 4
for i in range(nb_agents):
    swarm.add_vehicle(Delta_t, t, v, std_omega, std_v, std_range, f_range, f_odom)

adjacency = np.ones((nb_agents, nb_agents)) - np.eye(nb_agents)

swarm.update_adjacency(adjacency)

# Set initial and end position
pos0 = np.array([[0.,100.,100.,0.],[0.,0.,100.,100.]])
posf = np.array([[3000., 4000., 4000., 3000.],[3000., 3000., 4000., 4000]])
swarm.set_swarm_initPos(pos0)
swarm.set_swarm_endpos(posf)

# propagate the swarm system
for tt in t:
    # update vehicle states and plot
    swarm.update_state(tt)
    swarm.update_measRange()
for j in range(nb_agents):
    print(swarm.vehicles[j].count)
swarm.get_swarm_states_history()
# swarm.plot_swarm_traj()
# swarm.plot_swarm_heading()
