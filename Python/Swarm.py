import matplotlib.pyplot as plt
import numpy as np
from Vehicle import Vehicle

class Swarm():
    def __init__(self):
        self.vehicles = []
        self.nb_agents = 0
        self.timestamp = []
        self.meas_range = 0
        self.swarm_measRange_history = []

        self.est_X = []


    def add_vehicle(self, Delta_t, t, v, std_omega, std_v, std_range, f_range, f_odom):
        self.nb_agents += 1
        vehicle = Vehicle(Delta_t, t, v, std_omega, std_v, std_range, f_range, f_odom)
        self.vehicles.append(vehicle)

    def add_n_vehicles(self, vehicle, n):
        for i in range(n):
            self.add_vehicle(vehicle)


    def set_swarm_initPos(self, pos):
        for i in range(self.nb_agents):
            self.vehicles[i].set_initPos(pos[:,i:i+1])

    def set_swarm_endpos(self, pos):
        for i in range(self.nb_agents):
            self.vehicles[i].set_endPos(pos[:,i:i+1])

    def set_swarm_initPose(self, pose):
        for i in range(self.nb_agents):
            self.vehicles[i].set_initPose(pose[:,i:i+1])

    def set_swarm_endpose(self, pose):
        for i in range(self.nb_agents):
            self.vehicles[i].set_endPose(pose[:,i:i+1])

    def set_swarm_waypoints(self, waypoints):
        for i in range(self.nb_agents):
            self.vehicles[i].waypoints = waypoints[i]

    def set_swarm_endStates(self, states):
        for i in range(self.nb_agents):
            self.vehicles[i].set_endPos(states[:,i:i+1])

    def update_state(self, time):
        for i in range(self.nb_agents):
            self.vehicles[i].update_state(time)

    def addToLoop_estimator(self):
        self.vehicles[0].use_estimation = True
        self.vehicles[1].use_estimation = True

        #GTSAM
        self.vehicles[0].states_est = self.est_X[:3,:]
        self.vehicles[1].states_est = self.est_X[3:,:]

    def update_adjacency(self, adjacency):
        for i in range(self.nb_agents):
            self.vehicles[i].adjacency = adjacency
            self.vehicles[i].measRange_history = np.empty((self.nb_agents,1))
            self.vehicles[i].count = 0

    def update_measRange(self):
        #update sensor measurements

        measRange_matrix = np.zeros((self.nb_agents, self.nb_agents))
        for j in range(self.nb_agents):
            for jj in range(self.nb_agents):
                if self.vehicles[j].adjacency[j,jj] == 1:
                    vehicle_pos = self.vehicles[j].states[:2,:]
                    neighbor_pos = self.vehicles[jj].states[:2,:]
                    meas_range  =  np.linalg.norm(vehicle_pos - neighbor_pos) + self.vehicles[j].std_range*np.random.randn()

                    # if measRange_matrix[j, jj] == 0 and measRange_matrix[jj,j] == 0:
                    measRange_matrix[j,jj] = meas_range

        for j in range(self.nb_agents):
            self.vehicles[j].measRange_history = np.hstack((self.vehicles[j].measRange_history, measRange_matrix[j:j+1,:].transpose()))


    def  get_swarm_states_history_(self):
        self.get_swarm_states_history = []
        for i in range(self.nb_agents):
            self.get_swarm_states_history.append(self.vehicles[i].states_history)


    def plot_swarm_traj(self):
        for i in range(self.nb_agents):
            states = self.get_swarm_states_history[i]
            time = self.vehicles[i].sim_t
            plt.plot(states[0,:], states[1,:], label= 'Vehicle '+str(i))
            plt.quiver(states[0,:], states[1,:], np.cos(states[2,:]), np.sin(states[2,:]), scale= 20)
        plt.legend()
        plt.title('Vehicle trajectories')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.show()

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




