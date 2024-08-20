import numpy as np

class Neighbor():
    def __init__(self, id):
        self.id = id
        self.states_est = np.array([-999, -999, -999])
        self.states_cov = np.zeros((3,3))
        self.est_err = np.empty((3,1))
        # self.data_shared = False
        self.meas_history = np.empty((2,1))


    def get_est(self):
        return self.states_est
    
    def set_est(self, states_est):
        self.states_est = states_est