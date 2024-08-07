import numpy as np

class Neighbor():
    def __init__(self, id):
        self.id = id
        self.states_est = np.array([-999, -999, -999])
        self.data_shared = False

    def get_est(self):
        return self.states_est
    
    def set_est(self, states_est):
        self.states_est = states_est