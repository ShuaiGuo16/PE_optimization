### Definition of manager class
import numpy as np
import utility
from two_sources import thermal_distribution_maxT


class region_manager:

    def __init__(self, GP, n_batch, Data, Tjmax, X_seed, Q1, Q2):
        self.GP = GP
        self.n_batch = n_batch
        self.Data = (25, 50e-3, 65e-3, 61.4e-3, 106e-3)
        self.Tjmax = Tjmax
        self.opt = [X_seed]
        self.converge_flag = False
        self.Q1 = Q1
        self.Q2 = Q2


    def _lb_ub(self, lb, ub):
        d, b, L, c, L_duct, n = self.opt[-1][2:8]
        features = [d, b, L, c, L_duct, n]
        self.lb, self.ub = [], []
        for j in range(len(features)):
            self.lb.append(np.max((lb[j], features[j]-(ub[j]-lb[j])/self.n_batch)))
            self.ub.append(np.min((ub[j], features[j]+(ub[j]-lb[j])/self.n_batch)))

        self.lb = np.array(self.lb)
        self.ub = np.array(self.ub)


    def init_weight(self, w):
        self.w = []
        self.assign_weight(w)


    def assign_weight(self, w):
        self.w.append(w)


    def update_model(self, new_model):
        self.GP = new_model


    def branch_and_bound(self, lb, ub, scaler, candidate_num=5000):

        # Create initial lb & ub bounds
        self._lb_ub(lb, ub)

        # Sample new candidates within the region
        new_candidates = utility.generate_candidates(self.lb, self.ub, num=candidate_num)
        Q1_array, Q2_array = self.Q1*np.ones((new_candidates.shape[0], 1)), self.Q2*np.ones((new_candidates.shape[0], 1))
        new_candidates = np.hstack((Q1_array, Q2_array, new_candidates))
        print(f"Generated {new_candidates.shape[0]} new candidates!")

        # Calculate new acquisitions
        _, index = utility.acquisition(self.GP, None, None, new_candidates, scaler, self.Tjmax)
        self.opt.append(new_candidates[index])

        # Calculate Tmax
        Tmax, _ = thermal_distribution_maxT(new_candidates[index], self.Data)

        # Update weight
        if Tmax <= self.Tjmax:
            w = utility.evaluate_weight(new_candidates[[index]])[0]

        else:
            w = np.inf

        self.assign_weight(w)

        return new_candidates[index], Tmax


    def check_converge(self):
        # TODO: need to improve the convergence checking?
        con1 = np.sum(np.array(self.w)!=np.inf) >= 2
        con2 = self.w[-1] >= 0.99*np.min(self.w[:-1])

        if con1 and con2:
            assert len(self.opt) == len(self.w), "Mismatch list length!"
            best_design = self.opt[np.argmin(self.w)]
            best_weight = np.min(self.w)
            self.converge_flag = True

        else:
            best_design = None
            best_weight = None
            self.converge_flag = False

        return best_design, best_weight, self.converge_flag
