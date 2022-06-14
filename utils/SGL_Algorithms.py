#%%
import time # record the efficiency
import itertools
import numpy as np
import pandas as pd
import cvxpy as cp
from hyperopt import tpe, hp, fmin  # for Bayesian method

from HC_SGL import SGL_Hillclimb

from tqdm import tqdm

from utils.utils import Monitor, Monitor_DC

#%%
def train_error(settings, data, x):
    return .5 / settings.num_train * np.sum(np.square( data.y_train - data.X_train @ x ))

def validation_error(settings, data, x):
    return .5 / settings.num_validate * np.sum(np.square( data.y_validate - data.X_validate @ x ))

def test_error(settings, data, x):
    return .5 / settings.num_test * np.sum(np.square( data.y_test - data.X_test @ x ))

class Training_model:
    def __init__(self, data_info) -> None:  
        settings = data_info.settings
        data = data_info.data
        n, p, M = settings.num_train, settings.num_features, settings.num_experiment_groups
        cal_group_sizes = [p//M] * M
        group_ind = np.concatenate( [[0], np.cumsum(cal_group_sizes)] )

        self.x = cp.Variable(p)
        self.lam = cp.Parameter(M+1, nonneg=True)
        LS_lower = .5/n * cp.sum_squares( data.y_train - data.X_train @ self.x )
        group_lasso_regularization = cp.sum([self.lam[i]*cp.pnorm(self.x[group_ind[i]:group_ind[i+1]], 2) for i in range(M)])
        sparsity_regularization = cp.pnorm(self.x, 1)
        self.training_problem = cp.Problem(cp.Minimize(LS_lower + group_lasso_regularization + self.lam[-1]*sparsity_regularization))

    def solve_training(self, lam):
        self.lam.value = lam
        self.training_problem.solve()
        return self.x.value

class Training_model_simple:
    def __init__(self, data_info) -> None:  
        settings = data_info.settings
        data = data_info.data
        n, p, M = settings.num_train, settings.num_features, settings.num_experiment_groups
        cal_group_sizes = [p//M] * M
        group_ind = np.concatenate( [[0], np.cumsum(cal_group_sizes)] )

        self.x = cp.Variable(p)
        self.lam = cp.Parameter(2, nonneg=True)
        LS_lower = .5/n * cp.sum_squares( data.y_train - data.X_train @ self.x )
        group_lasso_regularization = cp.sum([cp.pnorm(self.x[group_ind[i]:group_ind[i+1]], 2) for i in range(M)])
        sparsity_regularization = cp.pnorm(self.x, 1)
        self.training_problem = cp.Problem(cp.Minimize(LS_lower + self.lam[0]*group_lasso_regularization + self.lam[-1]*sparsity_regularization))

    def solve_training(self, lam):
        self.lam.value = lam
        self.training_problem.solve()
        return self.x.value

#%%
# iP-DCA
def iP_DCA(data_info, DC_Setting = dict()):
    settings = data_info.settings
    data = data_info.data
    p, M = settings.num_features, settings.num_experiment_groups
    cal_group_sizes = [p//M] * M
    group_ind = np.concatenate( [[0], np.cumsum(cal_group_sizes)] )

    MAX_ITERATION = DC_Setting["MAX_ITERATION"] if "MAX_ITERATION" in DC_Setting.keys() else 100
    TOL = DC_Setting["TOL"] if "TOL" in DC_Setting.keys() else 5e-2
    r = DC_Setting["initial_guess"] if "initial_guess" in DC_Setting.keys() else .1*np.ones(M+1)
    x = np.zeros(settings.num_features)

    # subproblems define
    class DC_lower:
        def __init__(self, settings, data) -> None:
            self.x_lower = cp.Variable(settings.num_features)
            self.r_lower = cp.Parameter(M+1, nonneg=True)
            self.constraints_lower = [cp.pnorm(self.x_lower[group_ind[i]:group_ind[i+1]], 2) <= self.r_lower[i] for i in range(M)] + [cp.pnorm(self.x_lower, 1) <= self.r_lower[M]]
            LS_lower = .5 / settings.num_train * cp.sum_squares( data.y_train - data.X_train @ self.x_lower )
            self.dc_lower = cp.Problem(cp.Minimize(LS_lower), self.constraints_lower)

        def solve(self, r):
            ECOS_TOL = 1e-4
            ECOS_ITER = 100
            self.r_lower.value = r
            result = self.dc_lower.solve(solver = cp.ECOS, abstol=ECOS_TOL, reltol=ECOS_TOL, abstol_inacc=ECOS_TOL, reltol_inacc=ECOS_TOL, max_iters=ECOS_ITER)
            return result, self.x_lower.value

        def dual_value(self):
            return np.array([float(self.constraints_lower[i].dual_value) for i in range(M+1)])

    class DC_approximated:
        def __init__(self, settings, data, DC_Setting = dict()) -> None:
            self.delta = DC_Setting["delta"] if "delta" in DC_Setting.keys() else 5.
            self.c_alo = DC_Setting["c"] if "c" in DC_Setting.keys() else .1
            epsilon_alo = DC_Setting["epsilon"] if "epsilon" in DC_Setting.keys() else 0
            rho = DC_Setting["rho"] if "rho" in DC_Setting.keys() else 5e-3
            beta_0 = DC_Setting["beta_0"] if "beta_0" in DC_Setting.keys() else 1

            self.x_upper, self.r = cp.Variable(settings.num_features), cp.Variable(M+1)
            self.x_upper_k, self.r_k = cp.Parameter(settings.num_features), cp.Parameter(M+1, pos=True)
            self.gamma_k, self.bias_k = cp.Parameter(M+1), cp.Parameter()
            self.beta_k = cp.Parameter(pos=True)
            self.beta_k.value = beta_0

            LS_upper = .5/settings.num_validate * cp.sum_squares( data.y_validate - data.X_validate @ self.x_upper )
            prox = cp.sum_squares(self.x_upper - self.x_upper_k) + cp.sum_squares(self.r - self.r_k) 
            beta_k_V_k = self.beta_k * .5 / settings.num_train * cp.sum_squares( data.y_train - data.X_train @ self.x_upper ) + self.gamma_k @ self.r - self.bias_k - self.beta_k * epsilon_alo
            # violation = 0
            violation = cp.maximum( *([cp.pnorm(self.x_upper[group_ind[i]:group_ind[i+1]], 2) - self.r[i] for i in range(M)] + [cp.pnorm(self.x_upper, 1) - self.r[M]]) )
            self.beta_k_penalty = cp.maximum(0, beta_k_V_k, 100*self.beta_k*violation)
            phi_k = LS_upper + rho/2 * prox + self.beta_k_penalty
            bi_constraints = [self.r >= 0]

            self.dc_approximated = cp.Problem(cp.Minimize(phi_k), bi_constraints)
        
        def solve(self, k):
            # try:
            # ECOS_TOL = 20/(k+1)
            # ECOS_ITER = 100
            # result = self.dc_approximated.solve(solver = cp.ECOS, feastol=ECOS_TOL,abstol=ECOS_TOL, reltol=ECOS_TOL, max_iters=ECOS_ITER, verbose=False)
            result = self.dc_approximated.solve(solver = cp.ECOS, verbose=False)
            # except:
                # result = self.dc_approximated.solve(solver=cp.SCS)
            return result, self.x_upper.value, self.r.value
        
        def update_beta(self, err):
            if err * self.beta_k.value <= self.c_alo * min( 1., self.beta_k_penalty.value ):
                self.beta_k.value = self.beta_k.value + self.delta
        
        def clare_V_k(self, gamma, obj_lower):
            self.gamma_k.value = gamma * self.beta_k.value
            self.bias_k.value = obj_lower * self.beta_k.value + self.gamma_k.value @ self.r_k.value 
        
        def clare_variable_k(self, x, r):
            self.x_upper_k.value = x 
            self.r_k.value = r
        
        def cal_penalty(self):
            return self.beta_k_penalty.value / self.beta_k.value

    def iteration_err(x, r, xp, rp):
        return np.sqrt(
            np.sum(np.square(x - xp)) + np.sum(np.square(r - rp))
        ) / np.sqrt(
            np.sum(np.square(x)) + np.sum(np.square(r))
        )

    # preparation
    Timer = time.time
    monitor_dc = Monitor_DC()
    time_start = Timer()

    # main part
    lower_problem = DC_lower(settings, data)
    approximated_problem = DC_approximated(settings, data, DC_Setting)

    for k in range(MAX_ITERATION):
        approximated_problem.clare_variable_k(x, r)
        obj_lower_k, x_k_tilde = lower_problem.solve(r)
        gamma = lower_problem.dual_value()
        approximated_problem.clare_V_k(gamma, obj_lower_k)
        _, x_p, r_p = approximated_problem.solve(k)
        r_p = np.maximum(r_p, 0)
        
        time_past = Timer() - time_start

        err = iteration_err(x, r, x_p, r_p)
        penalty = approximated_problem.cal_penalty()

        dic_for_monitor = {
            "time": time_past, 
            "train_error": train_error(settings, data, x_p),
            "validation_error": validation_error(settings, data, x_p),
            "test_error": test_error(settings, data, x_p),
            "lower_train_error": train_error(settings, data, x_k_tilde),
            "lower_validation_error": validation_error(settings, data, x_k_tilde), 
            "lower_test_error": test_error(settings, data, x_k_tilde),
            "diff_xk_xtilde": np.linalg.norm(x - x_k_tilde),
            "diff_xkp_xtilde": np.linalg.norm(x_p - x_k_tilde),
            "beta": approximated_problem.beta_k.value,
            "step_err": err,
            "penalty": penalty
        }

        monitor_dc.append(dic_for_monitor)

        # Stopping Test
        if err < TOL and penalty < TOL:
            print("Pass")
            break 
        
        approximated_problem.update_beta(err)

        x, r = x_p, r_p 

    return monitor_dc.to_df()
#%%
# BCD_iP_DCA
def BCD_iP_DCA(data_info, DC_Setting = dict()):
    settings = data_info.settings
    data = data_info.data
    p, M = settings.num_features, settings.num_experiment_groups
    cal_group_sizes = [p//M] * M
    group_ind = np.concatenate( [[0], np.cumsum(cal_group_sizes)] )

    MAX_ITERATION = DC_Setting["MAX_ITERATION"] if "MAX_ITERATION" in DC_Setting.keys() else 100
    TOL = DC_Setting["TOL"] if "TOL" in DC_Setting.keys() else 5e-2
    r = DC_Setting["initial_guess"] if "initial_guess" in DC_Setting.keys() else .1*np.ones(M+1)
    x = np.zeros(settings.num_features)

    # subproblems define
    class DC_lower:
        def __init__(self, settings, data) -> None:
            self.x_lower = cp.Variable(settings.num_features)
            self.r_lower = cp.Parameter(M+1, nonneg=True)
            self.constraints_lower = [cp.pnorm(self.x_lower[group_ind[i]:group_ind[i+1]], 2) <= self.r_lower[i] for i in range(M)] + [cp.pnorm(self.x_lower, 1) <= self.r_lower[M]]
            LS_lower = .5 / settings.num_train * cp.sum_squares( data.y_train - data.X_train @ self.x_lower )
            self.dc_lower = cp.Problem(cp.Minimize(LS_lower), self.constraints_lower)

        def solve(self, r):
            ECOS_TOL = 1e-4
            ECOS_ITER = 100
            self.r_lower.value = r
            result = self.dc_lower.solve(solver = cp.ECOS, abstol=ECOS_TOL, reltol=ECOS_TOL, abstol_inacc=ECOS_TOL, reltol_inacc=ECOS_TOL, max_iters=ECOS_ITER)
            return result, self.x_lower.value

        def dual_value(self):
            return np.array([float(self.constraints_lower[i].dual_value) for i in range(M+1)])

    class DC_approximated:
        def __init__(self, settings, data, DC_Setting = dict()) -> None:
            self.delta = DC_Setting["delta"] if "delta" in DC_Setting.keys() else 5.
            self.c_alo = DC_Setting["c"] if "c" in DC_Setting.keys() else .1
            epsilon_alo = DC_Setting["epsilon"] if "epsilon" in DC_Setting.keys() else 0
            rho = DC_Setting["rho"] if "rho" in DC_Setting.keys() else 5e-3
            beta_0 = DC_Setting["beta_0"] if "beta_0" in DC_Setting.keys() else 1

            self.x_upper, self.r = cp.Variable(settings.num_features), cp.Variable(M+1)
            self.x_upper_k, self.r_k = cp.Parameter(settings.num_features), cp.Parameter(M+1, pos=True)
            self.gamma_k, self.bias_k = cp.Parameter(M+1), cp.Parameter()
            self.beta_k = cp.Parameter(pos=True)
            self.beta_k.value = beta_0
            self.constant_1 = cp.Parameter()
            self.constant_2 = cp.Parameter()
            self.vector_2 = cp.Parameter(M+1)
            self.beta_x_k = cp.Parameter(settings.num_features)
            self.beta_r_k = cp.Parameter(M+1)

            LS_upper = .5/settings.num_validate * cp.sum_squares( data.y_validate - data.X_validate @ self.x_upper )
            prox_x = cp.sum_squares(self.x_upper - self.x_upper_k)
            prox_r = cp.sum_squares(self.r - self.r_k)

            # beta_k_V_k_1 = self.beta_k * .5 / settings.num_train * cp.sum_squares( data.y_train - data.X_train @ self.x_upper ) + self.gamma_k @ self.r_k - self.bias_k - self.beta_k * epsilon_alo
            beta_k_V_k_1 = self.beta_k * .5 / settings.num_train * cp.sum_squares( data.y_train - data.X_train @ self.x_upper ) + self.constant_1
            # violation = 0
            violation_1 = cp.maximum( *([self.beta_k*cp.pnorm(self.x_upper[group_ind[i]:group_ind[i+1]], 2) - self.beta_r_k[i] for i in range(M)] + [self.beta_k*cp.pnorm(self.x_upper, 1) - self.beta_r_k[M]]) )
            self.beta_k_penalty_1 = cp.maximum(0, beta_k_V_k_1, violation_1)
            beta_k_V_k_2 = self.constant_2 + self.gamma_k @ self.r 
            violation_2 = cp.maximum( *([self.vector_2[i] - self.beta_k*self.r[i] for i in range(M)] + [self.vector_2[M] - self.beta_k*self.r[M]]) )
            self.beta_k_penalty_2 = cp.maximum(0, beta_k_V_k_2, violation_2)

            phi_k_1 = LS_upper + rho/2 * prox_x + self.beta_k_penalty_1
            phi_k_2 = rho/2 * prox_r + self.beta_k_penalty_2
            bi_constraints = [self.r >= 0]

            self.dc_approximated_1 = cp.Problem(cp.Minimize(phi_k_1))
            self.dc_approximated_2 = cp.Problem(cp.Minimize(phi_k_2), bi_constraints)
        
        def solve(self):
            self.constant_1.value = self.gamma_k.value @ self.r_k.value - self.bias_k.value
            self.beta_r_k.value = self.beta_k.value * self.r_k.value
            result = self.dc_approximated_1.solve(solver = cp.ECOS, verbose=False)
            
            self.x_upper_k.value = self.x_upper.value 
            self.constant_2.value = (self.beta_k * .5 / settings.num_train * cp.sum_squares( data.y_train - data.X_train @ self.x_upper_k ) - self.bias_k).value
            self.vector_2.value = self.beta_k.value * np.concatenate([np.array([cp.pnorm(self.x_upper_k[group_ind[i]:group_ind[i+1]], 2).value for i in range(M)]), np.array([cp.pnorm(self.x_upper, 1).value])])
            result = self.dc_approximated_2.solve(solver = cp.ECOS, verbose=False)

            return result, self.x_upper.value, self.r.value
        
        def update_beta(self, err):
            if err * self.beta_k.value <= self.c_alo * min( 1., self.beta_k_penalty_2.value ):
                self.beta_k.value = self.beta_k.value + self.delta
        
        def clare_V_k(self, gamma, obj_lower):
            self.gamma_k.value = gamma * self.beta_k.value
            self.bias_k.value = obj_lower * self.beta_k.value + self.gamma_k.value @ self.r_k.value 
        
        def clare_variable_k(self, x, r):
            self.x_upper_k.value = x 
            self.r_k.value = r
        
        def cal_penalty(self):
            return self.beta_k_penalty_2.value / self.beta_k.value

    def iteration_err(x, r, xp, rp):
        return np.sqrt(
            np.sum(np.square(x - xp)) + np.sum(np.square(r - rp))
        ) / np.sqrt(
            np.sum(np.square(x)) + np.sum(np.square(r))
        )

    # preparation
    Timer = time.time
    monitor_dc = Monitor_DC()
    time_start = Timer()

    # main part
    lower_problem = DC_lower(settings, data)
    approximated_problem = DC_approximated(settings, data, DC_Setting)

    for k in range(MAX_ITERATION):
        approximated_problem.clare_variable_k(x, r)
        obj_lower_k, x_k_tilde = lower_problem.solve(r)
        gamma = lower_problem.dual_value()
        approximated_problem.clare_V_k(gamma, obj_lower_k)
        _, x_p, r_p = approximated_problem.solve()
        r_p = np.maximum(r_p, 0)
        
        time_past = Timer() - time_start

        err = iteration_err(x, r, x_p, r_p)
        penalty = approximated_problem.cal_penalty()

        dic_for_monitor = {
            "time": time_past, 
            "train_error": train_error(settings, data, x_p),
            "validation_error": validation_error(settings, data, x_p),
            "test_error": test_error(settings, data, x_p),
            "beta": approximated_problem.beta_k.value,
            "step_err": err,
            "penalty": penalty
        }

        monitor_dc.append(dic_for_monitor)

        # Stopping Test
        if err < TOL and penalty < TOL:
            break 
        
        approximated_problem.update_beta(err)

        x, r = x_p, r_p 

    return monitor_dc.to_df()

#%%
# Grid Search
def Grid_Search(data_info):
    # preparation
    settings = data_info.settings
    data = data_info.data
    training_process = Training_model_simple(data_info)

    Timer = time.time
    monitor = Monitor()
    time_start = Timer()

    # main part
    lam1s = np.power(10, np.linspace(-3, 1, 10)) 
    lam2s = lam1s
    for lam1, lam2 in tqdm(itertools.product(lam1s, lam2s)):
        x = training_process.solve_training(np.array([lam1, lam2]))
        monitor.append({
            "time": Timer() - time_start,
            "train_error": train_error(settings, data, x), 
            "validation_error": validation_error(settings, data, x), 
            "test_error": test_error(settings, data, x)
        })

    return monitor.to_df()

# Random Search
def Random_Search(data_info):
    # preparation
    settings = data_info.settings
    data = data_info.data
    training_process = Training_model(data_info)

    Timer = time.time
    Random_Generator = np.random.rand
    monitor = Monitor()
    time_start = Timer()

    # main part
    N = 100
    for _ in tqdm(range(N)):
        # lam1, lam2 = np.power(10, -3+4*Random_Generator()), np.power(10, -3+4*Random_Generator())
        x = training_process.solve_training(np.power(10, -3+4*Random_Generator(settings.num_experiment_groups+1)))
        monitor.append({
            "time": Timer() - time_start,
            "train_error": train_error(settings, data, x), 
            "validation_error": validation_error(settings, data, x), 
            "test_error": test_error(settings, data, x)
        })

    return monitor.to_df()

# Bayesian Method
def Bayesian_Method(data_info, Debug = False):
    # define the object for bayesian method 
    settings = data_info.settings
    data = data_info.data
    M = settings.num_experiment_groups
    training_process = Training_model(data_info)

    def Bayesian_obj(param):
        nonlocal monitor
        x = training_process.solve_training(np.power(10, np.array(param)))
        val_err = validation_error(settings, data, x)
        monitor.append({
            "time": Timer() - time_start,
            "train_error": train_error(settings, data, x), 
            "validation_error": val_err, 
            "test_error": test_error(settings, data, x)
        })
        return val_err

    # preparation
    Timer = time.time
    monitor = Monitor()

    # main part
    space = [hp.uniform(str(i), -3, 1) for i in range(M+1)]
    time_start = Timer()
    Best = fmin(
        fn=Bayesian_obj, # Objective Function to optimize
        space=space, # Hyperparameter's Search Space
        algo=tpe.suggest, # Optimization algorithm (representative TPE)
        max_evals=100 # Number of optimization attempts
        )
    
    if Debug: print(Best['1'], Best[str(M)])
    
    return monitor.to_df()  

# Implicit Differentiation
def Imlicit_Differntiation(data_info, HC_Setting = dict()):
    data = data_info.data 
    settings = data_info.settings
    initial_guess = HC_Setting.pop("initial_guess") if "initial_guess" in HC_Setting.keys() else .05*np.ones(settings.num_experiment_groups + 1)
    HC_algo = SGL_Hillclimb(data, settings, HC_Setting)
    HC_algo.run([initial_guess], debug=False, log_file=None)
    return HC_algo.monitor.to_df()