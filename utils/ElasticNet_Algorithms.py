#%%
import time # record the efficiency
import itertools
from cvxpy.settings import ECOS
import numpy as np
import pandas as pd
import cvxpy as cp
from hyperopt import tpe, hp, fmin  # for Bayesian method

from HC_ElasticNet import Elastic_Net_Hillclimb

# from sklearn.preprocessing import normalize
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
        data = data_info.data
        settings = data_info.settings
        self.x = cp.Variable(settings.num_features)
        self.lam = cp.Parameter(2, nonneg=True)
        LS_lower = .5 * cp.sum_squares( data.y_train - data.X_train @ self.x ) + self.lam[0] * cp.norm(self.x, 1) + 0.5 * self.lam[1] * cp.sum_squares(self.x)
        self.training_problem = cp.Problem(cp.Minimize(LS_lower))

    def solve_training(self, lam):
        self.lam.value = lam
        self.training_problem.solve(solver = cp.ECOS)
        return self.x.value

#%%
def iP_DCA(data_info, DC_Setting = dict(), DEBUG = False):
    if DEBUG: print("DCA Debuging")
    data = data_info.data
    settings = data_info.settings

    MAX_ITERATION = DC_Setting["MAX_ITERATION"] if "MAX_ITERATION" in DC_Setting.keys() else 100
    TOL = DC_Setting["TOL"] if "TOL" in DC_Setting.keys() else 5e-2
    r = DC_Setting["initial_guess"] if "initial_guess" in DC_Setting.keys() else np.array([.1, .5])
    x = np.zeros(settings.num_features)

    # subproblems define
    class DC_lower:
        def __init__(self, settings, data) -> None:
            self.x_lower = cp.Variable(settings.num_features)
            self.r_lower = cp.Parameter(2, nonneg=True)
            self.constraints_lower = [cp.norm(self.x_lower, 1) <= self.r_lower[0], 0.5 * cp.sum_squares(self.x_lower) <= self.r_lower[1]]
            LS_lower = .5 * cp.sum_squares( data.y_train - data.X_train @ self.x_lower )
            self.dc_lower = cp.Problem(cp.Minimize(LS_lower), self.constraints_lower)

        def solve(self, r):
            self.r_lower.value = r
            result = self.dc_lower.solve(solver = cp.ECOS)
            return result, self.x_lower.value

        def dual_value(self):
            return np.array([float(self.constraints_lower[i].dual_value) for i in range(2)])

    class DC_approximated:
        def __init__(self, settings, data) -> None:
            self.delta = DC_Setting["delta"] if "delta" in DC_Setting.keys() else 5.
            self.c_alo = DC_Setting["c"] if "c" in DC_Setting.keys() else 1.
            epsilon_alo = DC_Setting["epsilon"] if "epsilon" in DC_Setting.keys() else 0
            rho = DC_Setting["rho"] if "rho" in DC_Setting.keys() else 5e-2
            beta_0 = DC_Setting["beta_0"] if "beta_0" in DC_Setting.keys() else 1

            self.x_upper, self.r = cp.Variable(settings.num_features), cp.Variable(2)
            self.x_upper_k, self.r_k = cp.Parameter(settings.num_features), cp.Parameter(2, nonneg=True)
            self.gamma_k, self.bias_k = cp.Parameter(2), cp.Parameter()
            self.beta_k = cp.Parameter(pos=True)
            self.beta_k.value = beta_0

            LS_upper = .5/settings.num_validate * cp.sum_squares( data.y_validate - data.X_validate @ self.x_upper )
            prox = cp.sum_squares(self.x_upper - self.x_upper_k) + cp.sum_squares(self.r - self.r_k) 
            beta_k_V_k = self.beta_k * .5 / settings.num_train * cp.sum_squares( data.y_train - data.X_train @ self.x_upper ) + self.gamma_k @ self.r - self.bias_k - self.beta_k * epsilon_alo
            violation = cp.maximum(*[cp.norm(self.x_upper, 1) - self.r[0], 0.5 * cp.sum_squares(self.x_upper) - self.r[1]])
            self.beta_k_penalty = cp.maximum(0, beta_k_V_k, self.beta_k*violation)
            phi_k = LS_upper + rho/2 * prox + self.beta_k_penalty
            bi_constraints = [self.r >= 0]

            self.dc_approximated = cp.Problem(cp.Minimize(phi_k), bi_constraints)
        
        def solve(self):
            result = self.dc_approximated.solve(solver = cp.ECOS)
            return result, self.x_upper.value, np.maximum(0, self.r.value)
        
        def update_beta(self, err):
            if err * self.beta_k.value <= self.c_alo * min( 1., self.beta_k_penalty.value ):
                self.beta_k.value = self.beta_k.value + self.delta
        
        def clare_V_k(self, gamma, obj_lower):
            self.gamma_k.value = gamma * self.beta_k.value / settings.num_train
            self.bias_k.value = obj_lower / settings.num_train * self.beta_k.value + self.gamma_k.value @ self.r_k.value 
        
        def clare_variable_k(self, x, r):
            self.x_upper_k.value = x 
            self.r_k.value = r
        
        def cal_penalty(self):
            return self.beta_k_penalty.value / self.beta_k.value

    def iteration_err(x, r, xp, rp):
        return np.sqrt(
            np.sum(np.square(x - xp)) + np.sum(np.square(r - rp))
        ) / np.sqrt(
            1 + np.sum(np.square(x)) + np.sum(np.square(r))
        )

    # preparation
    Timer = time.time
    monitor_dc = Monitor_DC()
    time_start = Timer()

    # main part
    lower_problem = DC_lower(settings, data)
    approximated_problem = DC_approximated(settings, data)

    for _ in range(MAX_ITERATION):
        approximated_problem.clare_variable_k(x, r)
        obj_lower_k, x_k_tilde = lower_problem.solve(r)
        gamma = lower_problem.dual_value()
        approximated_problem.clare_V_k(gamma, obj_lower_k)
        _, x_p, r_p = approximated_problem.solve()
        
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
            # print("Pass")
            break 
        
        approximated_problem.update_beta(err)

        x, r = x_p, r_p 

    return monitor_dc.to_df()

#%%
# Grid Search
def Grid_Search(data_info, DEBUG=False):
    if DEBUG: print("Grid Search Debuging")
    # preparation
    data = data_info.data
    settings = data_info.settings
    training_process = Training_model(data_info)

    Timer = time.time
    monitor = Monitor()
    time_start = Timer()

    # main part
    lam1s = np.power(10, np.linspace(-9, -2, 10)) 
    lam2s = lam1s
    if DEBUG: BEST_OBJ = np.inf
    for lam1, lam2 in itertools.product(lam1s, lam2s):
        x = training_process.solve_training(np.array([lam1, lam2]))
        monitor.append({
            "time": Timer() - time_start,
            "train_error": train_error(settings, data, x), 
            "validation_error": validation_error(settings, data, x), 
            "test_error": test_error(settings, data, x)
        })

        if DEBUG and BEST_OBJ > validation_error(settings, data, x):
            BEST_OBJ = validation_error(settings, data, x)
            print("obj:%.2e lambda: (%.2e, %.2e)" % (BEST_OBJ, lam1, lam2))
    
    return monitor.to_df()

# Random Search
def Random_Search(data_info, DEBUG=False):
    if DEBUG: print("Random Search Debuging")
    # preparation
    data = data_info.data
    settings = data_info.settings
    training_process = Training_model(data_info)

    Timer = time.time
    Random_Generator = np.random.rand
    monitor = Monitor()
    time_start = Timer()

    # main part
    N = 100
    if DEBUG: BEST_OBJ = np.inf
    for _ in range(N):
        lam1, lam2 = np.power(10, -9+7*Random_Generator()), np.power(10, -9+7*Random_Generator())
        x = training_process.solve_training(np.array([lam1, lam2]))
        monitor.append({
            "time": Timer() - time_start,
            "train_error": train_error(settings, data, x), 
            "validation_error": validation_error(settings, data, x), 
            "test_error": test_error(settings, data, x)
        })

        if DEBUG and BEST_OBJ > validation_error(settings, data, x):
            BEST_OBJ = validation_error(settings, data, x)
            print("obj:%.2e lambda: (%.2e, %.2e)" % (BEST_OBJ, lam1, lam2))

    return monitor.to_df()

# Bayesian Method
def Bayesian_Method(data_info, DEBUG = False):
    if DEBUG: print("Bayesian Method Debuging")
    # define the object for bayesian method 
    data = data_info.data
    settings = data_info.settings
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
    space = [hp.uniform("lam1", -9, -2),
        hp.uniform("lam2", -9, -2)]
    time_start = Timer()
    Best = fmin(
        fn=Bayesian_obj, # Objective Function to optimize
        space=space, # Hyperparameter's Search Space
        algo=tpe.suggest, # Optimization algorithm (representative TPE)
        max_evals=100 # Number of optimization attempts
        )

    if DEBUG: 
        print("lambda: (%.2e, %.2e)" % (10**Best["lam1"], 10**Best["lam2"]))
    
    return monitor.to_df()

# Implicit Differentiation: IGJO
def IGJO(data_info, HC_Setting = dict()):
    data = data_info.data 
    initial_guess = HC_Setting.pop("initial_guess") if "initial_guess" in HC_Setting.keys() else .1*np.ones(2)
    HC_algo = Elastic_Net_Hillclimb(data)
    HC_algo.run([initial_guess], debug=False, log_file=None)
    return HC_algo.monitor.to_df()
    
# %%
# from sklearn import linear_model

# from sparse_ho import ImplicitForward
# from sparse_ho.criterion import HeldOutMSE
# from sparse_ho.models import ElasticNet
# from sparse_ho.ho import grad_search
# from sparse_ho.optimizers import GradientDescent

# class Monitor_IFDM():
#     """
#     Class used to store computed metrics at each iteration of the outer loop.
#     """
#     def __init__(self, callback=None):
#         self.t0 = time.time()
#         self.objs = []   # TODO rename, use self.value_outer?
#         self.times = []
#         self.alphas = []
#         self.grads = []
#         self.callback = callback
#         self.acc_vals = []
#         self.acc_tests = []
#         self.all_betas = []

#     def __call__(
#             self, obj, grad, mask=None, dense=None, alpha=None,
#             acc_val=None, acc_test=None):
#         self.objs.append(obj)
#         try:
#             self.alphas.append(alpha.copy())
#         except Exception:
#             self.alphas.append(alpha)
#         self.times.append(time.time() - self.t0)
#         self.grads.append(grad)
#         if self.callback is not None:
#             self.callback(obj, grad, mask, dense, alpha)
#         if acc_val is not None:
#             self.acc_vals.append(acc_val)
#         if acc_test is not None:
#             self.acc_tests.append(acc_test)


# def IFDM(data_info, IF_Setting = dict()):
#     max_iter = 10000
#     tol = IF_Setting.pop("tol") if "tol" in IF_Setting.keys() else 1e-5
#     data = data_info.data
#     X = np.vstack([data.X_train, data.X_validate, data.X_test])
#     y = np.concatenate([data.y_train, data.y_validate, data.y_test])
#     idx_train = np.arange(len(data.y_train))
#     idx_val = np.arange(len(data.y_train), len(data.y_train) + len(data.y_test))
#     estimator = linear_model.ElasticNet(
#         fit_intercept=False, max_iter=max_iter, warm_start=True)
#     # print("Started grad-search")
#     t_grad_search = - time.time()
#     monitor = Monitor_IFDM()
#     n_outer = IF_Setting.pop("n_outer") if "n_outer" in IF_Setting.keys() else 100
#     if "alpha0" in IF_Setting.keys():
#         alpha0 = IF_Setting.pop("alpha0")
#     else:
#         alpha_max = np.max(np.abs(data.X_train.T @ data.y_train)) / len(data.y_train)
#         alpha0 = np.array([alpha_max / 100, alpha_max / 100])
    
#     tol_jac = IF_Setting.pop("tol_jac") if "tol_jac" in IF_Setting.keys() else 1e-3
#     n_iter_jac = IF_Setting.pop("n_iter_jac") if "n_iter_jac" in IF_Setting.keys() else  100
    
#     model = ElasticNet(estimator=estimator)
#     criterion = HeldOutMSE(idx_train, idx_val)
#     algo = ImplicitForward(tol_jac=tol_jac, n_iter_jac=n_iter_jac, max_iter=max_iter)
#     optimizer = GradientDescent(
#         n_outer=n_outer, tol=tol, p_grad_norm=1.5, verbose=False)
#     grad_search(
#         algo, criterion, model, optimizer, X, y, alpha0=alpha0,
#         monitor=monitor)
#     t_grad_search += time.time()
#     monitor.alphas = np.array(monitor.alphas)
    
#     df = pd.DataFrame({
#         "time": monitor.times, 
#         "validation_error": np.array(monitor.objs) / 2,
#         "test_error": np.array(monitor.acc_tests) / 2
#     })
#     return df 