import time # record the efficiency
import itertools
import numpy as np
import pandas as pd
import cvxpy as cp
from hyperopt import tpe, hp, fmin  # for Bayesian method

from utils.utils import Monitor, Monitor_DC

#%%
def train_error(data, alpha, beta, Gamma):
    M = data.observed_matrix
    X = data.row_features
    Z = data.col_features
    n, p = X.shape
    train_idx = data.train_idx
    LS_tmp = M - X @ alpha @ np.ones([1, n]) - (Z @ beta @ np.ones([1, n])).T - Gamma
    LS_tmp = np.reshape(LS_tmp, (LS_tmp.size, 1), "F")
    return .5 / train_idx.size * np.sum(np.square( LS_tmp[train_idx] ))

def validation_error(data, alpha, beta, Gamma):
    M = data.observed_matrix
    X = data.row_features
    Z = data.col_features
    n, p = X.shape
    validate_idx = data.validate_idx
    LS_tmp = M - X @ alpha @ np.ones([1, n]) - (Z @ beta @ np.ones([1, n])).T - Gamma
    LS_tmp = np.reshape(LS_tmp, (LS_tmp.size, 1), "F")
    return .5 / validate_idx.size * np.sum(np.square( LS_tmp[validate_idx] ))

def test_error(data, alpha, beta, Gamma):
    M = data.observed_matrix
    X = data.row_features
    Z = data.col_features
    n, p = X.shape
    test_idx = data.test_idx
    LS_tmp = M - X @ alpha @ np.ones([1, n]) - (Z @ beta @ np.ones([1, n])).T - Gamma
    LS_tmp = np.reshape(LS_tmp, (LS_tmp.size, 1), "F")
    return .5 / test_idx.size * np.sum(np.square( LS_tmp[test_idx] ))

class Training_model:
    def __init__(self, data) -> None:  
        M = data.observed_matrix
        X = data.row_features
        Z = data.col_features
        G1 = data.num_alphas
        G2 = data.num_betas
        n, p = X.shape
        cal_group_alpha = [p//G1] * G1
        group_alpha_ind = np.concatenate( [[0], np.cumsum(cal_group_alpha)] )
        cal_group_beta = [p//G2] * G2
        group_beta_ind = np.concatenate( [[0], np.cumsum(cal_group_beta)] )

        self.alpha = cp.Variable([p, 1])
        self.beta = cp.Variable([p, 1])
        self.Gamma = cp.Variable([n, n])
        self.lam = cp.Parameter(1 + G1 + G2, pos=True)

        LS_tmp = M - X @ self.alpha @ np.ones([1, n]) - (Z @ self.beta @ np.ones([1, n])).T - self.Gamma
        self.training_problem = cp.Problem(cp.Minimize(
            .5 / data.train_idx.size * cp.sum_squares( cp.vec(LS_tmp)[data.train_idx] ) 
            + self.lam[0] * cp.norm(self.Gamma, "nuc") 
            + cp.sum([ self.lam[1+g] * cp.norm(self.alpha[group_alpha_ind[g]:group_alpha_ind[g+1]])  for g in range(G1)])
            + cp.sum([ self.lam[1+G1+g] * cp.norm(self.beta[group_beta_ind[g]:group_beta_ind[g+1]])  for g in range(G2)])
        ))

    def solve_training(self, lam):
        self.lam.value = lam
        self.training_problem.solve(solver = cp.SCS, eps = 1e-2)
        return self.alpha.value, self.beta.value, self.Gamma.value

#%%
def Grid_Search(data, GS_Setting = dict(), DEBUG=False):
    if DEBUG: print("Grid Search Debuging")
    # preparation
    training_process = Training_model(data)

    Timer = time.time
    monitor = Monitor()
    time_start = Timer()

    # main part
    n_grid = GS_Setting["n_grid"] if "n_grid" in GS_Setting.keys() else 10
    lam1s = np.power(10, np.linspace(-3.5, -1, n_grid)) 
    lam2s = lam1s
    if DEBUG: BEST_OBJ = np.inf
    for lam1, lam2 in itertools.product(lam1s, lam2s):
        lam = lam2 * np.ones(1 + data.num_alphas + data.num_betas)
        lam[0] = lam1
        alpha, beta, Gamma = training_process.solve_training(lam)
        monitor.append({
            "time": Timer() - time_start,
            "train_error": train_error(data, alpha, beta, Gamma), 
            "validation_error": validation_error(data, alpha, beta, Gamma), 
            "test_error": test_error(data, alpha, beta, Gamma)
        })

        if DEBUG and BEST_OBJ > validation_error(data, alpha, beta, Gamma):
            BEST_OBJ = validation_error(data, alpha, beta, Gamma)
            print("obj:%.2e lambda: (%.2e, %.2e)" % (BEST_OBJ, lam1, lam2))

    return monitor.to_df()

#%%
def Random_Search(data, RS_Setting = dict(), DEBUG=False):
    if DEBUG: print("Random Search Debuging")
    # preparation
    training_process = Training_model(data)

    Timer = time.time
    monitor = Monitor()
    time_start = Timer()

    # main part
    num_search = RS_Setting["num_search"] if "num_search" in RS_Setting.keys() else 100
    if DEBUG: BEST_OBJ = np.inf
    for _ in range(num_search):
        lam = np.power(10, -3.5 + 2.5 * np.random.rand(1 + data.num_alphas + data.num_betas))
        alpha, beta, Gamma = training_process.solve_training(lam)
        monitor.append({
            "time": Timer() - time_start,
            "train_error": train_error(data, alpha, beta, Gamma), 
            "validation_error": validation_error(data, alpha, beta, Gamma), 
            "test_error": test_error(data, alpha, beta, Gamma)
        })

        if DEBUG and BEST_OBJ > validation_error(data, alpha, beta, Gamma):
            BEST_OBJ = validation_error(data, alpha, beta, Gamma)
            print("obj:%.2e lambda: (%.2e, %.2e)" % (BEST_OBJ, lam[0], lam[1]))
            lam_best = lam

    if DEBUG: print("%s" % lam_best)

    return monitor.to_df()

#%%
# Bayesian Method
def Bayesian_Method(data, TPE_Setting = dict(), DEBUG = False):
    if DEBUG: print("Bayesian Method Debuging")
    # define the object for bayesian method 
    training_process = Training_model(data)

    def Bayesian_obj(param):
        nonlocal monitor
        alpha, beta, Gamma = training_process.solve_training(np.power(10, np.array(param)))
        val_err = validation_error(data, alpha, beta, Gamma)
        monitor.append({
            "time": Timer() - time_start,
            "train_error": train_error(data, alpha, beta, Gamma), 
            "validation_error": val_err, 
            "test_error": test_error(data, alpha, beta, Gamma)
        })
        return val_err

    # preparation
    Timer = time.time
    monitor = Monitor()

    # main part
    space = [hp.uniform("0", -3.5, -1)] + [
        hp.uniform(str(i+1), -3.5, -1) for i in range(data.num_alphas + data.num_betas)]
    time_start = Timer()
    Best = fmin(
        fn=Bayesian_obj, # Objective Function to optimize
        space=space, # Hyperparameter's Search Space
        algo=tpe.suggest, # Optimization algorithm (representative TPE)
        max_evals=TPE_Setting["max_evals"] if "max_evals" in TPE_Setting.keys() else 100 # Number of optimization attempts
        )

    if DEBUG: 
        print("lambda: (%.2e, %.2e)" % (10**Best["0"], 10**Best["1"]))
    
    return monitor.to_df()

#%%
def VF_iDCA(data, DC_Setting = dict(), DEBUG = False):
    if DEBUG: print("DCA Debuging")

    MAX_ITERATION = DC_Setting["MAX_ITERATION"] if "MAX_ITERATION" in DC_Setting.keys() else 10
    TOL = DC_Setting["TOL"] if "TOL" in DC_Setting.keys() else 5e-2

    M = data.observed_matrix
    X = data.row_features
    Z = data.col_features
    G1 = data.num_alphas
    G2 = data.num_betas
    n, p = X.shape
    train_idx = data.train_idx
    validate_idx = data.validate_idx

    cal_group_alpha = [p//G1] * G1
    group_alpha_ind = np.concatenate( [[0], np.cumsum(cal_group_alpha)] )
    cal_group_beta = [p//G2] * G2
    group_beta_ind = np.concatenate( [[0], np.cumsum(cal_group_beta)] )

    class DC_lower:
        def __init__(self) -> None:
            self.alpha = cp.Variable([p, 1])
            self.beta = cp.Variable([p, 1])
            self.Gamma = cp.Variable([n, n])
            self.r = cp.Parameter(1 + G1 + G2, pos=True)

            LS_tmp = M - X @ self.alpha @ np.ones([1, n]) - (Z @ self.beta @ np.ones([1, n])).T - self.Gamma

            self.MCG_constraints = [cp.norm(self.Gamma, "nuc") <= self.r[0]] + [
                cp.norm(self.alpha[group_alpha_ind[g]:group_alpha_ind[g+1]]) <= self.r[1+g] for g in range(G1)
            ] + [
                cp.norm(self.beta[group_beta_ind[g]:group_beta_ind[g+1]]) <= self.r[1+G1+g]  for g in range(G2)
            ]
            self.MCG_train = cp.Problem(cp.Minimize(
                .5 / train_idx.size * cp.sum_squares( cp.vec(LS_tmp)[train_idx] )),
                self.MCG_constraints
            )

        def solve(self, r, iter):
            self.r.value = r 
            result = self.MCG_train.solve(solver = cp.SCS, eps = 1e-2/(np.min([iter+1,100])), warm_start=True)
            # result = self.MCG_train.solve(solver = cp.CVXOPT, abstol = 1e-1, festol = 1e-1, retol = 1e-1)
            return result, self.alpha.value, self.beta.value, self.Gamma.value

        def dual_value(self):
            return np.array([float(cons.dual_value) for cons in self.MCG_constraints])

    class DC_approximated:
        def __init__(self, DC_Setting = dict()) -> None:
            self.delta = DC_Setting["delta"] if "delta" in DC_Setting.keys() else 5
            self.c_alo = DC_Setting["c"] if "c" in DC_Setting.keys() else .1
            epsilon_alo = DC_Setting["epsilon"] if "epsilon" in DC_Setting.keys() else 0
            rho = DC_Setting["rho"] if "rho" in DC_Setting.keys() else 5e-2
            beta_0 = DC_Setting["beta_0"] if "beta_0" in DC_Setting.keys() else 1

            self.a = cp.Variable([p, 1])
            self.b = cp.Variable([p, 1])
            self.Gamma = cp.Variable([n, n])
            self.r = cp.Variable(1 + G1 + G2)

            self.a_k = cp.Parameter([p, 1])
            self.b_k = cp.Parameter([p, 1])
            self.Gamma_k = cp.Parameter([n, n])
            self.r_k = cp.Parameter(1 + G1 + G2, nonneg=True)

            self.xi_r = cp.Parameter(1 + G1 + G2)
            self.bias_k = cp.Parameter()

            self.beta_k = cp.Parameter(pos = True)
            self.beta_k.value = beta_0 

            LS_tmp = M - X @ self.a @ np.ones([1, n]) - (Z @ self.b @ np.ones([1, n])).T - self.Gamma
            LS_tmp = cp.vec(LS_tmp)

            prox = cp.sum_squares(self.a - self.a_k) + cp.sum_squares(self.b - self.b_k) + cp.sum_squares(self.Gamma - self.Gamma_k) + cp.sum_squares(self.r - self.r_k)

            beta_k_V_k = self.beta_k * .5 / train_idx.size * cp.sum_squares(LS_tmp[train_idx]) + self.xi_r @ self.r - self.bias_k - self.beta_k * epsilon_alo

            violation = cp.maximum( *([
                cp.norm(self.Gamma, "nuc") - self.r[0]
            ] + [
                cp.norm(self.a[group_alpha_ind[g]:group_alpha_ind[g+1]]) - self.r[1+g] for g in range(G1)
            ] + [
                cp.norm(self.b[group_beta_ind[g]:group_beta_ind[g+1]]) - self.r[1+G1+g] for g in range(G2)
            ]) )

            self.beta_k_penalty = cp.maximum(0, beta_k_V_k, self.beta_k*violation)
            phi_k = .5 / validate_idx.size * cp.sum_squares(LS_tmp[validate_idx]) + rho/2 * prox + self.beta_k_penalty
            bi_constraints = [self.r >= 0]

            self.dc_approximated = cp.Problem(cp.Minimize(phi_k), bi_constraints)

        def solve(self, iter):
            result = self.dc_approximated.solve(solver = cp.SCS, eps = 1e-1/(np.min([iter+1,1000])), warm_start=True)
            return result, self.a.value, self.b.value, self.Gamma.value, self.r.value

        def update_beta(self, err):
            # if 10 * err * self.beta_k.value <= min( 1., self.beta_k_penalty.value ):
            if err * self.beta_k.value <= self.c_alo * min( 1., self.beta_k_penalty.value ):
                self.beta_k.value = self.beta_k.value + self.delta

        def clare_V_k(self, gamma_r, obj_lower):
            self.xi_r.value = gamma_r * self.beta_k.value
            self.bias_k.value = obj_lower * self.beta_k.value + self.xi_r.value @ self.r_k.value 

        def clare_variable_k(self, a, b, Gamma, r):
            self.a_k.value = a 
            self.b_k.value = b
            self.Gamma_k.value = Gamma
            self.r_k.value = r
        
        def cal_penalty(self):
            return self.beta_k_penalty.value / self.beta_k.value

    def iteration_err(a, b, Gamma, r, a_p, b_p, Gamma_p, r_p):
        return np.sqrt(
            np.sum(np.square(a - a_p)) + np.sum(np.square(b - b_p))
            + np.sum(np.square(Gamma - Gamma_p)) + np.sum(np.square(r - r_p))
        ) / np.sqrt(
            1 + np.sum(np.square(a)) + np.sum(np.square(b))
            + np.sum(np.square(Gamma)) + np.sum(np.square(r))
        )

    # preparation
    Timer = time.time
    monitor_dc = Monitor_DC()
    time_start = Timer()

    # main part
    a = np.zeros([p, 1])
    b = np.zeros([p, 1])
    Gamma = np.zeros([n, n])
    dic_for_monitor = {
        "time": 0, 
        "train_error": train_error(data, a, b, Gamma),
        "validation_error": validation_error(data, a, b, Gamma), 
        "test_error": test_error(data, a, b, Gamma),
        "beta": DC_Setting["beta_0"] if "beta_0" in DC_Setting.keys() else 1,
        "step_err": 0,
        "penalty": 0
    }
    monitor_dc.append(dic_for_monitor)
    r = DC_Setting["initial_r"] if "initial_r" in DC_Setting.keys() else np.ones(1 + G1 + G2)

    lower_problem = DC_lower()
    approximated_problem = DC_approximated(DC_Setting)

    for iter in range(MAX_ITERATION):
        time_0 = Timer()
        obj_lower_k, a_t, b_t, Gamma_t = lower_problem.solve(r, iter)
        time_1 = Timer()
        gamma_r = lower_problem.dual_value()
        # if iter == 0:
            # a, b, Gamma = a_t, b_t, Gamma_t
        approximated_problem.clare_variable_k(a, b, Gamma, r)
        approximated_problem.clare_V_k(gamma_r, obj_lower_k)
        _, a_p, b_p, Gamma_p, r_p = approximated_problem.solve(iter)
        time_2 = Timer()
        r_p = np.maximum(r_p, 0)
        
        time_past = Timer() - time_start

        err = iteration_err(a, b, Gamma, r, a_p, b_p, Gamma_p, r_p)
        penalty = approximated_problem.cal_penalty()

        dic_for_monitor = {
            "time": time_past, 
            "train_error": train_error(data, a_p, b_p, Gamma_p),
            "validation_error": validation_error(data, a_p, b_p, Gamma_p), 
            "test_error": test_error(data, a_p, b_p, Gamma_p),
            "train_error_l": train_error(data, a_t, b_t, Gamma_t),
            "validation_error_l": validation_error(data, a_t, b_t, Gamma_t), 
            "test_error_l": test_error(data, a_t, b_t, Gamma_t),
            "beta": approximated_problem.beta_k.value,
            "time_lower": time_1 - time_0,
            "time_approx": time_2 - time_1,
            "step_err": err,
            "penalty": penalty
        }

        monitor_dc.append(dic_for_monitor)

        if DEBUG: print(
            "%3d-th time: %3.1f validation error: %.1e test error: %.1e " % (iter, dic_for_monitor["time"], dic_for_monitor["validation_error"], dic_for_monitor["test_error"]) + 
            "err: %.2e, penalty: %.2e " % (err, penalty) +
            "beta: %3d" % approximated_problem.beta_k.value)

        # Stopping Test
        if err < TOL and penalty < TOL:
            print("Pass")
            break 
        
        approximated_problem.update_beta(err)

        a, b, Gamma, r = a_p, b_p, Gamma_p, r_p

    if DEBUG: 
        print("lambda: %s" % gamma_r)
        print("r: %s" % r)

    return monitor_dc.to_df()

