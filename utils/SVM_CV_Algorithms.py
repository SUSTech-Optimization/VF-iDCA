import time # record the efficiency
import itertools
# from cvxpy.settings import ECOS
import numpy as np
import pandas as pd
import cvxpy as cp
from hyperopt import tpe, hp, fmin  # for Bayesian method

from utils.utils import Monitor, Monitor_DC

#%%
def cal_error(X, y, w, c):
    return np.sum(np.maximum( 1 - y * (X @ w - c), 0))

def cal_error_rate(X, y, w, c):
    predict = np.sign( X @ w - c )
    return .5 * np.mean(np.abs( predict - y ))

def train_error(data, w, c):
    return np.sum([
        cal_error(data.X[data.iTr[i]], data.y[data.iTr[i]], w[i], c[i] )
    for i in range(len(c))]) / np.sum([len(data.iTr[i]) for i in range(len(c))])

def validation_error(data, w, c):
    return np.sum([
        cal_error(data.X[data.iVal[i]], data.y[data.iVal[i]], w[i], c[i] )
    for i in range(len(c))]) / np.sum([len(data.iVal[i]) for i in range(len(c))])

def test_error_rate(data, w, c):
    return cal_error_rate(data.X_test, data.y_test, w, c)

def full_validation_error(data_info, gamma, w_bar):
    training_process = Training_model(data_info)
    w, c = training_process.solve_training( gamma, w_bar )
    return validation_error(data_info.data, w, c)

class Test_DC_model:
    def __init__(self, data_info) -> None:  
        data = data_info.data
        settings = data_info.settings
        
        self.w = cp.Variable(settings.num_features)
        self.c = cp.Variable()

        self.r = cp.Parameter(nonneg=True)
        self.w_bar = cp.Parameter(settings.num_features, nonneg=True)

        SVM = 1. / len(data.y) * cp.sum( cp.maximum(1 - cp.multiply(data.y, (data.X @ self.w - self.c)), 0 ) )

        SVM_cons = [0.5 * cp.sum_squares(self.w) <= self.r]
        SVM_cons += [self.w  <=  self.w_bar]
        SVM_cons += [self.w  >= -self.w_bar]
        
        self.test_problem = cp.Problem(cp.Minimize(SVM), SVM_cons)

    def solve(self, r, w_bar):
        self.r.value = r
        self.w_bar.value = w_bar
        self.test_problem.solve(solver = cp.ECOS)
        return self.w.value, self.c.value 
    
    def performance(self, data, r, w_bar):
        w, c = self.solve(r, w_bar)
        return test_error_rate(data, w,c )


class Test_model:
    def __init__(self, data_info) -> None:  
        data = data_info.data
        settings = data_info.settings
        
        self.w = cp.Variable(settings.num_features)
        self.c = cp.Variable()
        self.w_bar = cp.Parameter(settings.num_features, nonneg=True)
        self.lam = cp.Parameter(nonneg=True)

        SVM = 1. / len(data.y) * cp.sum( cp.maximum(1 - cp.multiply(data.y, (data.X @ self.w - self.c)), 0 ) ) + 0.5 * self.lam * cp.sum_squares(self.w)

        SVM_cons = []
        SVM_cons += [self.w  <=  self.w_bar]
        SVM_cons += [self.w  >= -self.w_bar]
        
        self.test_problem = cp.Problem(cp.Minimize(SVM), SVM_cons)

    def solve(self, lam, w_bar):
        self.lam.value = lam
        self.w_bar.value = w_bar
        try:
            self.test_problem.solve(solver = cp.ECOS)
        except:
            self.test_problem.solve(solver = cp.SCS)
        return self.w.value, self.c.value 
    
    def performance(self, data, lam, w_bar):
        w, c = self.solve(lam, w_bar)
        return test_error_rate(data, w,c )

class Training_model:
    def __init__(self, data_info) -> None:  
        data = data_info.data
        settings = data_info.settings
        
        self.w = cp.Variable([settings.num_CV, settings.num_features])
        self.c = cp.Variable(settings.num_CV)
        self.w_bar = cp.Parameter(settings.num_features, nonneg=True)
        self.lam = cp.Parameter(nonneg=True)

        SVMs = 1. / settings.num_train * cp.sum([ cp.sum( cp.maximum(1 - cp.multiply(data.y[data.iTr[i]], (data.X[data.iTr[i]] @ self.w[i] - self.c[i])), 0 ) ) + 0.5 * self.lam * cp.sum_squares(self.w[i]) for i in range(settings.num_CV)])

        SVM_cons = []
        SVM_cons += [self.w[i] <= self.w_bar for i in range(settings.num_CV)]
        SVM_cons += [self.w[i] >= -self.w_bar for i in range(settings.num_CV)]
        
        self.training_problem = cp.Problem(cp.Minimize(SVMs), SVM_cons)

    def solve_training(self, lam, w_bar):
        self.lam.value = lam
        self.w_bar.value = w_bar
        try:
            self.training_problem.solve(cp.ECOS)
        except:
            self.training_problem.solve(cp.SCS)
        return self.w.value, self.c.value 


#%%
def VF_iDCA(data_info, DC_Setting = dict(), DEBUG = False):
    if DEBUG: print("DCA Debuging")
    data = data_info.data
    settings = data_info.settings
    tester1 = Test_DC_model(data_info)
    # tester2 = Test_model(data_info)

    MAX_ITERATION = DC_Setting["MAX_ITERATION"] if "MAX_ITERATION" in DC_Setting.keys() else 100
    TOL = DC_Setting["TOL"] if "TOL" in DC_Setting.keys() else 5e-2
    w_bar = DC_Setting["initial_w_bar"] if "initial_w_bar" in DC_Setting.keys() else .1 * np.ones(settings.num_features)
    r = DC_Setting["initial_r"] if "initial_r" in DC_Setting.keys() else .1
    w = np.zeros([settings.num_CV, settings.num_features])
    c = np.zeros(settings.num_CV)

    # subproblems define
    class DC_lower:
        def __init__(self, settings, data) -> None:
            self.w = cp.Variable([settings.num_CV, settings.num_features])
            self.c = cp.Variable(settings.num_CV)
            self.w_bar = cp.Parameter(settings.num_features, nonneg=True)
            self.r = cp.Parameter(nonneg=True)

            SVM_lower = 1. / settings.num_train * cp.sum([ cp.sum( cp.maximum(1 - cp.multiply(data.y[data.iTr[i]], (data.X[data.iTr[i]] @ self.w[i] - self.c[i])), 0 ) ) for i in range(settings.num_CV)]) # 1
            # SVM_lower = cp.sum([ cp.sum( cp.maximum(1 - cp.multiply(data.y[data.iTr[i]], (data.X[data.iTr[i]] @ self.w[i] - self.c[i])), 0 ) ) for i in range(settings.num_CV)])

            self.SVM_constraints = [
                # 0.5 * cp.sum_squares(self.w[i]) <= self.r for i in range(settings.num_CV)
                0.5 * cp.sum_squares(self.w) <= self.r
            ] + [self.w[i] <=  self.w_bar for i in range(settings.num_CV)
            ] + [self.w[i] >= -self.w_bar for i in range(settings.num_CV)]
            
            self.training_problem = cp.Problem(cp.Minimize(SVM_lower), self.SVM_constraints)

        def solve(self, r, w_bar):
            self.r.value = r
            self.w_bar.value = w_bar
            try:
                result = self.training_problem.solve(solver = cp.ECOS)
            except:
                result = self.training_problem.solve(solver = cp.SCS)
            return result, self.w.value, self.c.value 

        def dual_value(self):
            return float(self.SVM_constraints[0].dual_value), np.sum(np.array([
                self.SVM_constraints[1 + i].dual_value for i in range(settings.num_CV)
            ]), axis=0)

    class DC_approximated:
        def __init__(self, settings, data) -> None:
            self.delta = DC_Setting["delta"] if "delta" in DC_Setting.keys() else 5.
            self.c_alo = DC_Setting["c"] if "c" in DC_Setting.keys() else 1.
            epsilon_alo = DC_Setting["epsilon"] if "epsilon" in DC_Setting.keys() else 0
            rho = DC_Setting["rho"] if "rho" in DC_Setting.keys() else 5e-2
            beta_0 = DC_Setting["beta_0"] if "beta_0" in DC_Setting.keys() else 1

            self.w = cp.Variable([settings.num_CV, settings.num_features])
            self.c = cp.Variable(settings.num_CV)
            self.r = cp.Variable()
            self.w_bar = cp.Variable(settings.num_features)
            
            self.w_k = cp.Parameter([settings.num_CV, settings.num_features])
            self.c_k = cp.Parameter(settings.num_CV,)
            self.r_k = cp.Parameter(nonneg=True)
            self.w_bar_k = cp.Parameter(settings.num_features, nonneg=True)
            
            self.xi_wb = cp.Parameter(settings.num_features)
            self.xi_r = cp.Parameter()
            self.bias_k = cp.Parameter()

            self.beta_k = cp.Parameter(pos=True)
            self.beta_k.value = beta_0

            LS_upper = 1./ settings.num_validate * cp.sum([
                cp.sum( cp.maximum(1 - cp.multiply(data.y[data.iVal[i]], (data.X[data.iVal[i]] @ self.w[i] - self.c[i])), 0 ) ) 
                for i in range(settings.num_CV)])
            prox = cp.sum_squares(self.w - self.w_k) + cp.sum_squares(self.c - self.c_k) + cp.sum_squares(self.r - self.r_k) + cp.sum_squares(self.w_bar - self.w_bar_k)
            beta_k_V_k = self.beta_k * 1. / settings.num_train * cp.sum([
                cp.sum( cp.maximum(1 - cp.multiply(data.y[data.iTr[i]], (data.X[data.iTr[i]] @ self.w[i] - self.c[i])), 0 ) ) 
                for i in range(settings.num_CV)]) - self.xi_wb @ self.w_bar + self.xi_r * self.r - self.bias_k - self.beta_k * epsilon_alo

            violation = cp.maximum( *([
                0.5 * cp.sum_squares(self.w[i]) - self.r for i in range(settings.num_CV)
            ] + [
               cp.max(self.w[i] - self.w_bar) for i in range(settings.num_CV)
            ] + [
                cp.max(- self.w[i] - self.w_bar) for i in range(settings.num_CV)
            ]) )

            self.beta_k_penalty = cp.maximum(0, beta_k_V_k, self.beta_k*violation)
            phi_k = LS_upper + rho/2 * prox + self.beta_k_penalty
            bi_constraints = [self.r >= 0] + [self.w_bar >= 1e-6] + [self.w_bar <= 10]

            self.dc_approximated = cp.Problem(cp.Minimize(phi_k), bi_constraints)
        
        def solve(self, DEBUG = False):
            try:
                result = self.dc_approximated.solve(solver = cp.ECOS)
            except:
                result = self.dc_approximated.solve(solver = cp.SCS)
            if DEBUG:
                print("violations of lower constraints are: (%.2f, %.2f, %.2f) " % (
                    np.max([
                0.5 * cp.sum_squares(self.w[i]).value - self.r.value for i in range(settings.num_CV)
            ]), np.max([
                np.max(self.w.value[i] - self.w_bar.value) for i in range(settings.num_CV)
            ]), np.max([
                np.max(self.w.value[i] + self.w_bar.value) for i in range(settings.num_CV)
            ])) + "lower object penaltyis %.2f" % (
                    self.beta_k_penalty.value / self.beta_k.value))
            # except:
                # result = self.dc_approximated.solve(solver=cp.SCS)
            return result, self.w.value, self.c.value, self.r.value, self.w_bar.value
        
        def update_beta(self, err):
            if err * self.beta_k.value <= self.c_alo * min( 1., self.beta_k_penalty.value ):
                self.beta_k.value = self.beta_k.value + self.delta
                # self.beta_k.value = self.beta_k.value
        
        def clare_V_k(self, gamma_r, gamma_wb, obj_lower):
            self.xi_r.value = gamma_r * self.beta_k.value
            self.xi_wb.value = (- gamma_wb ) * self.beta_k.value 
            self.bias_k.value = obj_lower * self.beta_k.value - self.xi_wb.value @ self.w_bar_k.value + self.xi_r.value * self.r_k.value 
        
        def clare_variable_k(self, w, c, r, w_bar):
            self.w_k.value = w 
            self.c_k.value = c
            self.r_k.value = r
            self.w_bar_k.value = w_bar
        
        def cal_penalty(self):
            return self.beta_k_penalty.value / self.beta_k.value

    def iteration_err(w, c, r, w_bar, wp, cp, rp, w_bar_p):
        return np.sqrt(
            np.sum(np.square(w - wp)) + np.sum(np.square(c - cp)) 
            + np.sum(np.square(r - rp)) + np.sum(np.square(w_bar - w_bar_p)) 
        ) / np.sqrt( 1 +
            np.sum(np.square(w)) + np.sum(np.square(c)) 
            + np.sum(np.square(r)) + np.sum(np.square(w_bar))
        )

    # preparation
    Timer = time.time
    monitor_dc = Monitor_DC()
    time_start = Timer()
    # main part
    lower_problem = DC_lower(settings, data)
    approximated_problem = DC_approximated(settings, data)


    dic_for_monitor = {
        "time": 0, 
        "validation_error": validation_error(data, w, c),
        "test_error_rate": tester1.performance(data, r, w_bar),
        "beta": approximated_problem.beta_k.value,
    }

    monitor_dc.append(dic_for_monitor)

    for k in range(MAX_ITERATION):
        if DEBUG: print("%2d-th iteration" % (k+1))
        approximated_problem.clare_variable_k(w, c, r, w_bar)
        obj_lower_k, w_tilde, c_tilde = lower_problem.solve(r, w_bar)
        gamma_r, gamma_wb = lower_problem.dual_value()
        approximated_problem.clare_V_k(gamma_r, gamma_wb, obj_lower_k)
        obj_upper, w_p, c_p, r_p, w_bar_p = approximated_problem.solve(DEBUG=DEBUG)
        r_p = np.maximum(r_p, 0)
        w_bar_p = np.maximum(w_bar_p, 0)
        
        time_past = Timer() - time_start

        err = iteration_err(w, c, r, w_bar, w_p, c_p, r_p, w_bar_p)
        penalty = approximated_problem.cal_penalty()

        dic_for_monitor = {
            "time": time_past, 
            "validation_error": validation_error(data, w_p, c_p),
            "test_error_rate": tester1.performance(data, r_p, w_bar_p),
            "beta": approximated_problem.beta_k.value,
            "step_err": err,
            "penalty": penalty
        }

        monitor_dc.append(dic_for_monitor)

        if DEBUG: print(
            "upper: %.2e, " %  (dic_for_monitor["validation_error"]) + 
            "err: %.2e, penalty: %.2e\n" % (err, penalty) + 
            "r: %.2e, lambda: %.2e \nw_bar: %s, range: (%.2e, %.2e)\n" % (r, gamma_r, w_bar_p[0:4], np.min(w_bar_p), np.max(w_bar_p)) + 
            "w:%s, range: (%.2e, %.2e)\n" % (w_p[1, 0:4], np.min(w_p), np.max(w_p)) + 
            "w_l:%s, range: (%.2e, %.2e)" % (w_tilde[1, 0:4], np.min(w_tilde), np.max(w_tilde)))

        # Stopping Test
        if err < TOL and penalty < TOL:
            # print("Pass with (%.2e, %.2e)" % (err, penalty))
            break 
        
        approximated_problem.update_beta(err)

        w, c, r, w_bar = w_p, c_p, r_p, w_bar_p

    return monitor_dc.to_df()

#%%
# Grid Search
def Grid_Search(data_info, DEBUG=False):
    if DEBUG: print("Grid Search Debuging")
    # preparation
    data = data_info.data
    settings = data_info.settings
    training_process = Training_model(data_info)
    tester = Test_model(data_info)

    Timer = time.time
    monitor = Monitor()
    time_start = Timer()

    # main part
    lam1s = np.power(10, np.linspace(-4, 4, 10)) 
    lam2s = np.power(10, np.linspace(-6, 1, 10)) 
    if DEBUG: BEST_OBJ = np.inf
    for lam1, lam2 in itertools.product(lam1s, lam2s):
        w, c = training_process.solve_training( lam1, lam2*np.ones(settings.num_features) )
        monitor.append({
            "time": Timer() - time_start,
            # "train_error": train_error(data, w, c),
            "validation_error": validation_error(data, w, c), 
            "test_error_rate": tester.performance(data, lam1, lam2*np.ones(settings.num_features))
        })

        if DEBUG and BEST_OBJ >  validation_error(data, w, c):
            BEST_OBJ =  validation_error(data, w, c)
    
    return monitor.to_df()

# Random Search
def Random_Search(data_info, DEBUG=False):
    if DEBUG: print("Random Search Debuging")
    # preparation
    data = data_info.data
    settings = data_info.settings
    training_process = Training_model(data_info)
    tester = Test_model(data_info)

    Timer = time.time
    Random_Generator = np.random.rand
    monitor = Monitor()
    time_start = Timer()

    # main part
    N = 100
    if DEBUG: BEST_OBJ = np.inf
    for _ in range(N):
        lam1, lam2 = np.power(10, -4+8*Random_Generator()), np.power(10, -6+7*Random_Generator(settings.num_features))
        w, c = training_process.solve_training( lam1, lam2 )
        monitor.append({
            "time": Timer() - time_start,
            # "train_error": train_error(data, w, c),
            "validation_error": validation_error(data, w, c), 
            "test_error_rate": tester.performance(data, lam1, lam2*np.ones(settings.num_features))
        })

        if DEBUG and BEST_OBJ > validation_error(data, w, c):
            BEST_OBJ = validation_error(data, w, c)
            print("obj:%.2e lambda: (%.2e, %.2e)" % (BEST_OBJ, lam1, lam2))
            print("w: %s range: %.2e - %.2e " % (w[0:4], min(w), max(w)) )

    return monitor.to_df()

# Bayesian Method
def Bayesian_Method(data_info, TPE_Setting = dict(), DEBUG = False):
    if DEBUG: print("Bayesian Method Debuging")
    # define the object for bayesian method 
    data = data_info.data
    settings = data_info.settings
    training_process = Training_model(data_info)
    tester = Test_model(data_info)

    def Bayesian_obj(param):
        nonlocal monitor
        lam = np.power(10, np.array(param[0]))
        # w_bar = np.power(10, np.array(param[1])) * np.ones(settings.num_features)
        w_bar = np.power(10, np.array(param[1:]))
        w, c = training_process.solve_training(lam, w_bar)
        val_err = validation_error(data, w, c)
        monitor.append({
            "time": Timer() - time_start,
            # "train_error": train_error(data, w, c),
            "validation_error": val_err, 
            "test_error_rate": tester.performance(data, lam, w_bar)
        })
        return val_err

    # preparation
    Timer = time.time
    monitor = Monitor()

    # main part
    # space = [hp.uniform(str(i), -5, 2) for i in range(2)]
    space = [hp.uniform(str(0), -4, 4)] + [
        hp.uniform(str(i+1), -6, 7) for i in range(settings.num_features)]
    time_start = Timer()
    Best = fmin(
        fn=Bayesian_obj, # Objective Function to optimize
        space=space, # Hyperparameter's Search Space
        algo=tpe.suggest, # Optimization algorithm (representative TPE)
        max_evals= TPE_Setting["max_evals"] if "max_evals" in TPE_Setting.keys() else 100 # Number of optimization attempts
        )

    if DEBUG: 
        print("lambda: (%.2e, %.2e)" % (10**Best["lam1"], 10**Best["lam2"]))
    
    return monitor.to_df()

def Bayesian_Method_Simple(data_info, TPE_Setting = dict(), DEBUG = False):
    if DEBUG: print("Bayesian Method Debuging")
    # define the object for bayesian method 
    data = data_info.data
    settings = data_info.settings
    training_process = Training_model(data_info)
    tester = Test_model(data_info)

    def Bayesian_obj(param):
        nonlocal monitor
        lam = np.power(10, np.array(param[0]))
        w_bar = np.power(10, np.array(param[1])) * np.ones(settings.num_features)
        # w_bar = np.power(10, np.array(param[1:]))
        w, c = training_process.solve_training(lam, w_bar)
        val_err = validation_error(data, w, c)
        monitor.append({
            "time": Timer() - time_start,
            # "train_error": train_error(data, w, c),
            "validation_error": val_err, 
            "test_error_rate": tester.performance(data, lam, w_bar)
        })
        return val_err

    # preparation
    Timer = time.time
    monitor = Monitor()

    # main part
    # space = [hp.uniform(str(i), -5, 2) for i in range(2)]
    space = [hp.uniform(str(0), -4, 4)] + [
        hp.uniform(str(1), -6, 7)]
    time_start = Timer()
    Best = fmin(
        fn=Bayesian_obj, # Objective Function to optimize
        space=space, # Hyperparameter's Search Space
        algo=tpe.suggest, # Optimization algorithm (representative TPE)
        max_evals= TPE_Setting["max_evals"] if "max_evals" in TPE_Setting.keys() else 100 # Number of optimization attempts
        )

    if DEBUG: 
        print("lambda: (%.2e, %.2e)" % (10**Best["lam1"], 10**Best["lam2"]))
    
    return monitor.to_df()
