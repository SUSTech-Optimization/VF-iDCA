#%%
import numpy as np
import cvxpy as cp
import time
import scipy as sp
import pandas as pd
from Data_Generator import Data

#%%
class Monitor_HC():
    def __init__(self):
        self.time = []
        self.train_error = []
        self.validation_error = []
        self.test_error = []
        self.step_error = []
        self.obj_error = []
    
    def append(self, data_dic):
        for attr in self.__dict__.keys():
            self.append_one(data_dic, attr)
    
    def append_one(self, data_dic, attr):
        if attr in data_dic.keys():
            self.__dict__[attr].append(data_dic[attr])
        else:
            self.__dict__[attr].append(0)

    def to_df(self):
        return pd.DataFrame(self.__dict__)

#%%
class HC_Settings:
    def __init__(self, settings):
        self.num_train = settings.num_train
        self.num_validate = settings.num_validate
        self.num_test = settings.num_test
        self.num_features = settings.num_features
        self.num_experiment_groups = settings.num_experiment_groups
        self.num_true_groups = settings.num_true_groups
        self.snr = 2

    def get_true_group_sizes(self):
        assert(self.num_features % self.num_true_groups == 0)
        return [self.num_features // self.num_true_groups] * self.num_true_groups

    def get_expert_group_sizes(self):
        assert(self.num_features % self.num_experiment_groups == 0)
        return [self.num_features // self.num_experiment_groups] * self.num_experiment_groups

def matrix_data(data, settings):
    m_data = Data()
    m_data.X_train, m_data.X_validate, m_data.X_test = np.matrix(data.X_train), np.matrix(data.X_validate), np.matrix(data.X_test)
    m_data.y_train, m_data.y_validate, m_data.y_test = np.matrix(data.y_train).T, np.matrix(data.y_validate).T, np.matrix(data.y_test).T

    m_data.num_train = settings.num_train
    m_data.num_validate = settings.num_validate
    m_data.num_test = settings.num_test
    m_data.num_samples = m_data.num_train + m_data.num_validate + m_data.num_test

    m_data.num_experiment_groups = settings.num_experiment_groups
    return m_data

#%%
class GroupedLassoProblemWrapper:
    def __init__(self, X, y, group_feature_sizes):
        self.group_ind = np.concatenate( [[0], np.cumsum(group_feature_sizes)] )
        self.group_range = range(0, len(group_feature_sizes))
        self.betas = cp.Variable([np.sum(group_feature_sizes), 1])
        # self.betas = [cp.Variable([feature_size, 1]) for feature_size in group_feature_sizes]
        self.lambda1s = [cp.Parameter(pos=True) for i in self.group_range]
        self.lambda2 = cp.Parameter(pos=True)

        model_prediction = X @ self.betas 
        group_lasso_regularization = cp.sum(
            [self.lambda1s[i] * cp.norm(self.betas[self.group_ind[i] : self.group_ind[i+1]], 2)
            for i in self.group_range]
        )
        sparsity_regularization = cp.norm(self.betas, 1)


        objective = cp.Minimize(0.5 / y.size * cp.sum_squares(y - model_prediction)
            + group_lasso_regularization
            + self.lambda2 * sparsity_regularization)
        self.problem = cp.Problem(objective, [])

    def solve(self, lambdas, quick_run=False):
        VERBOSE = False
        SCS_MAX_ITERS = 10000
        SCS_EPS = 1e-3 # default eps
        SCS_HIGH_ACC_EPS = 1e-6
        ECOS_TOL = 1e-12
        REALDATA_MAX_ITERS = 4000
        for idx in self.group_range:
            self.lambda1s[idx].value = lambdas[idx]

        self.lambda2.value = lambdas[-1]

        ECOS_ITERS = 200
        self.problem.solve(solver=cp.ECOS)
        # self.problem.solve(solver=cp.ECOS, verbose=VERBOSE, abstol=ECOS_TOL, reltol=ECOS_TOL, abstol_inacc=ECOS_TOL, reltol_inacc=ECOS_TOL, max_iters=ECOS_ITERS)
        # try:
        #     self.problem.solve(solver=cp.ECOS, verbose=VERBOSE, abstol=ECOS_TOL, reltol=ECOS_TOL, abstol_inacc=ECOS_TOL, reltol_inacc=ECOS_TOL, max_iters=ECOS_ITERS)
        # except cp.SolverError:
        #     # self.problem.solve(solver=cp.SCS, verbose=VERBOSE, eps=SCS_HIGH_ACC_EPS/100, max_iters=SCS_MAX_ITERS * 4, use_indirect=False, normalize=False, warm_start=True)
        #     self.problem.solve(solver=cp.SCS, verbose=VERBOSE, eps=SCS_HIGH_ACC_EPS/10, max_iters=SCS_MAX_ITERS, use_indirect=False, normalize=False, warm_start=True)

        b_return = [
            self.betas[self.group_ind[i] : self.group_ind[i+1]].value for i in self.group_range
        ]

        return b_return

# (2) from gradient_descent_algo import Gradient_Descent_Algo
# (2-1) from fitted_model import Fitted_Model
class Fitted_Model:
    def __init__(self, num_lambdas):
        self.num_lambdas = num_lambdas
        self.lambda_history = []
        self.model_param_history = []
        self.cost_history = []

        self.current_cost = None
        self.best_cost = None
        self.current_lambdas = None
        self.best_lambdas = None

        self.num_solves = 0

    def update(self, new_lambdas, new_model_params, cost):
        self.lambda_history.append(new_lambdas)
        self.model_param_history.append(new_model_params)
        self.cost_history.append(cost)

        self.current_model_params = self.model_param_history[-1]
        self.current_cost = self.cost_history[-1]
        self.current_lambdas = self.lambda_history[-1]

        if self.best_cost is None or cost < self.best_cost:
            self.best_cost = cost
            self.best_lambdas = new_lambdas
            self.best_model_params = new_model_params

    def set_runtime(self, runtime):
        self.runtime = runtime

    def incr_num_solves(self):
        self.num_solves += 1

    def set_num_solves(self, num_solves):
        self.num_solves = num_solves

    def get_cost_diff(self):
        return self.cost_history[-2] - self.cost_history[-1]

    def __str__(self):
        return "cost %f, current_lambdas %s" % (self.current_cost, self.current_lambdas)

class Gradient_Descent_Algo:
    def __init__(self, data, settings=None, algo_settings=dict()):
        self._create_descent_settings(algo_settings)
        self.data = matrix_data(data, settings)
        self.settings = HC_Settings(settings)

        self._create_problem_wrapper()
        self._create_lambda_configs()

    def run(self, initial_lambda_set, debug=True, log_file=None):
        # self.log_file = log_file
        self.monitor = Monitor_HC()
        start_time = time.time()

        self.fmodel = Fitted_Model(initial_lambda_set[0].size)
        # best_cost = None
        # best_initial_lambdas = None
        for initial_lambdas in initial_lambda_set:
            # self.log("%s: initial_lambdas %s" % (self.method_label, initial_lambdas))
            self._run_lambdas(initial_lambdas, debug=debug)
            # if best_cost is None or best_cost > self.fmodel.best_cost:
                # best_cost = self.fmodel.best_cost
                # best_initial_lambdas = initial_lambdas
            # self.log("%s: best start lambda %s" % (self.method_label, best_initial_lambdas))

        runtime = time.time() - start_time
        # self.log("%s: runtime %s" % (self.method_label, runtime))
        self.fmodel.set_runtime(runtime)

    def _run_lambdas(self, initial_lambdas, debug=True):
        # start_history_idx = len(self.fmodel.cost_history)
        # warm up the problem
        Timer = time.time
        time_monitor = Timer()
        self._solve_wrapper(initial_lambdas, quick_run=True)
        # do a real run now
        model_params = self._solve_wrapper(initial_lambdas, quick_run=False)

        # Check that no model params are None
        if self._any_model_params_none(model_params):
            # self.log("ERROR: No model params fit for initial lambda values")
            self.fmodel.update(initial_lambdas, None, None)
            return

        current_cost = self.get_validate_cost(model_params)
        self.fmodel.update(initial_lambdas, model_params, current_cost)
        self.monitor.append({
            "time": Timer()-time_monitor, 
            "train_error": self.get_train_cost(model_params),
            "validation_error": current_cost, 
            "test_error": self.get_test_cost(model_params),
            "step_error": 0,
            "obj_error": 0})
        # self.log("self.fmodel.current_cost %f" % self.fmodel.current_cost)
        step_size = self.step_size_init
        for i in range(self.num_iters):
            lambda_derivatives = self._get_lambda_derivatives_wrapper()

            potential_lambdas, potential_model_params, potential_cost = self._run_potential_lambdas(
                step_size,
                lambda_derivatives,
                quick_run=True
            )

            while self._check_should_backtrack(potential_cost, step_size, lambda_derivatives) and step_size > self.step_size_min:
                if potential_cost is None: # If can't find a solution, shrink faster
                    step_size *= self.shrink_factor**3
                else:
                    step_size *= self.shrink_factor
                potential_lambdas, potential_model_params, potential_cost = self._run_potential_lambdas(
                    step_size,
                    lambda_derivatives,
                    quick_run=True
                )

            if self.fmodel.current_cost < potential_cost:
                break
            else:
                potential_lambdas, potential_model_params, potential_cost = self._run_potential_lambdas(
                    step_size,
                    lambda_derivatives,
                    quick_run=False
                )

                self.fmodel.update(potential_lambdas, potential_model_params, potential_cost)
                self.monitor.append({
                    "time": Timer()-time_monitor, 
                    "train_error": self.get_train_cost(potential_model_params),
                    "validation_error": potential_cost, 
                    "test_error": self.get_test_cost(potential_model_params),
                    "step_error": step_size,
                    "obj_error": self.fmodel.get_cost_diff()})

                # self.log("%s iter: %d step_size %f" % (self.method_label, i, step_size))
                # self.log("current model %s" % self.fmodel)
                # self.log("cost_history %s" % self.fmodel.cost_history[start_history_idx:])
                # self.log("current test cost %s" % self.get_test_cost(self.fmodel.best_model_params))

                if self.fmodel.get_cost_diff() < self.decr_enough_threshold:
                    # self.log("decrease amount too small %f" % self.fmodel.get_cost_diff())
                    break

            if step_size < self.step_size_min:
                # self.log("STEP SIZE TOO SMALL %f" % step_size)
                break

            # sys.stdout.flush()

        # self.log("TOTAL ITERS %d" % i)
        # self.log("full_cost_hist: %s" % self.fmodel.cost_history[start_history_idx:])
        # self.log("current_test_cost: %s" % self.get_test_cost(self.fmodel.best_model_params))

    def _check_should_backtrack(self, potential_cost, step_size, lambda_derivatives):
        if potential_cost is None:
            return True
        backtrack_thres_raw = self.fmodel.current_cost - self.backtrack_alpha * step_size * np.linalg.norm(lambda_derivatives)**2
        backtrack_thres = self.fmodel.current_cost if backtrack_thres_raw < 0 else backtrack_thres_raw
        return potential_cost > backtrack_thres

    def _run_potential_lambdas(self, step_size, lambda_derivatives, quick_run=False):
        potential_lambdas = self._get_updated_lambdas(
            step_size,
            lambda_derivatives
        )
        try:
            potential_model_params = self._solve_wrapper(potential_lambdas, quick_run=quick_run)
        except cp.error.SolverError:
            potential_model_params = None

        if self._any_model_params_none(potential_model_params):
            potential_cost = None
        else:
            potential_cost = self.get_validate_cost(potential_model_params)
        return potential_lambdas, potential_model_params, potential_cost

    def _solve_wrapper(self, lambdas, quick_run):
        start_solve_time = time.time()
        model_params = self.problem_wrapper.solve(lambdas, quick_run=quick_run)
        if quick_run is False:
            self.fmodel.incr_num_solves()
        # self.log("solve runtime %f" % (time.time() - start_solve_time))
        return model_params

    def _get_lambda_derivatives_wrapper(self):
        start_solve_time = time.time()
        lambda_derivatives = self._get_lambda_derivatives()
        # self.log("lambda_derivatives runtime %f" % (time.time() - start_solve_time))
        # self.log("lambda_derivatives %s" % lambda_derivatives)
        return lambda_derivatives

    def _get_updated_lambdas(self, method_step_size, lambda_derivatives):
        current_lambdas = self.fmodel.current_lambdas
        new_step_size = method_step_size
        if self.use_boundary:
            potential_lambdas = current_lambdas - method_step_size * lambda_derivatives

            for idx in range(0, current_lambdas.size):
                if current_lambdas[idx] > self.lambda_mins[idx] and potential_lambdas[idx] < self.lambda_mins[idx]:
                    smaller_step_size = self.boundary_factor * (current_lambdas[idx] - self.lambda_mins[idx]) / lambda_derivatives[idx]
                    new_step_size = min(new_step_size, smaller_step_size)
                    # self.log("USING THE BOUNDARY %f" % new_step_size)

        return np.maximum(current_lambdas - new_step_size * lambda_derivatives, self.lambda_mins)

    def log(self, log_str):
        if self.log_file is None:
            print (log_str)
        else:
            self.log_file.write("%s\n" % log_str)
            self.log_file.flush()

    @staticmethod
    def _any_model_params_none(model_params):
        if model_params is None:
            return True
        else:
            return any([m is None for m in model_params])

## Main part
class SGL_Hillclimb_Base(Gradient_Descent_Algo):
    def _create_descent_settings(self, algo_settings=None):
        self.num_iters = 20
        self.step_size_init = 1
        self.step_size_min = 1e-6
        self.shrink_factor = 0.1
        self.decr_enough_threshold = 1e-8 * 5
        self.use_boundary = False
        self.boundary_factor = 0.999999
        self.backtrack_alpha = 0.001

        for attr in algo_settings.keys():
            assert (attr in self.__dict__.keys()), print(attr + " is not in setting of HC method")
            if attr in self.__dict__.keys():
                self.__dict__[attr] = algo_settings[attr]
            else:
                print()

    def get_train_cost(self, model_params):
        return testerror_grouped(
            self.data.X_train,
            self.data.y_train,
            model_params
        )

    def get_validate_cost(self, model_params):
        return testerror_grouped(
            self.data.X_validate,
            self.data.y_validate,
            model_params
        )

    def get_test_cost(self, model_params):
        return testerror_grouped(
            self.data.X_test,
            self.data.y_test,
            model_params
        )

    def _get_lambda_derivatives(self):
        # restrict the derivative to the differentiable space
        beta_minis = []
        beta_nonzeros = []
        for beta in self.fmodel.current_model_params:
            beta_nonzero = self._get_nonzero_indices(beta)
            beta_nonzeros.append(beta_nonzero)
            beta_minis.append(beta[beta_nonzero])

        complete_beta_nonzero = np.concatenate(beta_nonzeros)
        X_train_mini = self.data.X_train[:, complete_beta_nonzero]
        X_validate_mini = self.data.X_validate[:, complete_beta_nonzero]

        if complete_beta_nonzero.size == 0:
            return np.zeros(self.fmodel.current_lambdas)

        return self._get_lambda_derivatives_mini(X_train_mini, X_validate_mini, beta_minis)

class SGL_Hillclimb(SGL_Hillclimb_Base):
    method_label = "SGL_Hillclimb"

    def _create_lambda_configs(self):
        self.lambda_mins = [1e-6] * (self.settings.num_experiment_groups + 1)

    def _create_problem_wrapper(self):
        self.problem_wrapper = GroupedLassoProblemWrapper(
            self.data.X_train,
            self.data.y_train,
            self.settings.get_expert_group_sizes()
        )

    def _get_lambda_derivatives_mini(self, X_train_mini, X_validate_mini, beta_minis):
        def _get_block_diag_component(idx):
            beta = beta_minis[idx]
            if beta.size == 0:
                return np.matrix(np.zeros((0,0))).T

            betabeta = beta * beta.T
            block_diag_component = -1 * self.fmodel.current_lambdas[idx] / get_norm2(beta, power=3) * betabeta
            return block_diag_component

        def _get_diagmatrix_component(idx):
            beta = beta_minis[idx]
            if beta.size == 0:
                return np.matrix(np.zeros((0,0))).T
            return self.fmodel.current_lambdas[idx] / get_norm2(beta) * np.identity(beta.size)

        def _get_dbeta_dlambda1(beta, matrix_to_invert, num_features_before):
            if beta.size == 0:
                return np.zeros((matrix_to_invert.shape[0], 1))
            else:
                normed_beta = beta / get_norm2(beta)
                zero_normed_beta = np.concatenate([
                    np.matrix(np.zeros(num_features_before)).T,
                    normed_beta,
                    np.matrix(np.zeros(total_features - normed_beta.size - num_features_before)).T
                ])

                dbeta_dlambda1 = sp.sparse.linalg.lsmr(matrix_to_invert, -1 * np.ravel(zero_normed_beta))[0]
                return np.matrix(dbeta_dlambda1).T

        total_features = X_train_mini.shape[1]
        complete_beta = np.concatenate(beta_minis)

        XX = X_train_mini.T * X_train_mini

        block_diag_components = [_get_block_diag_component(idx) for idx in range(0, self.settings.num_experiment_groups)]
        diagonal_components = [_get_diagmatrix_component(idx) for idx in range(0, self.settings.num_experiment_groups)]
        dgrouplasso_dlambda = sp.linalg.block_diag(*block_diag_components) + sp.linalg.block_diag(*diagonal_components)

        matrix_to_invert = 1.0 / self.settings.num_train * XX + dgrouplasso_dlambda

        dbeta_dlambda1s = None
        num_features_before = 0
        for beta in beta_minis:
            dbeta_dlambda1 = _get_dbeta_dlambda1(beta, matrix_to_invert, num_features_before)
            num_features_before += beta.size

            if dbeta_dlambda1s is None:  # not initialized yet
                dbeta_dlambda1s = dbeta_dlambda1
            else:
                dbeta_dlambda1s = np.hstack([dbeta_dlambda1s, dbeta_dlambda1])

        dbeta_dlambda1s = np.matrix(dbeta_dlambda1s)
        dbeta_dlambda2 = np.matrix(sp.sparse.linalg.lsmr(matrix_to_invert, -1 * np.ravel(np.sign(complete_beta)))[0]).T

        err_vector = self.data.y_validate - X_validate_mini * complete_beta
        df_dlambda1s = -1.0 / self.settings.num_validate * (X_validate_mini * dbeta_dlambda1s).T * err_vector
        df_dlambda1s = np.reshape(np.array(df_dlambda1s), df_dlambda1s.size)
        df_dlambda2 = -1.0 / self.settings.num_validate * (X_validate_mini * dbeta_dlambda2).T * err_vector
        return np.concatenate((df_dlambda1s, [df_dlambda2[0,0]]))

    @staticmethod
    def _get_nonzero_indices(beta, threshold=1e-4):
        return np.reshape(np.array(np.greater(np.abs(beta), threshold).T), (beta.size, ))

def get_norm2(vector, power=1):
    return np.power(np.linalg.norm(vector, ord=None), power)
 
def testerror_grouped(X, y, betas):
    complete_beta = np.concatenate(betas)
    diff = y - X * complete_beta
    return 0.5 / y.size * get_norm2(diff, power=2)