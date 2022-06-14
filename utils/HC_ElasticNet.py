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

def matrix_data(data):
    m_data = Data()
    m_data.X_train, m_data.X_validate, m_data.X_test = np.matrix(data.X_train), np.matrix(data.X_validate), np.matrix(data.X_test)
    m_data.y_train, m_data.y_validate, m_data.y_test = np.matrix(data.y_train).T, np.matrix(data.y_validate).T, np.matrix(data.y_test).T
    return m_data

#%%
class ElasticNetProblemWrapper:
    def __init__(self, X, y):
        n = X.shape[1]
        self.beta = cp.Variable([n, 1])
        self.lambda1 = cp.Parameter(pos=True)
        self.lambda2 = cp.Parameter(pos=True)
        objective = cp.Minimize(0.5 * cp.sum_squares(y - X @ self.beta)
            + self.lambda1 * cp.norm(self.beta, 1)
            + 0.5 * self.lambda2 * cp.sum_squares(self.beta))
        self.problem = cp.Problem(objective, [])

    def solve(self, lambdas, quick_run=None, warm_start=True):
        self.lambda1.value = lambdas[0]
        self.lambda2.value = lambdas[1]
        self.problem.solve(solver=cp.ECOS)
        return self.beta.value

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
    def __init__(self, data, settings=None):
        self.data = matrix_data(data)
        self.settings = settings

        self._create_descent_settings()
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
                # if potential_cost is not None:
                    # self.log("(shrinking) potential_lambdas %s, cost %f, step, %f" % (potential_lambdas, potential_cost, step_size))
                # else:
                    # self.log("(shrinking) potential_lambdas None!")

            if self.fmodel.current_cost < potential_cost:
                # self.log("COST IS INCREASING! %f" % potential_cost)
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

    def get_test_cost(self, model):
        return None

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
class Elastic_Net_Hillclimb(Gradient_Descent_Algo):
    method_label = "Elastic_Net_Hillclimb"

    def _create_descent_settings(self):
        self.num_iters = 100
        self.step_size_init = 1
        self.step_size_min = 1e-8
        self.shrink_factor = 0.1
        self.use_boundary = False
        self.boundary_factor = 0.7
        self.decr_enough_threshold = 1e-4 * 5
        # self.decr_enough_threshold = 1e-8 * 5
        self.backtrack_alpha = 0.001

    def _create_lambda_configs(self):
        self.lambda_mins = [1e-5] * 2

    def _create_problem_wrapper(self):
        self.problem_wrapper = ElasticNetProblemWrapper(
            self.data.X_train,
            self.data.y_train
        )

    def get_train_cost(self, model_params):
        return testerror_elastic_net(
            self.data.X_train,
            self.data.y_train,
            model_params
        )

    def get_validate_cost(self, model_params):
        return testerror_elastic_net(
            self.data.X_validate,
            self.data.y_validate,
            model_params
        )

    def get_test_cost(self, model_params):
        return testerror_elastic_net(
            self.data.X_test,
            self.data.y_test,
            model_params
        )

    def _get_lambda_derivatives(self):
        betas = self.fmodel.current_model_params
        nonzero_indices = self.get_nonzero_indices(betas)

        # If everything is zero, gradient is zero
        if np.sum(nonzero_indices) == 0:
            return np.zeros((1,2))

        X_train_mini = self.data.X_train[:, nonzero_indices]
        X_validate_mini = self.data.X_validate[:, nonzero_indices]
        betas_mini = betas[nonzero_indices]

        eye_matrix = np.matrix(np.identity(betas_mini.size))
        # Note: on certain computers, it will be difficult to run X_train_mini.T * X_train_mini in parallel
        to_invert_matrix = X_train_mini.T * X_train_mini + self.fmodel.current_lambdas[1] * eye_matrix

        dbeta_dlambda1, istop, itn, normr, normar, norma, conda, normx = sp.sparse.linalg.lsmr(to_invert_matrix, -1 * np.ravel(np.sign(betas_mini)))
        dbeta_dlambda1 = np.matrix(dbeta_dlambda1).T
        dbeta_dlambda2, istop, itn, normr, normar, norma, conda, normx = sp.sparse.linalg.lsmr(to_invert_matrix, -1 * np.ravel(betas_mini))
        dbeta_dlambda2 = np.matrix(dbeta_dlambda2).T

        err_vector = self.data.y_validate - X_validate_mini * betas_mini
        gradient_lambda1 = -1 * (X_validate_mini * dbeta_dlambda1).T * err_vector
        gradient_lambda2 = -1 * (X_validate_mini * dbeta_dlambda2).T * err_vector

        return np.array([gradient_lambda1[0,0], gradient_lambda2[0,0]])

    @staticmethod
    def get_nonzero_indices(some_vector, threshold=1e-4):
        return np.reshape(np.array(np.greater(np.abs(some_vector), threshold).T), (some_vector.size, ))

def get_norm2(vector, power=1):
    return np.power(np.linalg.norm(vector, ord=None), power)
 
def testerror_elastic_net(X, y, b):
    return 0.5/y.size * get_norm2(y - X @ b, power=2)