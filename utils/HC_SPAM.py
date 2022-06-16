#%%
import numpy as np
import cvxpy as cp
import time
import scipy as sp
import pandas as pd

from Data_Generator import SPAM_Data

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

def matrix_data(data, settings):
    m_data = SPAM_Data(data, settings)
    m_data.X_train = data.X_train
    m_data.X_validate = data.X_validate
    m_data.X_test = data.X_test
    m_data.y_train = np.reshape(data.y_train, [settings.num_train, 1])
    m_data.y_validate = np.reshape(data.y_validate, [settings.num_validate, 1])
    m_data.y_test = np.reshape(data.y_test, [settings.num_test, 1])
    m_data.X_full = np.vstack((m_data.X_train, m_data.X_validate, m_data.X_test))
    m_data.train_idx = np.arange(0, settings.num_train)
    m_data.validate_idx = np.arange(settings.num_train, settings.num_train + settings.num_validate)
    m_data.test_idx = np.arange(settings.num_train + settings.num_validate, settings.num_train + settings.num_validate + settings.num_test)

    return m_data
            
#%%
def _make_discrete_diff_matrix_ord2(x_features):
    num_samples = len(x_features)
    d1_matrix = np.matrix(np.zeros((num_samples, num_samples)))
    # 1st, figure out ordering of samples for the feature
    sample_ordering = np.argsort(x_features)
    ordered_x = x_features[sample_ordering]
    d1_matrix[range(num_samples - 1), sample_ordering[:-1]] = -1
    d1_matrix[range(num_samples - 1), sample_ordering[1:]] = 1
    inv_dists = 1.0 / (ordered_x[np.arange(1, num_samples)] - ordered_x[np.arange(num_samples - 1)])
    inv_dists = np.append(inv_dists, 0)

    # Check that the inverted distances are all greater than zero
    assert(np.min(inv_dists) >= 0)
    D = d1_matrix * np.matrix(np.diagflat(inv_dists)) * d1_matrix
    return D

class SparseAdditiveModelProblemWrapper:
    def __init__(self, X, train_indices, y, tiny_e=0):
        self.tiny_e = tiny_e
        self.y = y

        num_samples, num_features = X.shape
        self.num_samples = num_samples
        self.num_features = num_features

        # Create smooth penalty matrix for each feature
        self.diff_matrices = []
        for i in range(num_features):
            D = _make_discrete_diff_matrix_ord2(X[:,i])
            self.diff_matrices.append(D)

        self.train_indices = train_indices
        self.lambdas = [cp.Parameter(pos=True)]
        for i in range(self.num_features):
            self.lambdas.append(cp.Parameter(pos=True))

        self.thetas = cp.Variable([self.num_samples, self.num_features])
        num_train = train_indices.size
        objective = 0.5/num_train * cp.sum_squares(self.y - self.thetas[self.train_indices, :] @ np.ones([self.num_features, 1]))
        objective += sum([1.0/num_train * self.lambdas[0] * cp.pnorm(self.thetas[:,i], 2) for i in range(self.num_features)])
        for i in range(len(self.diff_matrices)):
            objective += 1.0/num_train * self.lambdas[i + 1] * cp.pnorm(self.diff_matrices[i] @ self.thetas[:,i], 1)
        objective += 0.5/num_train * self.tiny_e * cp.sum_squares(self.thetas)
        self.problem = cp.Problem(cp.Minimize(objective))

    def solve(self, lambdas, warm_start=True, quick_run=False):
        SCS_MAX_ITERS = 10000
        SCS_EPS = 1e-3 # default eps
        SCS_HIGH_ACC_EPS = 1e-6
        ECOS_TOL = 1e-12
        REALDATA_MAX_ITERS = 4000
        for i in range(lambdas.size):
            self.lambdas[i].value = lambdas[i]

        if not quick_run:
            eps = SCS_HIGH_ACC_EPS * 1e-3
            max_iters = SCS_MAX_ITERS * 10
        else:
            eps = SCS_EPS
            max_iters = SCS_MAX_ITERS

        if quick_run:
            self.problem.solve(solver=cp.SCS, verbose=False, max_iters=max_iters, eps=eps, warm_start=warm_start)
        else:
            self.problem.solve(solver=cp.SCS, verbose=False, max_iters=max_iters, use_indirect=False, eps=eps, normalize=False, warm_start=warm_start)

        if self.problem.value > 0 and self.problem.status in [cp.OPTIMAL,  cp.OPTIMAL_INACCURATE]:
            return self.thetas.value
        else:
            if self.problem.value < 0:
                print ("Warning: Negative problem solution from cvxpy")
            # return None

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
        self.data = matrix_data(data, settings)
        self.settings = settings

        self._create_descent_settings()
        self._create_problem_wrapper()
        self._create_lambda_configs()

    def run(self, initial_lambda_set, debug=True, log_file=None):
        self.log_file = log_file
        self.monitor = Monitor_HC()
        start_time = time.time()

        self.fmodel = Fitted_Model(initial_lambda_set[0].size)
        # best_cost = None
        # best_initial_lambdas = None
        for initial_lambdas in initial_lambda_set:
            # self.log("%s: initial_lambdas %s" % (self.method_label, initial_lambdas))
            self._run_lambdas(initial_lambdas, debug=debug)

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
class BetaForm:
    eps = 1e-8

    def __init__(self, idx, theta, diff_matrix, log_file):
        self.log_file = log_file
        # self.log("create beta form")
        self.idx = idx
        self.theta = theta
        self.theta_norm = sp.linalg.norm(theta, ord=None)
        self.diff_matrix = diff_matrix

        # Find the null space for the subsetted diff matrix
        start = time.time()
        zero_theta_idx = self._get_zero_theta_indices(diff_matrix @ theta)
        u, s, v = sp.linalg.svd(diff_matrix[zero_theta_idx,:])
        # self.log("SVD done %f" % (time.time() - start))
        null_mask = np.ones(v.shape[1])
        null_mask[:s.size] = s <= self.eps
        # null_space = sp.compress(null_mask, v, axis=0)
        null_space = np.compress(null_mask, v, axis=0)
        # null_matrix = np.matrix(sp.transpose(null_space))
        null_matrix = np.matrix(np.transpose(null_space))
        start = time.time()
        beta, istop, itn, normr, normar, norma, conda, normx = sp.sparse.linalg.lsmr(null_matrix, np.ravel(theta), atol=self.eps, btol=self.eps)
        # self.log("sp.sparse.linalg.lsmr done %f, istop %d, itn %d" % ((time.time() - start), istop, itn))
        self.beta = np.matrix(beta).T
        self.u = null_matrix

        # Check that we reformulated theta but it is still very close to the original theta
        if sp.linalg.norm(self.u * self.beta - self.theta, ord=2) > self.eps:
            return None
            # self.log("Warning: Reformulation is off: diff %f" % sp.linalg.norm(self.u * self.beta - self.theta, ord=2))
        # self.log("create beta form success")

    def log(self, log_str):
        if self.log_file is None:
            print (log_str)
        else:
            self.log_file.write("%s\n" % log_str)

    def __str__(self):
        return "beta %s, theta %s" % (self.beta, self.theta)

    @staticmethod
    def _get_zero_theta_indices(theta, threshold=1e-10):
        return np.ravel(np.less(np.abs(theta), threshold))

class Sparse_Add_Model_Hillclimb_Base(Gradient_Descent_Algo):
    def _create_descent_settings(self):
        self.num_iters = 15
        self.step_size_init = 1
        self.step_size_min = 1e-6
        self.shrink_factor = 0.1
        self.decr_enough_threshold = 1e-4 * 5
        self.use_boundary = True
        self.boundary_factor = 0.999999
        self.backtrack_alpha = 0.001

    def get_test_cost(self, model_params):
        return testerror_sparse_add_smooth(
            self.data.y_test,
            self.data.test_idx,
            model_params
        )

    def get_train_cost(self, model_params):
        return testerror_sparse_add_smooth(
            self.data.y_train,
            self.data.train_idx,
            model_params
        )

    def get_validate_cost(self, model_params):
        return testerror_sparse_add_smooth(
            self.data.y_validate,
            self.data.validate_idx,
            model_params
        )

class Sparse_Add_Model_Hillclimb(Sparse_Add_Model_Hillclimb_Base):
    method_label = "Sparse_Add_Model_Hillclimb"

    def _create_lambda_configs(self):
        self.lambda_mins = [1e-6] * (self.settings.num_features + 1)

    def _create_problem_wrapper(self):
        self.problem_wrapper = SparseAdditiveModelProblemWrapper(
            self.data.X_full,
            self.data.train_idx,
            self.data.y_train
        )
        self.train_I = np.matrix(np.eye(self.settings.num_samples)[self.data.train_idx,:])

    def _get_lambda_derivatives(self):
        # First filter out the thetas that are completely zero
        nonzero_thetas_idx = self._get_nonzero_theta_vectors(self.fmodel.current_model_params)
        # self.log("nonzero_thetas_idx %s" % nonzero_thetas_idx)
        # Now reformulate the remaining thetas using the differentiable space
        nonzeros_idx = np.where(nonzero_thetas_idx)[0]

        if nonzeros_idx.size == 0:
            return np.array([0] * self.fmodel.num_lambdas)

        beta_u_forms = list(map(
            lambda i: BetaForm(i, self.fmodel.current_model_params[:, i], self.problem_wrapper.diff_matrices[i], log_file=self.log_file),
            nonzeros_idx
        ))
        sum_dtheta_dlambda = self._get_sum_dtheta_dlambda(beta_u_forms, nonzero_thetas_idx)
        fitted_y_validate = np.sum(self.fmodel.current_model_params[self.data.validate_idx, :], axis=1, keepdims=True)
        dloss_dlambda = -1.0/self.data.y_validate.size * sum_dtheta_dlambda[self.data.validate_idx, :].T * (self.data.y_validate - fitted_y_validate)

        return np.ravel(dloss_dlambda) # flatten the matrix

    def _get_sum_dtheta_dlambda(self, beta_u_forms, nonzero_thetas_idx):
        def create_b_diag_elem(i):
            u = beta_u_forms[i].u
            beta = beta_u_forms[i].beta
            theta_norm = beta_u_forms[i].theta_norm
            b_diag_elem = 1.0/theta_norm * (np.eye(beta.size) - beta * beta.T/(theta_norm**2))
            return b_diag_elem

        def make_rhs_col0(i):
            theta_norm = beta_u_forms[i].theta_norm
            beta = beta_u_forms[i].beta
            return beta/theta_norm

        def make_diag_rhs(i):
            u = beta_u_forms[i].u
            theta = beta_u_forms[i].theta
            diff_matrix = beta_u_forms[i].diff_matrix
            theta_norm = beta_u_forms[i].theta_norm
            # Zero out the entries that are essentially zero.
            # Otherwise np.sign will give non-zero values
            zeroed_diff_theta = self._zero_theta_indices(diff_matrix @ theta)
            return u.T * diff_matrix.T * np.sign(zeroed_diff_theta).T
        

        # beta_u_forms = beta_u_forms
        num_nonzero_thetas = len(beta_u_forms)

        # Create part of the Hessian matrix
        b_diag_elems = list(map(create_b_diag_elem, range(num_nonzero_thetas)))
        b_diag = sp.linalg.block_diag(*b_diag_elems)

        # Create rhs elements
        rhs_col0 = list(map(make_rhs_col0, range(num_nonzero_thetas)))
        rhs_col0 = np.vstack(rhs_col0)
        rhs_diag = list(map(make_diag_rhs, range(num_nonzero_thetas)))
        rhs_diag = sp.linalg.block_diag(*rhs_diag)
        insert_idx = np.minimum(np.arange(self.settings.num_features)[~nonzero_thetas_idx], rhs_diag.shape[1])
        # insert zero columns that corresponded to the zero thetas
        rhs_diag = np.insert(rhs_diag, insert_idx, np.zeros((rhs_diag.shape[0], 1)), axis=1)
        rhs_matrix = np.hstack((rhs_col0, rhs_diag))

        lambda0 = self.fmodel.current_lambdas[0]
        u_matrices = list(map(lambda i: beta_u_forms[i].u, range(num_nonzero_thetas)))
        u_matrices = np.hstack(u_matrices)
        uu = u_matrices.T * self.train_I.T * self.train_I * u_matrices
        tiny_e_matrix = self.problem_wrapper.tiny_e * np.eye(uu.shape[0])
        hessian = uu + lambda0 * b_diag + tiny_e_matrix

        # start = time.time()
        dbeta_dlambda = list(map(
            lambda j: np.matrix(sp.sparse.linalg.lsmr(hessian, -1 * np.ravel(rhs_matrix[:,j]))[0]).T,
            range(rhs_matrix.shape[1])
        ))
        dbeta_dlambda = np.hstack(dbeta_dlambda)
        # self.log("lsmr time %f" % (time.time() - start))
        # self.log("u_matrices shape: %d %d" % (u_matrices.shape[0], u_matrices.shape[1]))
        # self.log("dbeta_dlambda shape: %d %d" % (dbeta_dlambda.shape[0], dbeta_dlambda.shape[1])) 
        sum_dtheta_dlambda = u_matrices * dbeta_dlambda
        # self.log("sum_dtheta_dlambda shape: %d %d" % (sum_dtheta_dlambda.shape[0], sum_dtheta_dlambda.shape[1])) 
        return sum_dtheta_dlambda

    @staticmethod
    def _get_nonzero_theta_vectors(thetas, threshold=1e-8):
        nonzero_thetas_idx = [sp.linalg.norm(thetas[:,i], ord=2) > threshold for i in range(thetas.shape[1])]
        return np.array(nonzero_thetas_idx)

    @staticmethod
    def _zero_theta_indices(theta, threshold=1e-8):
        return np.multiply(theta, np.greater(np.abs(theta), threshold))

def get_norm2(vector, power=1):
    return np.power(np.linalg.norm(vector, ord=None), power)
 
def testerror_sparse_add_smooth(y, test_indices, thetas):
    err = y - np.sum(thetas[test_indices, :], axis=1, keepdims=True)
    return 0.5/y.size * get_norm2(err, power=2)