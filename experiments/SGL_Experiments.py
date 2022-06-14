#%%
import sys
sys.path.append("..") 

import numpy as np

from utils.Data_Generator import Data_Generator_Wrapper, SGL_Data_Generator, SGL_Setting
from utils.SGL_Algorithms import iP_DCA, Bayesian_Method, Grid_Search, Random_Search, Imlicit_Differntiation

from multiprocessing import Pool
from functools import partial

from utils.utils import performance_reporter, results_printer

#%%
def main():
    np.random.seed(42) # for reproduction
    num_threads = 1
    num_repeat  = 1
    num_train = 100
    num_validate = 100
    num_test = 100
    num_features = 600
    num_experiment_groups = 30

    unique_marker = "_20220110_eps5"
    marker = unique_marker + "_%d_%d_%d_%d_%d" % (num_train, num_validate, num_test, num_features, num_experiment_groups)
    result_path = "../results"

    running_methods = ["GS", "RS", "TPE", "DC", "HC"]
    # running_methods = ["DC"]

    print(marker)

    settings = SGL_Setting(num_train, num_validate, num_test, num_features, num_experiment_groups)
    list_data_info = [
        Data_Generator_Wrapper(SGL_Data_Generator, settings, i+1) 
        for i in range(num_repeat)]

    dc_initial = 10*np.ones(settings.num_experiment_groups + 1)

    DC_Setting = {
        "TOL": 1e-1,
        "initial_guess": dc_initial,
        "epsilon": 0, 
        "beta_0": 1,
        "rho": .1,
        "MAX_ITERATION": 50,
        "c": .01,
        "delta": 5
    }

    iP_DCA_Runner = partial(iP_DCA, DC_Setting = DC_Setting)

    HC_Setting = {
        "num_iters": 50,
        "step_size_min": 1e-6,
        "initial_guess": 1e-2*np.ones(settings.num_experiment_groups + 1),
        "decr_enough_threshold": 1e-4 * 5
    }
    Implicit_Runner = partial(Imlicit_Differntiation, HC_Setting = HC_Setting)

    if num_threads > 1:
        print("Do multiprocessing")
        pool = Pool(processes=num_threads)
        if "GS" in running_methods: 
            List_Result_GS = pool.map(Grid_Search, list_data_info)
        if "RS" in running_methods: 
            List_Result_RS = pool.map(Random_Search, list_data_info)
        if "TPE" in running_methods: 
            List_Result_Bayes = pool.map(Bayesian_Method, list_data_info)
        if "DC" in running_methods: 
            List_Result_DC = pool.map(iP_DCA_Runner, list_data_info)
        if "HC" in running_methods:
            List_Result_HC = pool.map(Implicit_Runner, list_data_info)
    else:
        print("No multiprocessing")
        if "GS" in running_methods: 
            List_Result_GS = list(map(Grid_Search, list_data_info))
        if "RS" in running_methods: 
            List_Result_RS = list(map(Random_Search, list_data_info))
        if "TPE" in running_methods: 
            List_Result_Bayes = list(map(Bayesian_Method, list_data_info))
        if "DC" in running_methods: 
            List_Result_DC = list(map(iP_DCA_Runner, list_data_info))
        if "HC" in running_methods: 
            List_Result_HC = list(map(Implicit_Runner, list_data_info))

    for i in range(num_repeat):
        print("Experiments: " + str(i+1) + "/" + str(num_repeat))
        data_info = list_data_info[i]

        if "GS" in running_methods: 
            Result_GS = List_Result_GS[i]
            performance_reporter(Result_GS, 'Grid Search', "best")
            Result_GS.to_pickle("../results/sgl/GS_" + str(data_info.data_index) + marker + ".pkl")

        if "RS" in running_methods: 
            Result_RS = List_Result_RS[i]
            performance_reporter(Result_RS, 'Random Search', "best")
            Result_RS.to_pickle("../results/sgl/RS_" + str(data_info.data_index) + marker + ".pkl")

        if "TPE" in running_methods: 
            Result_Bayes = List_Result_Bayes[i]
            performance_reporter(Result_Bayes, 'Bayesian Method', "best")
            Result_Bayes.to_pickle("../results/sgl/Bayes_" + str(data_info.data_index) + marker + ".pkl")

        if "DC" in running_methods: 
            Result_DC = List_Result_DC[i]
            performance_reporter(Result_DC, 'Approx sol', "latest")
            Result_DC.to_pickle("../results/sgl/DC_" + str(data_info.data_index) + marker + ".pkl")

        if "HC" in running_methods: 
            Result_HC = List_Result_HC[i]
            performance_reporter(Result_HC, 'HC method', "latest")
            Result_HC.to_pickle("../results/sgl/HC_" + str(data_info.data_index) + marker + ".pkl")

    results_printer(num_repeat, "sgl", running_methods, result_path, suffix=marker)

#%%
if __name__ == "__main__":
    main()