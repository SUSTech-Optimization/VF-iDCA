#%%
import sys
sys.path.append("..") 

import numpy as np
import pandas as pd

from utils.Data_Generator import Data_Generator_Wrapper, ElasticNet_Data_Generator, ElasticNet_Setting
from utils.ElasticNet_Algorithms import iP_DCA, Bayesian_Method, Grid_Search, Random_Search, IGJO
# from utils.ElasticNet_Algorithms import IFDM # IFDM requires package sparse_ho and a little modification to the source code

from utils.utils import performance_reporter, results_printer

#%%
def main():
    np.random.seed(42) # for reproduction
    num_repeat = 1
    num_train = 100
    num_validate = 20
    num_test = 250
    num_features = 250

    unique_marker = "sth"
    marker = unique_marker + "_%d_%d_%d_%d" % (num_train, num_validate, num_test, num_features)

    result_path = "../results"

    Methods = ["GS", "RS", "Bayes", "DC", "HC"]
    # Methods = ["DC"]

    DC_Setting = {
        "TOL": 1e-2,
        "initial_guess": np.array([10, 5]),
        "MAX_ITERATION": 100,
        "rho": 1e-3,
        "delta": 5,
        "c": 0.5
    }

    HC_Setting = {
        "num_iters": 50,
        "step_size_min": 1e-12,
        "initial_guess": 1e-2*np.ones(2),
        "decr_enough_threshold": 1e-12
    }

    IF_Setting = {
        "n_outer": 50,
        "alpha0": 1e-2*np.ones(2)
    }

    settings = ElasticNet_Setting(num_train, num_validate, num_test, num_features)
    list_data_info = [
        Data_Generator_Wrapper(ElasticNet_Data_Generator, settings, i+1)
        for i in range(num_repeat)
    ]
    

    for i in range(num_repeat):
        print("Experiments: " + str(i+1) + "/" + str(num_repeat))
        data_info = list_data_info[i]

        if "GS" in Methods:
            Result_GS = Grid_Search(data_info)
            performance_reporter(Result_GS, 'Grid Search', "best")
            Result_GS.to_pickle(result_path + "/elasticnet/GS_" + str(data_info.data_index) + marker + ".pkl")

        if "RS" in Methods:
            Result_RS = Random_Search(data_info)
            performance_reporter(Result_RS, 'Random Search', "best")
            Result_RS.to_pickle(result_path + "/elasticnet/RS_" + str(data_info.data_index) + marker + ".pkl")

        if "Bayes" in Methods:
            Result_Bayes = Bayesian_Method(data_info)
            performance_reporter(Result_Bayes, 'Bayesian Method', "best")
            Result_Bayes.to_pickle(result_path + "/elasticnet/Bayes_" + str(data_info.data_index) + marker + ".pkl")
        
        if "HC" in Methods:
            Result_HC = IGJO(data_info, HC_Setting)
            performance_reporter(Result_HC, 'HC method', "latest")
            Result_HC.to_pickle(result_path + "/elasticnet/HC_" + str(data_info.data_index) + marker + ".pkl")

        # if "IF" in Methods:
        #     Result_IF = IFDM(data_info, IF_Setting)
        #     performance_reporter(Result_IF, 'IF method', "latest")
        #     Result_IF.to_pickle(result_path + "/elasticnet/IF_" + str(data_info.data_index) + marker + ".pkl")

        if "DC" in Methods:
            Result_DC = iP_DCA(data_info, DC_Setting)
            performance_reporter(Result_DC, 'VF-iDCA', "latest")
            Result_DC.to_pickle(result_path + "/elasticnet/DC_" + str(data_info.data_index) + marker + ".pkl")

    results_printer(num_repeat, "elasticnet", Methods, result_path, suffix=marker, latex=False)

#%%
if __name__ == "__main__":
    main()