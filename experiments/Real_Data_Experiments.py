#%%
import sys
sys.path.append("..") 

import numpy as np
import pandas as pd

from utils.Data_Generator import Data_Generator_Wrapper, Real_Data_Fetcher, Real_Data_Setting
from utils.ElasticNet_Algorithms import iP_DCA, Bayesian_Method, Grid_Search, Random_Search, IGJO

# from utils.ElasticNet_Algorithms import IFDM

from utils.utils import performance_reporter, results_printer

#%%
def main():
    np.random.seed(42) # for reproduction
    num_repeat = 1
    num_train = 25
    num_validate = 25
    num_test = 0
    result_path = "../results"

    datasets = ["australian_scale"] # 690 * 14 230 230 230
    # datasets = ["gisette"] # 6000 * 5000
    # datasets = ["bodyfat"] # regression 252 14 数据过小
    # datasets = ["gisette", "madelon"] # 100 100 500
    # datasets = ["australian_scale", "madelon", "gisette"]
    # datasets = ["real-sim"] # 72,309 * 20,958 too large 
    # datasets = ["duke breast-cancer"] # 44 * 7129 11-11-22
    # num_train = 11
    # num_validate = 11
    # datasets = ["cifar10"] # 50,000 * 3,072
    # datasets = ["madelon"]
    # datasets = ["sensit"] # 78823 * 100

    Methods = ["GS", "RS", "Bayes", "DC", "HC"]

    DC_Setting = {
        "TOL": 1e-1,
        # "TOL": 1e-2,
        "initial_guess": np.array([10, 5]),
        "epsilon": 0,
        "rho": 1e-2,
        "beta_0": 1,
        "MAX_ITERATION": 20,
    }

    HC_Setting = {
        "num_iters": 50,
        "step_size_min": 1e-8,
        "initial_guess": 1e-2*np.ones(2),
        "decr_enough_threshold": 1e-8
    }

    IF_Setting = {
        "n_outer": 10,
    }

    # running_methods = ["DC"]

    for dataset in datasets:
        unique_marker = "_" + dataset + "_20220112"
        settings = Real_Data_Setting(dataset, num_train, num_validate, num_test, print_flag=True)
        for i in range(num_repeat):
            print("Experiment %2d/%2d" % (i+1, num_repeat))
            data_info = Data_Generator_Wrapper(Real_Data_Fetcher, settings, i+1)
            marker = unique_marker + "_%d_%d_%d" % (num_train, num_validate, num_test)

            if "GS" in Methods:
                Result_GS = Grid_Search(data_info)
                performance_reporter(Result_GS, 'Grid Search', "best")
                Result_GS.to_pickle("../results/real_data/GS_" + str(data_info.data_index) + marker + ".pkl")
            
            if "RS" in Methods:
                Result_RS = Random_Search(data_info)
                performance_reporter(Result_RS, 'Random Search', "best")
                Result_RS.to_pickle("../results/real_data/RS_" + str(data_info.data_index) + marker + ".pkl")

            if "Bayes" in Methods:
                Result_Bayes = Bayesian_Method(data_info)
                performance_reporter(Result_Bayes, 'Bayesian Method', "best")
                Result_Bayes.to_pickle("../results/real_data/Bayes_" + str(data_info.data_index) + marker + ".pkl")

            if "HC" in Methods:
                Result_HC = IGJO(data_info, HC_Setting)
                performance_reporter(Result_HC, 'HC method', "latest")
                Result_HC.to_pickle("../results/real_data/HC_" + str(data_info.data_index) + marker + ".pkl")
            
            if "DC" in Methods:
                Result_DC = iP_DCA(data_info, DC_Setting)
                performance_reporter(Result_DC, 'VF-iDCA', "latest")
                Result_DC.to_pickle("../results/real_data/DC_" + str(data_info.data_index) + marker + ".pkl")

            # if "IF" in Methods:
                # Result_IF = IFDM(data_info, IF_Setting)
                # performance_reporter(Result_IF, 'IF method', "latest")
                # Result_IF.to_pickle(result_path + "/real_data/IF_" + str(data_info.data_index) + marker + ".pkl")

        results_printer(num_repeat, "real_data", Methods, result_path, suffix=marker, latex = False)
#%%
if __name__ == "__main__":
    main()