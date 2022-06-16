#%%
import sys
sys.path.append("..") 

import numpy as np
import pandas as pd
from utils.SVM_CV_Algorithms import VF_iDCA, Random_Search, Grid_Search, Bayesian_Method, Bayesian_Method_Simple
from utils.Data_Generator import Data_Generator_Wrapper, CV_Real_Data_Setting, CV_Real_Data_Fetcher

from utils.utils import performance_reporter, results_printer

#%%
def main():
    np.random.seed(42) # for reproduction
    num_repeat = 1
    num_CV = 3
    num_one_fold = 0 # it will be calculated automatically if num_one_fold = 0 
    num_test = 0 # it will be calculated automatically if num_test = 0 

    result_path = "../results"

    # datasets = [
    #     "liver-disorders_scale",
    #     "diabetes_scale",  
    #     "breast-cancer_scale",
    #     "sonar", "a1a", "w1a"
    # ]
    datasets = ["liver-disorders_scale"]

    Methods = ["GS", "RS", "Bayes", "DC"]
    # Methods = ["DC"]

    TPE_Setting = {
        "max_evals": 10
    }

    for dataset in datasets:
        settings = CV_Real_Data_Setting(dataset, num_one_fold, num_CV, num_test, print_flag=True)
        data_info = Data_Generator_Wrapper(CV_Real_Data_Fetcher, settings, 0)
        settings.print_flag = False 
        unique_marker = "_CV_2_" + dataset 
        marker = unique_marker + "_%d_%d_%d" % (settings.num_one_fold, settings.num_CV, settings.num_test) 

        DC_Setting = {
            "TOL": 1e-1, # 1
            # "TOL": 1e-2,
            # "TOL": 0, # for plot
            "initial_w_bar": 1e-6 * np.ones(settings.num_features),
            "initial_r": 10,
            "epsilon": 0,
            "rho": 1e-3,
            "beta_0": 1,
            "MAX_ITERATION": 100,
            "delta": 1
        }


        for i in range(num_repeat):
            print("Experiments: " + str(i+1) + "/" + str(num_repeat))
            data_info = Data_Generator_Wrapper(CV_Real_Data_Fetcher, settings, i+1)

            if "GS" in Methods:
                Result_GS = Grid_Search(data_info)
                performance_reporter(Result_GS, 'Grid Search', "best")
                Result_GS.to_pickle(result_path + "/svm/GS_" + str(data_info.data_index) + marker + ".pkl")

            if "RS" in Methods:
                Result_RS = Random_Search(data_info)
                # performance_reporter(Result_RS, 'Random Search', "best")
                Result_RS.to_pickle(result_path + "/svm/RS_" + str(data_info.data_index) + marker + ".pkl")

            if "Bayes" in Methods:
                # Result_Bayes = Bayesian_Method_Simple(data_info)  # 1 
                Result_Bayes = Bayesian_Method(data_info, TPE_Setting)   # 2
                performance_reporter(Result_Bayes, 'Bayesian Method', "best")
                Result_Bayes.to_pickle(result_path + "/svm/Bayes_" + str(data_info.data_index) + marker + ".pkl")

            if "DC" in Methods:
                Result_DC = VF_iDCA(data_info, DC_Setting, DEBUG=False)
                performance_reporter(Result_DC, 'VF-iDCA', "latest")
                Result_DC.to_pickle(result_path + "/svm/DC_" + str(data_info.data_index) + marker + ".pkl")

        results_printer(num_repeat, "svm", Methods, result_path, suffix=marker, latex=False)

#%%
if __name__ == "__main__":
    main()

