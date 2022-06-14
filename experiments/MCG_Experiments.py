#%%
import sys
sys.path.append("..") 

import numpy as np
import pandas as pd
from utils.MCG_Algorithms import Grid_Search, Random_Search, iP_DCA, Bayesian_Method
# this problelm IGJO

from utils.Data_Generator import MCG_data, MCG_DataGenerator

from utils.utils import performance_reporter, results_printer

#%%
def main():
    class Matrix_Completion_Group_Settings():
        num_nonzero_s = 1
        num_rows = 60
        num_cols = 60
        num_row_groups = 12
        num_col_groups = 12
        num_row_features = 3 # num features per group
        num_col_features = 3 # num features per group
        num_nonzero_row_groups = 4
        num_nonzero_col_groups = 2
        train_perc = 0.3
        validate_perc = 0.15
        snr = 2
        gamma_to_row_col_m = 1.
        feat_factor = 1.

    np.random.seed(42)

    Methods = ["GS", "RS", "Bayes", "DC"]
    # Methods = ["DC"]
    DEBUG = False
    # DEBUG = True

    num_repeat = 1
    result_path = "../results"
    problem_name = "MCG"
    suffix = "_0125"

    settings = Matrix_Completion_Group_Settings()
    data_gen = MCG_DataGenerator(settings)

    o_data_list = [data_gen.matrix_completion_groups() for _ in range(num_repeat)]

    # matrix_data_list = ["../matrix_data/matrices_" + str(i) + ".npy" for i in range(num_repeat)]
    # for i in range(num_repeat):
    #     with open(matrix_data_list[i], "wb") as f:
    #         np.save(f, np.array(o_data_list[i].row_features))
    #         np.save(f, np.array(o_data_list[i].col_features))
    #         np.save(f, np.array(o_data_list[i].train_idx))
    #         np.save(f, np.array(o_data_list[i].validate_idx))
    #         np.save(f, np.array(o_data_list[i].test_idx))
    #         np.save(f, np.array(o_data_list[i].observed_matrix))
    #         np.save(f, np.array(o_data_list[i].real_alphas))
    #         np.save(f, np.array(o_data_list[i].real_betas))
    #         np.save(f, np.array(o_data_list[i].real_gamma))
    #         np.save(f, np.array(o_data_list[i].real_matrix))

    data = MCG_data(o_data_list[0])

    r = .1 * np.ones(1 + data.num_alphas + data.num_betas)
    r[0] = 1
    DC_Setting = {
        "MAX_ITERATION": 500,
        "TOL": 5e-2,
        "initial_r": r,
        "rho" : 1e-3,
        "c": 1
    }

    GS_Setting = {"n_grid" : 40}
    RS_Setting = {"num_search" : 1000}
    TPE_Setting = {"max_evals" : 1000}

    for ind in range(num_repeat):
        print("Experiment %2d/%2d" % (ind+1, num_repeat))
        
        observed_data = o_data_list[ind]
        data = MCG_data(observed_data)

        if "GS" in Methods:
            Result_GS = Grid_Search(data, GS_Setting, DEBUG = DEBUG)
            performance_reporter(Result_GS, 'Grid', "best")
            Result_GS.to_pickle(result_path + "/" + problem_name + "/GS_" + str(ind+1) + suffix + ".pkl")

        if "RS" in Methods:
            Result_RS = Random_Search(data, RS_Setting, DEBUG = DEBUG)
            performance_reporter(Result_RS, 'Random', "best")
            Result_RS.to_pickle(result_path + "/" + problem_name + "/RS_" + str(ind+1) + suffix + ".pkl")

        if "Bayes" in Methods:
            Result_Bayes = Bayesian_Method(data, TPE_Setting, DEBUG = DEBUG)
            performance_reporter(Result_Bayes, 'TPE', "best")
            Result_Bayes.to_pickle(result_path + "/" + problem_name + "/Bayes_" + str(ind+1) + suffix + ".pkl")

        if "DC" in Methods:
            Result_DC = iP_DCA(data, DC_Setting, DEBUG = DEBUG)
            performance_reporter(Result_DC, 'VF-iDCA', "latest")
            Result_DC.to_pickle(result_path + "/" + problem_name + "/DC_" + str(ind+1) + suffix + ".pkl")
        

    results_printer(num_repeat, "MCG", Methods, result_path, suffix=suffix, latex=False)
#%%
if __name__ == "__main__":
    main()

