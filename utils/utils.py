import numpy as np
import pandas as pd

class Monitor():
    def __init__(self):
        self.time = []
        self.train_error = []
        self.validation_error = []
        self.test_error = []
    
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

class Monitor_DC(Monitor):
    def __init__(self):
        super().__init__()
        self.step_err = []
        self.penalty = []
        self.beta = []

def performance_reporter(Result_df, method, TYPE = 'best' , VERBOSE=True):
    if TYPE == "best":
        Best = Result_df[Result_df['validation_error'] == min(Result_df['validation_error'])].iloc[0]
        if VERBOSE:
            print('%20s | time cost %.2fs, validation error: %.2e, test error: %.2e' %
            (method, Result_df.iloc[-1]['time'], Best['validation_error'], Best['test_error']))
        else:
            return [
                Result_df.iloc[-1]['time'], 
                Best['validation_error'], 
                Best['test_error']]
    elif TYPE == "latest":
        if VERBOSE:
            print('%20s | time cost %.2fs, validation error: %.2e, test error: %.2e' %
            (method, Result_df.iloc[-1]['time'], Result_df.iloc[-1]['validation_error'], Result_df.iloc[-1]['test_error']))
        else:
            return [
                Result_df.iloc[-1]['time'], 
                Result_df.iloc[-1]['validation_error'], 
                Result_df.iloc[-1]['test_error']]


def result_printer(method, result_list):
    result_mean = np.mean(np.array(result_list), axis=0)
    result_std = np.std(np.array(result_list), axis=0)
    print("%25s | time cost: %.2f(%.2f), validation error: %.2f(%.2f), test error: %.2f(%.2f)" %(
        method, result_mean[0], result_std[0], result_mean[1], result_std[1], 
        result_mean[2], result_std[2], 
    ))

def latex_table_printer(method, result_list):
    if len(result_list[0]) == 3:
        result_mean = np.mean(np.array(result_list), axis=0)
        result_std = np.std(np.array(result_list), axis=0)
        print("%25s & %.2f $\\pm$ %.2f & %.2f $\\pm$ %.2f & %.2f $\\pm$ %.2f \\\\" %(
            method, result_mean[0], result_std[0], result_mean[1], result_std[1], 
            result_mean[2], result_std[2], 
        ))
    if len(result_list[0]) == 4:
        result_mean = np.mean(np.array(result_list), axis=0)
        result_std = np.std(np.array(result_list), axis=0)
        print("%25s & %.2f $\\pm$ %.2f & %.2e $\\pm$ %.2e & %.2f $\\pm$ %.2f & %.2f $\\pm$ %.2f \\\\" %(
            method, result_mean[0], result_std[0], result_mean[1], result_std[1], 
            result_mean[2], result_std[2], result_mean[3], result_std[3]
        ))

def results_printer(num_repeat, problem_name, Methods, result_path = "../results", suffix="", latex=False):
    if "GS" in Methods:
        list_GS = []
        for ind in range(num_repeat):
            Result_GS = pd.read_pickle(result_path + "/" + problem_name + "/GS_" + str(ind+1) + suffix + ".pkl")
            list_GS.append(performance_reporter(Result_GS, ' ', "best" , False))

    if "RS" in Methods:
        list_RS = []
        for ind in range(num_repeat):
            Result_RS = pd.read_pickle(result_path + "/" + problem_name + "/RS_" + str(ind+1) + suffix + ".pkl")
            list_RS.append(performance_reporter(Result_RS, ' ', "best" , False))

    if "Bayes" in Methods:
        list_Bayes = []
        for ind in range(num_repeat):
            Result_Bayes = pd.read_pickle(result_path + "/" + problem_name + "/Bayes_" + str(ind+1) + suffix + ".pkl")
            list_Bayes.append(performance_reporter(Result_Bayes, ' ', "best", False))
    
    if "HC" in Methods:
        list_HC = []
        for ind in range(num_repeat):
            Result_HC = pd.read_pickle(result_path + "/" + problem_name + "/HC_" + str(ind+1) + suffix + ".pkl")
            list_HC.append(performance_reporter(Result_HC, ' ', "latest", False))

    if "DC" in Methods:
        list_DC = []
        for ind in range(num_repeat):
            Result_DC = pd.read_pickle(result_path + "/" + problem_name + "/DC_" + str(ind+1) + suffix + ".pkl")
            list_DC.append(performance_reporter(Result_DC, ' ', "latest", False))
    
    if "IF" in Methods:
        list_IF = []
        for ind in range(num_repeat):
            Result_IF = pd.read_pickle(result_path + "/" + problem_name + "/IF_" + str(ind+1) + suffix + ".pkl")
            list_IF.append(performance_reporter(Result_IF, ' ', "latest", False))
            
    if latex:
        if "GS" in Methods: latex_table_printer("Grid", list_GS)
        if "RS" in Methods: latex_table_printer("Random", list_RS)
        if "Bayes" in Methods: latex_table_printer("TPE", list_Bayes)
        if "HC" in Methods: latex_table_printer("IGJO", list_HC)
        if "IF" in Methods: latex_table_printer("IFDM", list_IF)
        if "DC" in Methods: latex_table_printer("VF-iDCA", list_DC)
    else:
        if "GS" in Methods: result_printer("Grid Search", list_GS)
        if "RS" in Methods: result_printer("Random Search", list_RS)
        if "Bayes" in Methods: result_printer("Bayesian Method", list_Bayes)
        if "HC" in Methods: result_printer("IGJO", list_HC)
        if "IF" in Methods: result_printer("IFDM", list_IF)
        if "DC" in Methods: result_printer("VF-iDCA", list_DC)
