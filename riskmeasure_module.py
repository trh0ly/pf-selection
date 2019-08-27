"""
#################################################################################
# Risk Measure Module                                                           #
# © Thomas Robert Holy 2019                                                     #
# Version 0.1.0                                                                 #
# E-Mail: thomas.robert.holy[at]uni-jena.de                                     #
#################################################################################
"""
#--------------------------------------------------------------------------------
# Package import

import numpy as np

#--------------------------------------------------------------------------------
# Define class risk_measure

class risk_measure:
    """
    Function to determine Value at Risk of Dataset;
    Given: Dataset, Alpha-Quantile
    Output: Value at Risk at Alpha-Quantile
    """
    def VaR(self, data, alpha=0.1):
        data = sorted(data)
        item = (int((alpha * len(data))) - 1)
        self.VaR = -(data[item])
    """
    Function to determine Conidtional Value at Risk of Dataset
    Given: Dataset, Alpha-Quantile
    Output: Conditional Value at Risk from Beginn of the Dataset
    to Alpha-Quantile
    """
    def CVaR(self, data, alpha=0.1):
        data = sorted(data)
        item = int((alpha * len(data))) 
        CVaR_list = data[0:item] 
        self.CVaR = -(np.sum(CVaR_list) / len(CVaR_list)) 

    """
    Function to determine Power Spectral Risk Measure of Dataset
    Given: Dataset, Gamma
    Output: Power Sepctral Risk Measure and Expected Value of Dataset
    """
    def Power(self, data, gamma=0.5):
        data = sorted(data)
        
        subj_ws_list = [] 
        counter_1 = len(data) 
        counter_2 = (len(data) - 1)

        for i in data:
            subj_ws = (np.power((counter_1 / len(data)), gamma)) - (np.power((counter_2 / len(data)), gamma))
            counter_1 -= 1 
            counter_2 -= 1 
            subj_ws_list.append(subj_ws) 
        subj_ws_list = subj_ws_list[::-1] 
        self.power = ((- np.matmul(np.transpose(data), subj_ws_list)), np.mean(data))

    """
    Function to determine standard deviation of Dataset
    Given: Dataset
    Output: Standard Deviation of Dataset
    """
    def std(self, data):
        self.std = np.std(data)

    """
    Function to determine Variance of Dataset
    Given: Dataset
    Output: Variance of Dataset
    """
    def var(self, data):
        self.var = np.var(data)

    """
    Function to determine every Risk Measure
    Function to determine Variance of Dataset
    Given: Dataset, Alpha-Quantile, Gamma
    Output: Value at Risk, Conditional Value at Risk,
    Power Sepctral Risk Measure
    """
    def get_all(self, data, alpha=0.1, gamma=0.5):
        # VaR
        data = sorted(data)
        item = (int((alpha * len(data))) - 1)

        # CVaR
        CVaR_list = data[0:item + 1] 

        # Power
        subj_ws_list = [] 
        counter_1 = len(data) 
        counter_2 = (len(data) - 1)

        for i in data:
            subj_ws = (np.power((counter_1 / len(data)), gamma)) - (np.power((counter_2 / len(data)), gamma))
            counter_1 -= 1 
            counter_2 -= 1 
            subj_ws_list.append(subj_ws) 
        subj_ws_list = subj_ws_list[::-1] 
        self.all = (-(data[item]), -(np.sum(CVaR_list) / len(CVaR_list)) , (- np.matmul(np.transpose(data), subj_ws_list)), np.mean(data))
