import operator
import numpy as np

class risk_measure:
    
    def VaR(self, data, alpha=0.1):
        data = sorted(data)
        item = (int((alpha * len(data))) - 1)
        self.VaR = -(data[item])

    def CVaR(self, data, alpha=0.1):
        data = sorted(data)
        item = int((alpha * len(data))) 
        CVaR_list = data[0:item] 
        self.CVaR = -(np.sum(CVaR_list) / len(CVaR_list)) 

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

    def std(self, data):
        self.std = np.std(data)

    def var(self, data):
        self.var = np.var(data)