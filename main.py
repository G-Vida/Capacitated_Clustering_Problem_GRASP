# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from Value_of_SS.VSS import ValueSS,Test_Scenarios_VSS, Test_Sigma_VSS, Test_Lambda_VSS
from instances.instanceCCP import InstanceCCP
from solver.solveCCP_stoch import SolverExactCCP_stoch
from solver.solveCCP_ev import SolverExactCCP_ev
from stability.stability import out_stability, in_stability
from heuristic.Grasp import Modified_Kmean, Local_Search, Local_Search_Bis
from heuristic.Tests import Test1, Test2, Test3, Test4, Test5, Test_Points
import time


np.random.seed(23)



#*******************************************************************************************************
# VALUE OF THE STOCHASTIC SOLUTION
#**************************************************
'''
# VSS ----------------------------------------------
N_POINTS = 100
N_SCENARIOS = 10
p=10

inst = InstanceCCP(N_POINTS, N_SCENARIOS)
inst.plot()

VSS, _, _ = ValueSS(inst, p)
print(f"VSS: {VSS}")
'''

'''
# TEST with different N_SCENARIOS ------------------
N_POINTS = 50
p = 10

VSS_mean_list = Test_Scenarios_VSS(N_POINTS, p)
print(f"VSS: {VSS_mean_list}")
'''

'''
# TEST with different values of SIGMA --------------------
N_POINTS = 50
N_SCENARIOS = 10
p = 10

VSS_sigma_list = Test_Sigma_VSS(N_POINTS, N_SCENARIOS, p)
print(f"VSS: {VSS_sigma_list}")
'''

'''
# TEST with different values of LAMBDA --------------------
N_POINTS = 50
N_SCENARIOS = 10
p = 10

VSS_lambda_list = Test_Lambda_VSS(N_POINTS, N_SCENARIOS, p)
print(f"VSS: {VSS_lambda_list}")
'''




#*******************************************************************************************************
# IN-SAMPLE & OUT-OF-SAMPLE STABILITY
#**************************************************
'''
#Large tree dim N'<S
N_POINTS = 50
p = 10
N_large = 1000
inst_large = InstanceCCP(N_POINTS, N_large)
sol_large =  SolverExactCCP_stoch(inst_large, p)


out_stability(sol_large, p)
in_stability(sol_large, p)

'''







#*******************************************************************************************************
# HEURISTICS
#***************************************************

#**************
# SMALL SETTING
#**************
'''
#TEST ONE - LOCAL SEARCH vs. LOCAL SEARCH BIS vs. GUROBI
N_POINTS = 150
N_SCENARIOS = 5
p_list = [10, 15, 20, 25, 30, 35]

Test1(N_POINTS, N_SCENARIOS, p_list)

'''
#TEST TWO - GRASP vs. GUROBI for different number of clusters
N_POINTS = 150
N_SCENARIOS = 5
p_list = [15]
Grasp_runs = 10

Test2(N_POINTS, N_SCENARIOS, p_list, Grasp_runs)    

'''
#TEST THREE - GRASP vs. GUROBI for different number Grasp's runs

N_POINTS = 150
N_SCENARIOS = 5
p = 19   # To avoid the critical behaviour of Gurobi at 20-24 clusters
Grasp_runs = [10, 20, 30, 40, 50]
Test3(N_POINTS, N_SCENARIOS, p, Grasp_runs)



#DIVENTA THREEEEEEEEEEEE
#TEST FOUR - GRASP vs. GUROBI changing the parameter lamda
N_POINTS = 150
N_SCENARIOS = 5
p = 18       # To avoid the critical behaviour of Gurobi at 20-24 clusters
lambda_list = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
Grasp_runs = 10

Test4(N_POINTS, N_SCENARIOS, p, lambda_list, Grasp_runs) 


#DIVENTA FOURRRRRRRRRRRRR
#TEST FIVE - GRASP vs. GUROBI changing the number of scenarios
N_POINTS = 150
scenario_list = [3, 6, 9, 12, 15, 18, 20]
p = 19       # To avoid the critical behaviour of Gurobi at 20-24 clusters
Grasp_runs = 10

Test5(N_POINTS, scenario_list, p, Grasp_runs) 




    

#******************
# CHANGING SETTINGS
#******************
Grasp_it = 10
n_scenarios = 3
n_points_list = [100, 100]
n_clusters_list = [12, 13]

Test_Points(Grasp_it, n_points_list, n_clusters_list, n_scenarios)

'''