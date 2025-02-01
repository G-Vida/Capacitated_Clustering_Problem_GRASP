import scipy.stats as st
import numpy as np
from instances.instanceCCP import InstanceCCP
from solver.solveCCP_stoch import SolverExactCCP_stoch
import matplotlib.pyplot as plt
import random


def out_stability(sol_test, p):

    N_large = sol_test.inst.n_scenarios
    #N_points = sol_test.inst.n_points
    N = 100    #30 <--
    M = 30    #5 <--
    step = 10  #10 <--
    k=0
    conf_int = []
    means = []
    x1 = np.zeros(0)
    n_rows = len(range(10, N+10, step))  # <--
    results = np.zeros((n_rows, M))    #we put results along the rows
    #names = [None for _ in range(5, N+5, step)]
    idx = list(range(N_large))
    
    for n in range(10, N+10, step): # <--
        for m in range(M):
            #Simulate M scenario trees di dimensins n => the scenarios are taken randomly from the large tree
            inst = InstanceCCP(instance = sol_test.inst)
            idx_sample = np.sort(np.array(random.sample(idx, n)))

            inst.sampled(idx_sample)
            sol = SolverExactCCP_stoch(inst, p)
            Y, X, _, _, _ = sol.solve()
            results[k, m] = sol_test.evaluate_f(X, Y)

        print("---")
        #We use the t-Student confidence interval
        conf_int.append((st.t.interval(0.95, M-1, loc=np.mean(results[k]), scale=st.sem(results[k]))))
        means.append(np.mean(results[k]))
        #names[k] = str(n)
        k=k+1
        a = np.repeat(n,M)
        x1 = np.append(x1,a)
        
    print("Results out_stability: ", results)
    print("Means: ", means)
    print("Confidence interval out_stability: ", conf_int)
    
    y1 = np.matrix.flatten(results)
    plt.scatter(x1, y1, s = 1.3, c = 'black', edgecolor = 'black')
    plt.title("Out of sample")
    plt.show()
        
    return



def in_stability(sol_test, p):
    N_large = sol_test.inst.n_scenarios
    #N_points = sol_test.inst.n_points
    
    N = 30   #100 <--
    M = 5    #30 <--
    step = 5    #10 <--
    k=0
    conf_int = []
    means = []
    x1 = np.zeros(0)
    n_rows = len(range(5, N+5, step))   #<--
    results = np.zeros((n_rows, M))    #we put results along the rows
    #names = [None for _ in range(5, N+5, step)]
    idx = list(range(N_large))
    
    for n in range(5, N+5, step):  #<--
        for m in range(M):
            #Simulate M scenario trees di dimensins n => the scenarios are taken randomly from the large tree
            inst = InstanceCCP(instance = sol_test.inst)
            idx_sample = np.sort(np.array(random.sample(idx, n)))
            inst.sampled(idx_sample)
            
            sol = SolverExactCCP_stoch(inst, p)
            Y, X, _, _, _ = sol.solve()
            results[k, m] = sol.evaluate_f(X, Y)   #We evaluate each X_m

        print("---")
        #We use the t-Student confidence interval
        conf_int.append((st.t.interval(0.95, M-1, loc=np.mean(results[k]), scale=st.sem(results[k]))))
        means.append(np.mean(results[k]))
        #names[k] = str(n)
        k=k+1
        a = np.repeat(n,M)
        x1 = np.append(x1,a)
        
    print("Results in_stability: ", results)
    print("Means: ", means)
    print("Confidence interval in_stability: ", conf_int)
    
    y1 = np.matrix.flatten(results)
    plt.scatter(x1, y1, s = 1.3, c = 'black', edgecolor = 'black')
    plt.title("In-sample stability")
    plt.xlabel('Number of scenarios')
    plt.ylabel('Values of the objective functions)')
    plt.show()
        
    return