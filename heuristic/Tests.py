# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from instances.instanceCCP import InstanceCCP
from solver.solveCCP_stoch import SolverExactCCP_stoch
from heuristic.Grasp import Modified_Kmean, Local_Search, Local_Search_Bis, feval, GRASP
import time






def Test1(N_POINTS, N_SCENARIOS, p_list):

    cpu_time_heu = []
    cpu_time_heu_bis = []
    cpu_time_Gurobi = []
    solutions_heu = []
    solutions_heu_bis = []
    solutions_Gurobi = []
    nit = []
    nit_bis = []

    inst_heuristics = InstanceCCP(N_POINTS, N_SCENARIOS)
    inst_heuristics.plot()

    for i in range(len(p_list)):
        print(i)
        startlocal = time.time()
        it, X_heu, Y_heu = Local_Search(inst_heuristics, p_list[i])
        endlocal = time.time()
        comp_time_local = endlocal - startlocal
        cpu_time_heu.append(comp_time_local)
        nit.append(it)
        print("Local search finished")
    
        startlocal2 = time.time()
        it_bis, X_heu_bis, Y_heu_bis = Local_Search_Bis(inst_heuristics, p_list[i])
        endlocal2 = time.time()
        comp_time_local2 = endlocal2 - startlocal2
        cpu_time_heu_bis.append(comp_time_local2)
        nit_bis.append(it_bis)
        print("Local search bis finished")

        startGurobi = time.time()
        sol = SolverExactCCP_stoch(inst_heuristics, p_list[i])
        Y_G, X_G, exact_solution, _, _ = sol.solve()
        endGurobi = time.time()
        comp_time_Gurobi = endGurobi - startGurobi
        cpu_time_Gurobi.append(comp_time_Gurobi)
        print("Exact solver finished")
        
        solutions_heu.append(feval(inst_heuristics, X_heu, Y_heu))
        solutions_heu_bis.append(feval(inst_heuristics, X_heu_bis, Y_heu_bis))
        solutions_Gurobi.append(exact_solution)

    # RESULTS
    print(f"N.iterations  Local search: {nit} s")
    print(f"N.iterations Local search bis: {nit_bis} s")
    print(f"computational time vector Local search: {cpu_time_heu} s")
    print(f"computational time vector Local search bis: {cpu_time_heu_bis} s")
    print(f"computational time vector Gurobi: {cpu_time_Gurobi} s")

    print("Local search solutions: ", solutions_heu)
    print("Local search bis solutions: ", solutions_heu_bis)
    print("Exact solutions: ", solutions_Gurobi)

    # PLOTS
    plt.plot(p_list, solutions_heu, label = 'Local search solutions')
    plt.plot(p_list, solutions_heu_bis, label = 'Local search bis solutions')
    plt.plot(p_list, solutions_Gurobi, label = 'Exact solutions')
    plt.xlabel('Maximum number of clusters')
    plt.ylabel('Value of the objective function')
    plt.title('Local search solution vs. Exact solution')
    plt.legend(['Local search solutions','Local search bis solutions', 'Exact solutions'])
    plt.xticks(p_list)
    plt.show()

    plt.plot(p_list, cpu_time_heu, label = 'Local search computational time')
    plt.plot(p_list, cpu_time_heu_bis, label = 'Local search bis computational time')
    plt.plot(p_list, cpu_time_Gurobi, label = 'Exact computational time')
    plt.xlabel('Maximum number of clusters')
    plt.ylabel('Computational time')
    plt.title('Local search vs. Exact computational times')
    plt.legend(['Local search comp. time', 'Local search bis comp. time', 'Exact comp. time'])
    plt.xticks(p_list)
    plt.show()
    
    return





def Test2(N_POINTS, N_SCENARIOS, p_list, Grasp_it):  
    
    cpu_time_Grasp = []
    cpu_time_Grasp_bis = []
    cpu_time_Gurobi = []
    solutions_Grasp = []
    solutions_Grasp_bis = []
    solutions_Gurobi = []

    inst_Grasp = InstanceCCP(N_POINTS, N_SCENARIOS)
    inst_Grasp.plot()

    for i in range(len(p_list)):
        print(i)
        startlocal = time.time()
        
        # '' for normal Local search, 'bis' for modified Local search
        X_Grasp, Y_Grasp, objf_Grasp = GRASP(inst_Grasp, p_list[i], Grasp_it, FLAG = '')
        endlocal = time.time()
        comp_time_local = endlocal - startlocal
        cpu_time_Grasp.append(comp_time_local)
        print("Grasp finished")
        
        startlocal = time.time()
        
        # '' for normal Local search, 'bis' for modified Local search
        X_Grasp_bis, Y_Grasp_bis, objf_Grasp_bis = GRASP(inst_Grasp, p_list[i], Grasp_it, FLAG = 'bis')
        endlocal = time.time()
        comp_time_local = endlocal - startlocal
        cpu_time_Grasp_bis.append(comp_time_local)
        print("GRASP bis finished")
    
        startGurobi = time.time()
        sol = SolverExactCCP_stoch(inst_Grasp, p_list[i])
        Y_G, X_G, exact_solution, _, _ = sol.solve()
        endGurobi = time.time()
        comp_time_Gurobi = endGurobi - startGurobi
        cpu_time_Gurobi.append(comp_time_Gurobi)
        print("Exact solver finished")
    
        solutions_Grasp.append(objf_Grasp)
        solutions_Grasp_bis.append(objf_Grasp_bis)
        solutions_Gurobi.append(exact_solution)

    # RESULTS
    print(f"computational time vector GRASP: {cpu_time_Grasp} s")
    print(f"computational time vector GRASP bis: {cpu_time_Grasp_bis} s")
    print(f"computational time vector Gurobi: {cpu_time_Gurobi} s")

    print("GRASP solutions: ", solutions_Grasp)
    print("Grasp bis solutions: ", solutions_Grasp_bis)
    print("Exact solutions: ", solutions_Gurobi)

    # PLOTS
    plt.plot(p_list, solutions_Grasp, label = 'GRASP solutions')
    plt.plot(p_list, solutions_Grasp_bis, label = 'GRASP bis solutions')
    plt.plot(p_list, solutions_Gurobi, label = 'Exact solutions')
    plt.xlabel('Maximum number of clusters')
    plt.ylabel('Value of the objective function')
    plt.title('GRASP solution vs. Exact solution')
    plt.legend(['GRASP solutions','GRASP bis solutions', 'Exact solutions'])
    plt.ylim(510, 2740)
    plt.yticks([750, 1000, 1250, 1500, 1750, 2000, 2250, 2500])
    plt.show()

    plt.plot(p_list, cpu_time_Grasp, label = 'GRASP computational time')
    plt.plot(p_list, cpu_time_Grasp_bis, label = 'GRASP bis computational time')
    plt.plot(p_list, cpu_time_Gurobi, label = 'Exact computational time')
    plt.xlabel('Maximum number of clusters')
    plt.ylabel('Computational time')
    plt.title('GRASP vs. Exact computational times')
    plt.legend(['GRASP comp. time', 'GRASP bis comp. time', 'Exact comp. time'])
    plt.show()
    
    
    gap = []
    for q in range(len(solutions_Grasp)):   
        #Percentage gap above the exact solution
        gap.append((solutions_Grasp[q] - solutions_Gurobi[q])/solutions_Gurobi[q]*100)  

    plt.plot(p_list, gap, c = 'm')
    plt.xlabel('N. of scenarios')
    plt.ylabel('Percentage gap')
    plt.title('Percentage gap for 150 points between GRASP and Exact sol.')
    plt.ylim([0, max(gap)+2])
    plt.xticks(p_list)
    plt.show()
    
        
    return


def Test3(N_POINTS, N_SCENARIOS, p, Grasp_list):
        
    cpu_time_Grasp = []
    cpu_time_Gurobi = []
    solutions_Grasp = []
    solutions_Gurobi = []

    inst_Grasp = InstanceCCP(N_POINTS, N_SCENARIOS)
    inst_Grasp.plot()

    for i in range(len(Grasp_list)):
        print(i)
        startlocal = time.time()
        
        # '' for normal Local search, 'bis' for modified Local search
        X_Grasp, Y_Grasp, objf_Grasp = GRASP(inst_Grasp, p, Grasp_list[i], FLAG = '')
        endlocal = time.time()
        comp_time_local = endlocal - startlocal
        cpu_time_Grasp.append(comp_time_local)
        print("GRASP finished")
        
        startlocal = time.time()
        
    
        startGurobi = time.time()
        sol = SolverExactCCP_stoch(inst_Grasp, p)
        Y_G, X_G, exact_solution, _, _ = sol.solve()
        endGurobi = time.time()
        comp_time_Gurobi = endGurobi - startGurobi
        cpu_time_Gurobi.append(comp_time_Gurobi)
        print("Exact solver finished")
    
        solutions_Grasp.append(objf_Grasp)
        solutions_Gurobi.append(exact_solution)

    # RESULTS
    print(f"computational time vector GRASP: {cpu_time_Grasp} s")
    print(f"computational time vector Gurobi: {cpu_time_Gurobi} s")

    print("GRASP solutions: ", solutions_Grasp)
    print("Exact solutions: ", solutions_Gurobi)

    # PLOTA
    plt.plot(Grasp_list, solutions_Grasp, label = 'GRASP solutions')
    plt.plot(Grasp_list, solutions_Gurobi, c = "g", label = 'Exact solutions')
    plt.xlabel('Runs')
    plt.ylabel('Value of the objective function')
    plt.title('GRASP solution vs. Exact solution')
    plt.legend(['GRASP solutions', 'Exact solutions'])
    plt.show()

    plt.plot(Grasp_list, cpu_time_Grasp, label = 'GRASP computational time')
    plt.plot(Grasp_list, cpu_time_Gurobi, c = "g", label = 'Exact computational time')
    plt.xlabel('Runs')
    plt.ylabel('Computational time')
    plt.title('GRASP vs. Exact computational times')
    plt.legend(['GRASP comp. time', 'Exact comp. time'])
    plt.show()
    
    gap = []
    for q in range(len(solutions_Grasp)):   
        #Percentage gap above the exact solution
        gap.append((solutions_Grasp[q] - solutions_Gurobi[q])/solutions_Gurobi[q]*100)   
        
    plt.plot(Grasp_list, gap, c = 'm')
    plt.xlabel('Runs')
    plt.ylim(0, max(gap)+2)
    plt.ylabel('Percentage gap')
    plt.title('Percentage gap between GRASP and Exact solutions')
    plt.show()
    return


def Test4(N_POINTS, N_SCENARIOS, p, l_list, Grasp_it):  
    
    cpu_time_Grasp = []
    cpu_time_Gurobi = []
    solutions_Grasp = []
    solutions_Gurobi = []

    inst_Grasp = InstanceCCP(N_POINTS, N_SCENARIOS)
    inst_Grasp.plot()

    for i in range(len(l_list)):
        print(i)
        inst_Grasp.l = l_list[i]     #We change lambda at each iteration
        
        startlocal = time.time()
        # '' for normal Local search, 'bis' for modified Local search
        X_Grasp, Y_Grasp, objf_Grasp = GRASP(inst_Grasp, p, Grasp_it, FLAG = '')
        endlocal = time.time()
        comp_time_local = endlocal - startlocal
        cpu_time_Grasp.append(comp_time_local)
        print("GRASP finished")
                
   
        startGurobi = time.time()
        sol = SolverExactCCP_stoch(inst_Grasp, p)
        Y_G, X_G, exact_solution, _, _ = sol.solve()
        endGurobi = time.time()
        comp_time_Gurobi = endGurobi - startGurobi
        cpu_time_Gurobi.append(comp_time_Gurobi)
        print("Exact solver finished")
    
        solutions_Grasp.append(objf_Grasp)
        solutions_Gurobi.append(exact_solution)

    # RESULTS
    print(f"computational time vector GRASP: {cpu_time_Grasp} s")
    print(f"computational time vector Gurobi: {cpu_time_Gurobi} s")

    print("GRASP solutions: ", solutions_Grasp)
    print("Exact solutions: ", solutions_Gurobi)

    # PLOTS
    plt.plot(l_list, solutions_Grasp, label = 'GRASP solutions')
    plt.plot(l_list, solutions_Gurobi, c = "g",label = 'Exact solutions')
    plt.xlabel('Parameter \u03BB')
    plt.ylabel('Value of the objective function')
    plt.title('GRASP solution vs. Exact solution')
    plt.legend(['GRASP solutions', 'Exact solutions'])
    plt.xticks(l_list)
    plt.show()

    plt.plot(l_list, cpu_time_Grasp, label = 'GRASP computational time')
    plt.plot(l_list, cpu_time_Gurobi, c = "g", label = 'Exact computational time')
    plt.xlabel('Parameter \u03BB')
    plt.ylabel('Computational time')
    plt.title('GRASP vs. Exact computational times')
    plt.legend(['GRASP comp. time', 'Exact comp. time'])
    plt.xticks(l_list)
    plt.show()
    
    
    gap = []
    for q in range(len(solutions_Grasp)):   
        #Percentage gap above the exact solution
        gap.append((solutions_Grasp[q] - solutions_Gurobi[q])/solutions_Gurobi[q]*100)  
    
    plt.plot(l_list, gap, c = 'm')
    plt.xlabel('Parameter \u03BB')
    plt.ylim(0, max(gap)+2)
    plt.xticks(l_list)
    plt.ylabel('Percentage gap')
    plt.title('Percentage gap between GRASP and Exact solutions')
    plt.show()
    return



def Test5(N_POINTS, scenario_list, p, Grasp_it):
        
    cpu_time_Grasp = []
    cpu_time_Gurobi = []
    solutions_Grasp = []
    solutions_Gurobi = []

    inst_Grasp = InstanceCCP(N_POINTS, scenario_list[0])
    inst_Grasp.plot()

    for i in range(len(scenario_list)):
        print(i)
        inst_Grasp.w = np.random.lognormal(mean=2.3, sigma=0.8, size = (N_POINTS, scenario_list[i]))
        
        startlocal = time.time()
        
        # '' for normal Local search, 'bis' for modified Local search
        X_Grasp, Y_Grasp, objf_Grasp = GRASP(inst_Grasp, p, Grasp_it, FLAG = '')
        endlocal = time.time()
        comp_time_local = endlocal - startlocal
        cpu_time_Grasp.append(comp_time_local)
        print("GRASP finished")
        
        startGurobi = time.time()
        sol = SolverExactCCP_stoch(inst_Grasp, p)
        Y_G, X_G, exact_solution, _, _ = sol.solve()
        endGurobi = time.time()
        comp_time_Gurobi = endGurobi - startGurobi
        cpu_time_Gurobi.append(comp_time_Gurobi)
        print("Exact solver finished")
    
        solutions_Grasp.append(objf_Grasp)
        solutions_Gurobi.append(exact_solution)

    # RESULTS
    print(f"computational time vector GRASP: {cpu_time_Grasp} s")
    print(f"computational time vector Gurobi: {cpu_time_Gurobi} s")

    print("GRASP solutions: ", solutions_Grasp)
    print("Exact solutions: ", solutions_Gurobi)

    # PLOTS
    ax = plt.plot(scenario_list, solutions_Grasp, label = 'GRASP solutions')
    plt.plot(scenario_list, solutions_Gurobi, c = "g", label = 'Exact solutions')
    plt.xticks(scenario_list)
    plt.xlabel('N. of scenarios')
    plt.ylabel('Value of the objective function')
    plt.title('GRASP solution vs. Exact solution')
    plt.legend(['GRASP solutions', 'Exact solutions'])
    plt.show()

    plt.plot(scenario_list, cpu_time_Grasp, label = 'GRASP computational time')
    plt.plot(scenario_list, cpu_time_Gurobi, c = "g", label = 'Exact computational time')
    plt.xticks(scenario_list)
    plt.xlabel('N. of scenarios')
    plt.ylabel('Computational time')
    plt.title('GRASP vs. Exact computational times')
    plt.legend(['GRASP comp. time', 'Exact comp. time'])
    plt.show()
    
    
    gap = []
    for q in range(len(solutions_Grasp)):   
        #Percentage gap above the exact solution
        gap.append((solutions_Grasp[q] - solutions_Gurobi[q])/solutions_Gurobi[q]*100)  
    


    plt.plot(scenario_list, gap, c = 'm')
    plt.xlabel('N. of scenarios')
    plt.ylabel('Percentage gap')
    plt.title('Percentage gap between GRASP and Exact solutions')
    plt.ylim([0, max(gap)+2])
    plt.xticks(scenario_list)
    plt.show()
    return




def Test_Points(Grasp_it, n_points_list, p_list, n_scenarios):

    cpu_time_Grasp = []
    cpu_time_Grasp_bis = []
    cpu_time_Gurobi = []
    solutions_Grasp = []
    solutions_Grasp_bis = []
    solutions_Gurobi = []
    
    for i in range(len(n_points_list)):
        print(n_points_list[i])
    
        inst_Grasp = InstanceCCP(n_points_list[i], n_scenarios)
        inst_Grasp.plot()
        
        startlocal = time.time()
        # '' for normal Local search, 'bis' for modified Local search
        X_Grasp, Y_Grasp, objf_Grasp = GRASP(inst_Grasp, p_list[i], Grasp_it, FLAG = 'bis')
        endlocal = time.time()
        comp_time_local = endlocal - startlocal
        cpu_time_Grasp.append(comp_time_local)
        print("GRASP finished")
        
        X_Grasp_bis, Y_Grasp_bis, objf_Grasp_bis = GRASP(inst_Grasp, p_list[i], Grasp_it, FLAG = 'bis')
        endlocal = time.time()
        comp_time_local = endlocal - startlocal
        cpu_time_Grasp_bis.append(comp_time_local)
        print("GRASP bis finished")
        
        startGurobi = time.time()
        sol = SolverExactCCP_stoch(inst_Grasp, p_list[i])
        Y_G, X_G, exact_solution, _, _ = sol.solve()
        endGurobi = time.time()
        comp_time_Gurobi = endGurobi - startGurobi
        cpu_time_Gurobi.append(comp_time_Gurobi)
        print("Exact solver finished")
          
        solutions_Grasp.append(objf_Grasp)
        solutions_Grasp_bis.append(objf_Grasp)
        solutions_Gurobi.append(exact_solution)

    # RESULTS
    print(f"computational time vector GRASP: {cpu_time_Grasp} s")
    print(f"computational time vector GRASP: {cpu_time_Grasp_bis} s")
    print(f"computational time vector Gurobi: {cpu_time_Gurobi} s")

    print("GRASP solutions: ", solutions_Grasp)
    print("GRASP solutions: ", solutions_Grasp_bis)
    print("Exact solutions: ", solutions_Gurobi)
    
    # PLOTS
    plt.plot(p_list, solutions_Grasp,c = 'royalblue', label = 'GRASP solutions')
    plt.plot(p_list, solutions_Grasp_bis, label = 'GRASP bis solutions')
    plt.plot(p_list, solutions_Gurobi, label = 'Exact solutions')
    plt.xlabel('N. of points')
    plt.ylabel('Value of the objective function')
    plt.title('GRASP solution vs. Exact solution')
    plt.legend(['GRASP solutions','GRASP bis solutions', 'Exact solutions'])
    plt.xticks(p_list)
    plt.show()

    plt.plot(p_list, cpu_time_Grasp, label = 'GRASP computational time')
    plt.plot(p_list, cpu_time_Grasp_bis, label = 'GRASP bis computational time')
    plt.plot(p_list, cpu_time_Gurobi, label = 'Exact computational time')
    plt.xlabel('N. of points')
    plt.ylabel('Computational time')
    plt.title('GRASP vs. Exact computational times')
    plt.legend(['GRASP comp. time', 'GRASP bis comp. time', 'Exact comp. time'])
    plt.xticks(p_list)
    plt.show()
    
          
    return