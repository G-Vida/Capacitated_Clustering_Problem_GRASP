# -*- coding: utf-8 -*-
from instances.instanceCCP import InstanceCCP
from solver.solveCCP_stoch import SolverExactCCP_stoch
from solver.solveCCP_ev import SolverExactCCP_ev
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import random
import itertools
import statistics as s

def ValueSS(inst, p, FLAG = -1):

    sol_s = SolverExactCCP_stoch(inst, p)    
    Ymin_s, Xmin_s, obj_value_s, cluster_s, comp_time_s = sol_s.solve()

    sol_ev = SolverExactCCP_ev(inst, p)    
    Ymin_ev, Xmin_ev, obj_value_ev, cluster_ev, comp_time_ev = sol_ev.solve()

    obj_value_mean = sol_ev.evaluate_meanf(Xmin_ev, Ymin_ev)

    value = abs(obj_value_s - obj_value_mean)  #We want to minimize the objective function
    
    
    # PLOTS---------------------------------------------------------------------------------
    if FLAG == -1:     #We do the plots

        N_POINTS = inst.n_points
        low_x = min(inst.xy[:, 0])
        low_y = min(inst.xy[:, 1])
        high_x = max(inst.xy[:, 0])
        high_y = max(inst.xy[:, 1])
        color_it = cm.rainbow(np.linspace(0, 1, p))
        
        
        # PLOT 1
        plt.figure() 
        plt.xlim([low_x-5, high_x+5])
        plt.ylim([low_y-5, high_y+5])
        
        color1 = iter(color_it)
        for i in range(N_POINTS):
            coord_x = []
            coord_y = []
            if Ymin_ev[i] ==1:     # i is a centroid
                c = next(color1)    
                for j in range(N_POINTS):     #We seek for all the points that are connected to the centroid i
                    if Xmin_ev[j, i] == 1:
                        coord_x.append(inst.xy[j, 0])
                        coord_y.append(inst.xy[j, 1])
                plt.scatter(coord_x, coord_y, c = c)
                plt.plot(inst.xy[i, 0], inst.xy[i, 1], '+', c = 'black', alpha=1, linewidth=5.0)
        plt.title('E.V. solution')
        plt.show()
    
    
        # PLOT 2
        plt.figure() 
        plt.xlim([low_x-5, high_x+5])
        plt.ylim([low_y-5, high_y+5])
        
        color2 = iter(color_it)
        for i in range(N_POINTS):
            coord_x = []
            coord_y = []
            if Ymin_s[i] ==1:
                c = next(color2)
                for j in range(N_POINTS):
                    if Xmin_s[j, i] == 1:
                        coord_x.append(inst.xy[j, 0])
                        coord_y.append(inst.xy[j, 1])
                plt.scatter(coord_x, coord_y, c = c)
                plt.plot(inst.xy[i, 0], inst.xy[i, 1], '+', c = 'black', alpha=1, linewidth=5.0)
        plt.title('Stochastic solution')
        plt.show()

    return value, obj_value_mean, obj_value_s





def Test_Scenarios_VSS(n_points, p):                    
    n_scen_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
 
        
    inst = []
    n_instances = 20
    #We generate 20 instances (20 different dispositions of points) without scenarios that will be added later
    for j in range(n_instances):
        inst.append(InstanceCCP(n_points, 1))
    
    VSS_mean = [0]*len(n_scen_list)
    objf_ev_mean = [0]*len(n_scen_list)   #Results of ev evalueated with stochastic scenarios
    objf_stoch_mean = [0]*len(n_scen_list)   #Results of stoch
    
    for j in range(n_instances):   # We test each instance with different scenarios
                
        for i in range(len(n_scen_list)):        #We consider different number of scenarios 
            inst[j].w = np.random.lognormal(mean=2.3, sigma=0.8, size = (n_points, n_scen_list[i]))
            VSS, obj_value_ev, obj_value_s = ValueSS(inst[j], p, FLAG = 1)    # FLAG = 1 meand that we don't want the plots
            
            # We consider directly the sample mean
            VSS_mean[i] += 1/n_instances*VSS   
            objf_ev_mean[i] += 1/n_instances*obj_value_ev
            objf_stoch_mean[i] += 1/n_instances*obj_value_s
            
        
    # PLOT 1
    low_y = 0
    low_x = 0
    high_y = max(objf_ev_mean)
    high_x = max(n_scen_list)
    plt.ylim([low_y, high_y+50])
    plt.xlim([low_x, high_x+2])
    plt.plot(n_scen_list, objf_ev_mean, alpha=1, label = 'Obj.f. value of E.V.')
    plt.plot(n_scen_list, objf_stoch_mean, alpha=1, label = 'Obj.f. value of Stoch.')
    plt.legend(['Obj.f. value of E.V.','Obj.f. value of Stoch.']) 
    plt.xlabel('Number of scenarios')
    plt.ylabel('Values of the objective functions (sample mean)')
    plt.title('Values of the ob.f. for an increasing n. of scenarios')
    plt.show()

    
    # PLOT 2
    low_y = 0
    low_x = 0
    high_y = max(VSS_mean)
    high_x = max(n_scen_list)
    plt.ylim([low_y, high_y+10])
    plt.xlim([low_x, high_x+2])
    plt.plot(n_scen_list, VSS_mean, label = 'VSS')
    plt.xlabel('Number of scenarios')
    plt.ylabel('VSS (sample mean)')
    plt.title('VSS for an increasing number of scenarios')
    plt.show()
                  
    return VSS_mean




def Test_Sigma_VSS(n_points, n_scenarios, p):
    
    # 0.8 is the original variance
    sigma_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
    
    inst = []
    n_instances = 10
    #We generate 20 instances (20 different dispositions of points)
    for j in range(n_instances):
        inst.append(InstanceCCP(n_points, n_scenarios))
    
    VSS_vector = [0]*len(sigma_list)
    objf_ev_vector = [0]*len(sigma_list)
    objf_stoch_vector = [0]*len(sigma_list)
    
    for j in range(n_instances):   # We test each instance with different variance for the weights
        
        for i in range(len(sigma_list)):
            inst[j].w = np.random.lognormal(mean=2.3, sigma=sigma_list[i], size = (n_points, n_scenarios))
            VSS, obj_value_ev, obj_value_s = ValueSS(inst[j], p, FLAG = 1)
            
            # We consider directly the sample mean
            VSS_vector[i] += 1/n_instances*VSS    
            objf_ev_vector[i] += 1/n_instances*obj_value_ev
            objf_stoch_vector[i] += 1/n_instances*obj_value_s
    
    
    # PLOT 1
    low_y = 0
    low_x = 0
    high_y = max(objf_ev_vector)
    plt.ylim([low_y, high_y+50])
    plt.xlim([low_x, 1.45])
    plt.plot(sigma_list, objf_ev_vector, alpha=1, label = 'Obj.f. value of E.V.')
    plt.plot(sigma_list, objf_stoch_vector, alpha=1, label = 'Obj.f. value of Stoch.')
    plt.legend(['Obj.f. value of E.V.','Obj.f. value of Stoch.']) 
    plt.xlabel('Parameter \u03C3')
    plt.ylabel('Values of the objective functions (sample mean)')
    plt.title('Values of the ob.f. for an increasing \u03C3')
    plt.show()
    
    
    # PLOT 2
    low_y = 0
    low_x = 0
    high_y = max(VSS_vector)
    plt.ylim([low_y, high_y+2])
    plt.xlim([low_x, 1.45])
    plt.plot(sigma_list, VSS_vector, label = 'VSS')
    plt.xlabel('Parameter \u03C3')
    plt.ylabel('VSS (sample mean)')
    plt.title('VSS for an increasing \u03C3 ')
    plt.show()
        
        
    return VSS_vector



def Test_Lambda_VSS(n_points, n_scenarios, p):
    
    # 0.8 is the original variance
    l_list = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    
    inst = []
    n_instances = 10
    #We generate 20 instances (20 different dispositions of points)
    for j in range(n_instances):
        inst.append(InstanceCCP(n_points, n_scenarios))
    
    VSS_vector = [0]*len(l_list)
    objf_ev_vector = [0]*len(l_list)
    objf_stoch_vector = [0]*len(l_list)
    
    for j in range(n_instances):   # We test each instance with different lambda
        print(j)
        for i in range(len(l_list)):
            inst[j].l = l_list[i]
            VSS, obj_value_ev, obj_value_s = ValueSS(inst[j], p, FLAG = 1)
            
            # We consider directly the sample mean
            VSS_vector[i] += 1/n_instances*VSS    
            objf_ev_vector[i] += 1/n_instances*obj_value_ev
            objf_stoch_vector[i] += 1/n_instances*obj_value_s
    
    
    # PLOT 1
    low_y = 0
    low_x = 0
    high_y = max(objf_ev_vector)
    plt.ylim([low_y, high_y+50])
    plt.xlim([low_x, 2.05])
    plt.plot(l_list, objf_ev_vector, alpha=1, label = 'Obj.f. value of E.V.')
    plt.plot(l_list, objf_stoch_vector, alpha=1, label = 'Obj.f. value of Stoch.')
    plt.legend(['Obj.f. value of E.V.','Obj.f. value of Stoch.']) 
    plt.xlabel('Parameter \u03BB')
    plt.ylabel('Values of the objective functions (sample mean)')
    plt.title('Values of the ob.f. for incrising \u03BB')
    plt.show()
    
    
    # PLOT 2
    low_y = 0
    low_x = 0
    high_y = max(VSS_vector)
    plt.ylim([low_y, high_y+2])
    plt.xlim([low_x, 2.05])
    plt.plot(l_list, VSS_vector, label = 'VSS')
    plt.xlabel('Parameter \u03BB')
    plt.ylabel('VSS (sample mean)')
    plt.title('VSS for an incrising \u03BB')
    plt.show()
        
        
    return VSS_vector
