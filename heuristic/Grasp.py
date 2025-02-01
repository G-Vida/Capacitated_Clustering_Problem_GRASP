# -*- coding: utf-8 -*-
"""
Created on Wed May 17 17:40:32 2023

@author: user
"""

import numpy as np
import random
from solver.sub_solveCCP_stoch import Sub_SolverExactCCP_stoch
from solver.solveCCP_stoch import SolverExactCCP_stoch
import time

#def modified_kmean(inst, k):
def Modified_Kmean(inst, k, centroids_idx):
    
    n_points = inst.n_points
    centroids_cap = np.copy(inst.C)
    points_vector = -np.ones(n_points).astype(int)
    points_dist = np.copy(inst.d)
    
    points_w_mean = np.zeros(n_points)
    for i in range(n_points):
        points_w_mean[i] = np.mean(inst.w[i,:])
    
    for i in range(k):
        j = int(centroids_idx[i])
        points_vector[j] = j
        centroids_cap[j] = centroids_cap[j] - points_w_mean[j] 
    
    for i in range(k):
        c_idx = int(centroids_idx[i])
        j = 0 
        candidates_idx = np.argsort(points_dist[c_idx,:]) #si scelgono prima quelli con distanze più piccole
        
        while centroids_cap[c_idx] > 0 and j < len(points_vector):
            cand_idx = candidates_idx[j]
            if points_vector[cand_idx] == -1 and points_w_mean[cand_idx] <= centroids_cap[c_idx]:    # se sta nel cluster e non è in altri cluster decremento la capacità e lo aggiungo
                centroids_cap[c_idx] = centroids_cap[c_idx] - points_w_mean[cand_idx]  
                points_vector[cand_idx] = c_idx
            j = j + 1
    
    for ele in range(n_points):
        if points_vector[ele] == -1:
            elements = points_dist[np.ix_([ele], centroids_idx.astype(int))]
            centre = np.argmin(elements)  # takes only the distances between points and centroids
            points_vector[ele] = int(centroids_idx[centre]) # we don't care about capacity because we conseder the UNDERCAPACITED PROBLEM (offerta più della richiesta)


    return points_vector 


     
    

        


def Local_Search(inst, k):
    
    # Create initial partition
    n_points = np.copy(inst.n_points)
    
    points_vector = -np.ones(n_points).astype(int)
    
    centroids_idx = np.array(random.sample(list(range(n_points)), k)).astype(int)
        
    points_vector  = Modified_Kmean(inst, k, centroids_idx)
    
    X_tot = np.zeros((inst.n_points,inst.n_points)).astype(int)
    Y_tot = np.zeros(inst.n_points).astype(int)

    FLAG = True
    it = 0
    while FLAG and it < 15:
        
        X_tot = np.zeros((inst.n_points,inst.n_points)).astype(int)
        Y_tot = np.zeros(inst.n_points).astype(int)
        new_centroids_idx = -np.ones(k).astype(int)
        for i in range(k):
            sub_points = np.where(points_vector == centroids_idx[i])[0]
            
            sub_inst = inst.sub_instance(sub_points) #0 1 2 3
            sub_solve = Sub_SolverExactCCP_stoch(sub_inst, 1) # OTTIMIZZAZIONE CHE VUOLE QUELLO CON ECCESSO DI CAPACITA, PER FARLA GIRARE IL VINCOLO <= p DEVE ESSERE = p (= 1)
            _, _, _, better_cen, _ = sub_solve.solve() # 2       
            centroid = sub_points[better_cen] #9

            Y_tot[centroid] = 1
            
            new_centroids_idx[i] = centroid[0]
    
            
        
        points_vector = Modified_Kmean(inst,k, new_centroids_idx)
        
        
        for i in range(n_points):
            X_tot[i, points_vector[i]] = 1
        
        
        if np.array_equal(centroids_idx, new_centroids_idx):
            FLAG = False
        
        centroids_idx = np.copy(new_centroids_idx)
        
        it += 1
        
            
    return it, X_tot, Y_tot
         



'''
SE LA COMPUTAZIONE E' TROPPO PESANTE:
1) FARE L'OTTIMIZZAZIONE EV, perchè il worst case interessa un numeo ristretto di punti
2) Prima di fare Gurobi controllare se il centroide che abbiamo è quello di massima capacità
3)   MODIFICARE L'OTTIMIZZAZIONE STOCASTICA in modo che prenda solo i primi n per capacità come possibili Y => Con n che aumeta nel tempo
'''


# Faster and less accurate version of Local Search
def Local_Search_Bis(inst, k):

    # Create initial partition
    n_points = np.copy(inst.n_points)    
    points_vector = -np.ones(n_points).astype(int)
    centroids_idx = np.array(random.sample(list(range(n_points)), k)).astype(int)
        
    points_vector  = Modified_Kmean(inst, k, centroids_idx)
    
    X_tot = np.zeros((inst.n_points,inst.n_points)).astype(int)
    Y_tot = np.zeros(inst.n_points).astype(int)
    
    FLAG = True
    it = 0
    while FLAG and it < 15:
        
        X_tot = np.zeros((inst.n_points,inst.n_points)).astype(int)
        Y_tot = np.zeros(inst.n_points).astype(int)
        new_centroids_idx = -np.ones(k).astype(int)
        for i in range(k):
            
            sub_points = np.where(points_vector == centroids_idx[i])[0]
            sub_inst = inst.sub_instance(sub_points) #0 1 2 3
            sorted_idx = np.argsort(sub_inst.C)
            
            n = int(len(sub_points)*0.80)+1
            #n = int(len(sub_points)-2)
            excluded_idx = sorted_idx[:-n]       #We consider only two point and Gurobi will choose the best one 
            sub_solve = Sub_SolverExactCCP_stoch(sub_inst, 1, excluded_idx) # OTTIMIZZAZIONE CHE VUOLE QUELLO CON ECCESSO DI CAPACITA, PER FARLA GIRARE IL VINCOLO <= p DEVE ESSERE = p (= 1)

            _, _, _, better_cen, _ = sub_solve.solve() # 2  
            
            centroid = sub_points[better_cen] #9

            Y_tot[centroid] = 1
            
            new_centroids_idx[i] = centroid[0]

        points_vector = Modified_Kmean(inst,k, new_centroids_idx)

        
        for i in range(n_points):
            X_tot[i, points_vector[i]] = 1
        
        
        if np.array_equal(centroids_idx, new_centroids_idx):
            FLAG = False
        
        centroids_idx = np.copy(new_centroids_idx)
        
        it += 1
        
            
    return it, X_tot, Y_tot
         







def GRASP(inst, k, n_it = 5, FLAG = ''):
    np.random.seed(23)
    # First run
    if FLAG == '':
        _, X, Y = Local_Search(inst, k)
        objf_best =feval(inst, X, Y)
    else:
        _, X, Y = Local_Search_Bis(inst, k)
        objf_best =feval(inst, X, Y)
        
    n_it -= 1  #We have already done one iteration
    
    # We search for better solutions , changing the seed so the Local search will start differently   
    for i in range(n_it):
        np.random.seed(23+i)
        
        if FLAG == '':
            print("Grasp is at iteration: ", i)
            _, X, Y = Local_Search(inst, k)
            objf_new =feval(inst, X, Y)
            if objf_new < objf_best:   #Minimization => a lower value is better
                objf_best = objf_new
                
        else:
            print("Grasp bis is at iteration: ", i)
            _, X, Y = Local_Search_Bis(inst, k)
            objf_new =feval(inst, X, Y)
            if objf_new < objf_best:   #Minimization => a lower value is better
                objf_best = objf_new
    
    return X, Y, objf_best





def feval(inst, Xopt, Yopt):
    points = range(inst.n_points)
    scenarios = range(inst.n_scenarios)
            
    fun = sum(inst.d[i, j] * Xopt[i, j] for i in points for j in points) 
    fun += inst.l/inst.n_scenarios * sum(sum(max(0, (sum(inst.w[i, s] * Xopt[i,j] for i in points) - inst.C[j]*Yopt[j])) for j in points) for s in scenarios)
    return fun
    
    
    
    