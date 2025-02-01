# -*- coding: utf-8 -*-
import time
import gurobipy as grb
import numpy as np

class SolverExactCCP_stoch:
    def __init__(self, inst, p):
        self.inst = inst
        self.model = grb.Model('ccp')
        self.n_points = self.inst.n_points
        self.n_scenarios = self.inst.n_scenarios


        points = range(self.n_points)
        scenarios = range(self.n_scenarios)

        # Y = 1 if cluster is the center
        self.Y = self.model.addVars(
            self.n_points,
            vtype=grb.GRB.BINARY,
            name='Y'
        )
        # X = 1 ...
        self.X = self.model.addVars(
            self.n_points, self.n_points,
            vtype=grb.GRB.BINARY,
            name='X'
        )
        self.Z = self.model.addVars(        #Vector of slack variables (the objective function is not linear)
            self.n_points, self.n_scenarios,
            vtype=grb.GRB.CONTINUOUS,       #The slack variables are continuous
            lb=0.0,                         #They have to be greater than 0
            name='Z'                        #Slack variables  => one for each cluster, one for each scenario 
        )

        # set objective function
        expr = sum(
            self.inst.d[i, j] * self.X[i, j] for i in points for j in points
        )
        expr += self.inst.l / self.n_scenarios * sum( self.Z[j, s] for j in points for s in scenarios)      #Is the expected value (using row montecarlo that is 1/N)

        self.model.setObjective(
            expr,
            grb.GRB.MINIMIZE          #gurobi solves => We get the best possible solution thet would give us the objective function equal to 0 wiyhput the constraints!
        )

        # add constraints
        self.model.addConstrs(
            ( grb.quicksum(self.X[i,j] for j in points) == 1 for i in points),
            name="x_assigned"
        )
        self.model.addConstr(
            grb.quicksum(self.Y[i] for i in points) <= p,     # == p
            name="p_upbound"
        )
        self.model.addConstrs(
            (self.X[i,j] <= self.Y[j] for i in points for j in points),
            name="linkXY"                           #We have to link x and y
        )
        self.model.addConstrs(    #We want to "pay" if the demand is not satisfied
            (self.Z[j, s] >= grb.quicksum( self.inst.w[i, s] * self.X[i,j] for i in points ) - self.inst.C[j]*self.Y[j] for j in points for s in scenarios), 

            name="linkZYX"       #We have to add a new contraint for the slack variables
        )
        self.model.update()

    def solve(self, lp_name=None, gap=None, time_limit=None, verbose=False):
        if gap:
            self.model.setParam('MIPgap', gap)
        if time_limit:
            self.model.setParam(grb.GRB.Param.TimeLimit, time_limit)
        if verbose:
            self.model.setParam('OutputFlag', 1)
        else:
            self.model.setParam('OutputFlag', 0)
        if lp_name:
            self.model.write(f"./logs/{lp_name}_stoch.lp")
        self.model.setParam('LogFile', './logs/gurobi_stoch.log')

        start = time.time()
        self.model.optimize()
        end = time.time()
        comp_time = end - start
        #print(f"computational time Gurobi: {comp_time} s")

        #INDEXES
        sol = []   
        if self.model.status == grb.GRB.Status.OPTIMAL:
            for j in range(self.n_points):
                if self.Y[j].X > 0.5:
                    sol.append(j)
                    
        all_vars = self.model.getVars()
        values = self.model.getAttr("X", all_vars)
        obj_value_stoch = self.model.getObjective().getValue()

        arr = np.array(values)
        Yminimizer = arr[0:self.n_points]
        low = self.n_points
        up = self.n_points*self.n_points + self.n_points
        Xminimizer = np.reshape(arr[low:up], (self.n_points,self.n_points))

        return Yminimizer, Xminimizer, obj_value_stoch, sol, comp_time




        # NON CREIAMO ATTRIBUTI Xopt Yopt PERCHE' NELLA STABILITY ABBIAMO X,Y DATI DA ALTROVE
    def evaluate_f(self, Xopt, Yopt):
        points = range(self.n_points)
        scenarios = range(self.n_scenarios)
            
        fun = sum(self.inst.d[i, j] * Xopt[i, j] for i in points for j in points) 
        fun += self.inst.l/self.n_scenarios * sum(sum(max(0, (sum(self.inst.w[i, s] * Xopt[i,j] for i in points) - self.inst.C[j]*Yopt[j])) for j in points) for s in scenarios)
        return fun
    
    
    
    
    


                    