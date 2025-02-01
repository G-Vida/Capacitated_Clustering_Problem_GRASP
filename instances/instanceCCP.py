# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class InstanceCCP:
    def __init__(self, n_points = -1, n_scenarios = -1, instance = None):
        if n_points == -1 and n_scenarios == -1:
            self.n_points = instance.n_points
            self.n_scenarios = instance.n_scenarios 
            self.xy = np.copy(instance.xy)
            self.center = np.copy(instance.center)
            self.coord = np.copy(instance.coord)
            self.d = np.copy(instance.d)
            self.C = np.copy(instance.C)
            self.w = np.copy(instance.w)
            self.l = instance.l
        
        else:
            self.n_points = n_points
            self.n_scenarios = n_scenarios
        
            n_clus = np.random.randint(2, int(self.n_points)/5)
        
            self.xy, self.center = make_blobs(n_samples = n_points, n_features = 2, centers = n_clus, 
                                          center_box=(-40.0, 40.0), shuffle=True, random_state=23, cluster_std=6)

            self.coord = []     
            for i in range(n_points):
                self.coord.append([self.xy[i,0], self.xy[i,1]])

            self.d = distance_matrix(self.coord, self.coord)          # between each couple of points

            self.C = np.random.randint(low=60, high=100, size=n_points)*0.99         #random integer capacity
        
            self.w = np.random.lognormal(mean=2.3, sigma=0.8, size = (n_points, n_scenarios)) #Lognormal distirb., because we want positive weights
        
            self.l = 1         #LAMBDA
        
    def sampled(self, indexes):
        self.n_scenarios = len(indexes)
        self.w = self.w[:,indexes]
    
    
    def sub_instance(self, points):
        sub_instance = InstanceCCP(instance = self)
        sub_instance.w = np.copy(self.w[points,:])
        sub_instance.n_points = len(points) 
        sub_instance.d = np.copy(self.d[np.ix_(points, points)])
        sub_instance.C = np.copy(self.C[points])

        # VERIFICARE CHE MANTENGA L?ORDINE DEGLI INDICI: cioè l'indice 0 sarà points[0]
        
        return sub_instance   # tornare l'istanza per poi fare l'ottimizzazione e tornare qual'è il centroide ottimo
        
    
    
    def plot(self):
        plt.scatter(self.xy[:, 0], self.xy[:, 1], c=self.center)
        plt.show()
        