# Capacitated Clustering Problem: EV Optimization Problem and Grasp Heuristic
## Project for "Numerical optimization for large scale problems and Stochastic Optimization" (Course 2022/2023)

Collaboration with: https://github.com/GabPir
<br>

In this project, the CCP - Capacitated Clustering Problem is addressed. The goal is to find an optimal clustering given a capacity constraint on the centroids and a stochastic weight for each individual cluster element (e.g., demand for a service, supply of a service in a spatial context).
The objective function to be minimized considers both the distance between points and their assigned cluster and the capacity constraint (the constraint is relaxed by introducing a penalty even when there is unused capacity).
<br>
![image](https://github.com/user-attachments/assets/103421b8-46a7-487c-aae4-71b7785b4aef)
<br>
The problem is a two-stage problem, where in the first stage, the centroids are selected, and in the second stage, the assignments to these centroids are made. Between the first and second stage, the stochastic weight factors w_i (capacity cost) of individual points are observed. The algorithm operates by simulating a finite number of scenarios, using a discrete expected value approach.
<br><br>
For optimization, the Python package Gurobi© (https://www.gurobi.com/) was used.
<br>
### Pt1: Expected Value Optimization
Uncertainty is addressed by considering the expected value (E.V.) formulation, where the stochastic weights are replaced by the expected value of the weights across different scenarios:
<br>
![immagine](https://github.com/user-attachments/assets/ce0ccdbd-1953-46e4-923a-d6ffcfc0c03c)

<br>The results obtained with the E.V. and Stochastic approaches are as follows:

![immagine](https://github.com/user-attachments/assets/beb12d66-d988-4b6d-b333-7dbc4867aa12)
![immagine](https://github.com/user-attachments/assets/7f162958-7d6e-4e7d-83e1-bba9f4d0f081)

<br><br>
### Pt2: Heuristic GRASP method
Larger-scale problems often require heuristic approaches. A Greedy Randomized Adaptive Search Procedure (GRASP) was used to cluster the data and solve the optimization problem. The heuristic consists of several Local Searches based on a variant of the K-means algorithm adapted for the constrained CCP stochastic problem. An second version of the heuristic, with lower computational cost, involves selecting the new centroids only from the top 20% of the most promising points.
<br>
The pseudocode is:
<br>
![immagine](https://github.com/user-attachments/assets/1e8af8cb-53dd-4140-9672-607d6fe5d276)
<br>
The heuristic generally yields higher values for the objective function but lower computational costs (depending on the number of clusters):
![immagine](https://github.com/user-attachments/assets/5b741eca-71d4-4840-a839-ec4da874ee9c)
<br>
<br>
Furthermore, it was possible to evaluated how the computational cost and performance varied by increasing the weight of the stochastic factor in the second stage (capacity mismatch in the clusters) in the optimization problem, achieving remarkable results from the use of the heuristic:
<br>
![immagine](https://github.com/user-attachments/assets/7eb67cb3-1fe0-4bc0-bb36-e11056fd302a)
<br><br>
Additionally, it is possible to parallelize the execution of the optimization subproblems, thereby enhancing the performance of the GRASP algorithm:<br>
![immagine](https://github.com/user-attachments/assets/60b6bda9-7327-4ff2-8005-db7dc5bdb87e)
<br>
<br>

