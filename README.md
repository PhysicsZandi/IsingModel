# Simulation of the Ising model with networks

In this repository, you will find a python implementation of the Ising model and its network analysis. We chose 4 different kinds of lattices: 1d, 2d square, 2d triangular and 2d hexagonal.

We numerically computed the theoretical solutions of the Ising model (using scipy) and their plots are in the folder [exact_solutions](https://github.com/PhysicsZandi/IsingModel/blob/main/plot/exact_solutions).

In [animation](https://github.com/PhysicsZandi/IsingModel/blob/main/plot/animation), you will find a gif animation describing the time evolution of the lattice when it is at equilibrium at different temperatures.

In [graph](https://github.com/PhysicsZandi/IsingModel/blob/main/plot/graph), you will find a plot of the Ising lattice. In [weighted_graph](https://github.com/PhysicsZandi/IsingModel/blob/main/plot/weighted_graph), you will find the weighted graph built from the two points correlation function.

Finally, the analysis of the lattice (energy, magnetisation, specific heat, susceptibility, free energy, entropy, correlation length) and the weighted graph (betweenness centrality, clustering coefficient, density, diameter, disparity, shortest path) can be found in [phase_transition](https://github.com/PhysicsZandi/IsingModel/blob/main/plot/phase_transition).

# How to compile it

For MacOs users (otherwise you need to modify where the os package is used):
1. For [exact_solutions.py](https://github.com/PhysicsZandi/IsingModel/blob/main/src/exact_solutions.py), you need to choose which kind of graph and compile it with python, uncommenting run().
2. For [ising.py](https://github.com/PhysicsZandi/IsingModel/blob/main/src/exact_solutions.py), you need to choose which kind of graph, the size of the graph, the evolution time to have equilibrium, the average time to make averages and compile it with python, uncommenting which plot you want.

Since computing times are very long, we chose to implement pickle to save values for each temperature and then make the plot with saved data.

# Relation

The relation can be found in [relation.pdf](https://github.com/PhysicsZandi/IsingModel/blob/main/relation.pdf). This project was made together with Federico Tonetto.