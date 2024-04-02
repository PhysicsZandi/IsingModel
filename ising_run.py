import networkx as nx
from ising_class import IsingNetwork, IsingEnsemble

def equilibrium():
    number_of_rows = 100
    number_of_columns = 100
    graph = nx.grid_2d_graph(number_of_rows, number_of_columns)

    temperatures = [0.2 + 0.2 * i for i in range(20)]
    for temperature in temperatures:
        time_evolution = 1000
        ising = IsingNetwork(graph, temperature)
        ising.reach_equilibrium(time_evolution)
        ising.plot_physics(temperature)

def sequence():
    number_of_rows = 100
    number_of_columns = 100
    graph = nx.grid_2d_graph(number_of_rows, number_of_columns)

    temperatures = [0.2 + 0.2 * i for i in range(20)]
    for temperature in temperatures:
        time_evolution = 10
        ising = IsingNetwork(graph, temperature)
        ising.reach_equilibrium(time_evolution)
        ising.plot_network_animation(temperature)

def phase_transition():
    number_of_rows = 10
    number_of_columns = 10
    graph = nx.grid_2d_graph(number_of_rows, number_of_columns)

    number_of_iterations = 3000
    initial_temperature = 0.2
    final_temperature = 4
    number_of_steps = 20
    ising = IsingEnsemble(number_of_iterations, initial_temperature, final_temperature, number_of_steps, graph)
    ising.phase_transitions()
    ising.plot_physics(number_of_rows)
    ising.get_critical_temperature()

#equilibrium()
#sequence()
phase_transition()

####################################################################################
    
#number_of_nodes = 1000
#probability = 0.4
#number_of_rows = 33
#dimension = 4
#number_of_columns = 33

#graph = nx.cycle_graph(number_of_nodes)
#graph = nx.erdos_renyi_graph(number_of_nodes, probability)
#graph = nx.grid_2d_graph(number_of_rows, number_of_columns)
#graph = nx.triangular_lattice_graph(number_of_rows, number_of_columns)
#graph = nx.hexagonal_lattice_graph(number_of_rows, number_of_columns)
#graph = nx.hypercube_graph(dimension)

# Graphical representation at a fixed temperature
#temperature = 2.26
#time_evolution = 50
#ising = IsingNetwork(graph, temperature)
#ising.reach_equilibrium(time_evolution)
#ising.plot_physics()
#ising.plot_network(20)
#ising.plot_network_animation("2.26")

#temperatures = [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6, 3.8, 4]
#
#for temperature in temperatures:
#    time_evolution = 100
#    ising = IsingNetwork(graph, temperature)
#    ising.reach_equilibrium(time_evolution)
#    #ising.plot_physics()
#    #ising.plot_network(20)
#    ising.plot_network_animation(temperature)

# Phase transitions for different temperatures
#number_of_iterations = 1000
#initial_temperature = 1
#final_temperature = 4
#number_of_steps = 20
#ising = IsingEnsemble(number_of_iterations, initial_temperature, final_temperature, number_of_steps, graph)
#ising.phase_transitions()
#ising.plot_physics("33x33")
#ising.get_critical_temperature()