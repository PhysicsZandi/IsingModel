import networkx as nx
from ising_class import IsingNetwork, IsingPhaseTransition
import os

def phase_transition(side):
    graph = nx.grid_graph((side, side))
    temperatures = [1.0, 1.5, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 3.0, 3.5, 4.0]
    ising = IsingPhaseTransition(graph, temperatures)
    time = 1500
    average_time = 1000
    ising.phase_transition(time, average_time)
    ising.plot_physics()

sides = [16, 32, 64, 128]

#for side in sides:
#    phase_transition(side) 

def sequence(side):    
    graph = nx.grid_2d_graph(side, side)
    temperatures = [1.0, 1.5, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 3.0, 3.5, 4.0]
    for temperature in temperatures:
        ising = IsingNetwork(graph, temperature)
        ising.initialise_spins_hot()
        time_evolution = 100
        ising.evolution(time_evolution)
        ising.plot_network_animation()

sides = [16, 32, 64, 128]

#for side in sides:
#    sequence(side)

def notify(title, text):
    os.system("""
              osascript -e 'display notification "{}" with title "{}"'
              """.format(text, title))

notify("Python", "Finished executing")