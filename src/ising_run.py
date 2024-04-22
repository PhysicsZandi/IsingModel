from ising_class import IsingNetwork, IsingPhaseTransition
from tqdm import tqdm
import os

def equilibrium_save_graph(kind_of_graph, sides):
    for graph in kind_of_graph:
        for side in sides:
            ising = IsingNetwork(graph, side)
            ising.save_graph()

def equilibrium_save_data(kind_of_graph, sides):
    for graph in kind_of_graph:
        for side in sides:
            for t in tqdm(temperatures):
                ising = IsingNetwork(graph, side)
                ising.load_graph()
                ising.set_j(j=1)
                ising.set_b(b=0)
                ising.set_temperature(t)
                ising.initialise_spins_hot()
                ising.evolution(time=4000)
                ising.save_data()
            
def equilibrium_save_data_random(kind_of_graph, sides):
    for graph in kind_of_graph:
        for side in sides:
            for t in tqdm(temperatures):
                ising = IsingNetwork(graph, side)
                ising.load_graph()
                ising.set_j(j=1)
                ising.set_b(b=0)
                ising.set_temperature(t)
                ising.initialise_spins_hot()
                ising.evolution_random(time=2000000)
                ising.save_data()
        
def equilibrium_plot_data(kind_of_graph, sides):
    for graph in kind_of_graph:
        for side in sides:
            for t in tqdm(temperatures):
                ising = IsingNetwork(graph, side)
                ising.load_graph()
                ising.set_temperature(t)
                ising.load_data()
                ising.plot_physics()
  
def phase_transition_save_data(kind_of_graph, sides):
    for graph in kind_of_graph:
        for side in sides:
            ising = IsingPhaseTransition(graph, side)
            ising.set_temperatures(temperatures)
            ising.phase_transition(average_time=1000)
            ising.save_data()

def phase_transition_save_data_random(kind_of_graph, sides):
    for graph in kind_of_graph:
        for side in sides:
            ising = IsingPhaseTransition(graph, side)
            ising.set_temperatures(temperatures)
            ising.phase_transition(average_time=100000)
            ising.save_data()

def phase_transition_plot_data(kind_of_graph, sides):
    for graph in kind_of_graph:
        for side in sides:
            ising = IsingPhaseTransition(graph, side)
            ising.set_temperatures(temperatures)
            ising.load_data()
            ising.plot_physics() 


######

temperatures = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00,
                1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45, 1.50,
                1.55, 1.60, 1.65, 1.70, 1.75, 1.80, 1.85, 1.90, 1.95, 2.00,
                2.05, 2.10, 2.15, 2.20, 2.25, 2.30, 2.35, 2.40, 2.45, 2.50,
                2.55, 2.60, 2.65, 2.70, 2.75, 2.80, 2.85, 2.90, 2.95, 3.00,
                3.05, 3.10, 3.15, 3.20, 3.25, 3.30, 3.35, 3.40, 3.45, 3.50,
                3.55, 3.60, 3.65, 3.70, 3.75, 3.80, 3.85, 3.90, 3.95, 4.00,
                4.05, 4.10, 4.15, 4.20, 4.25, 4.30, 4.35, 4.40, 4.45, 4.50,
                4.55, 4.60, 4.65, 4.70, 4.75, 4.80, 4.85, 4.90, 4.95, 5.00]

#kind_of_graph = ["1d"]
#sides = [1000, 10000, 20000, 32000, 50000, 64000, 100000, 128000]

#kind_of_graph = ["2d_square", "2d_triangular", "2d_hexagonal"]
#sides = [10, 20, 32, 50, 64, 100, 128]

#kind_of_graph = ["4d"]
#sides = [10, 20] 

#equilibrium_save_data(kind_of_graph, sides)
#equilibrium_plot_data(kind_of_graph, sides)
#phase_transition_save_data(kind_of_graph, sides)
#phase_transition_plot_data(kind_of_graph, sides)

def notify(title, text):
    os.system("""
              osascript -e 'display notification "{}" with title "{}"'
              """.format(text, title))

notify("Python", "Finished executing")
