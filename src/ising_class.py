import matplotlib.pyplot as plt
from tqdm import tqdm
import networkx as nx
import numpy as np
import pickle
import os
from num_int import ExactSolutionIsing

class IsingNetwork:
    def __init__(self, kind_of_graph, side):
        self.kind_of_graph = kind_of_graph
        self.side = side

        self.path_data = './data'
        if not os.path.exists(self.path_data):
            os.mkdir(self.path_data)
        self.path_data = './data/data_graph'
        if not os.path.exists(self.path_data):
            os.mkdir(self.path_data)
        self.path_data = './data/data_graph/' + self.kind_of_graph
        if not os.path.exists(self.path_data):
            os.mkdir(self.path_data)

        self.path_data_eq = './data/data_equilibrium'
        if not os.path.exists(self.path_data_eq):
            os.mkdir(self.path_data_eq)
        self.path_data_eq = './data/data_equilibrium/' + self.kind_of_graph
        if not os.path.exists(self.path_data_eq):
            os.mkdir(self.path_data_eq)

        self.path_plot = './plot'
        if not os.path.exists(self.path_plot):
            os.mkdir(self.path_plot)
        self.path_plot = './plot/plot_equilibrium'
        if not os.path.exists(self.path_plot):
            os.mkdir(self.path_plot)
        self.path_plot = './plot/plot_equilibrium/' + self.kind_of_graph
        if not os.path.exists(self.path_plot):
            os.mkdir(self.path_plot)

    # Save graph
    def save_graph(self):
        if self.kind_of_graph == "1d":
            self.graph = nx.cycle_graph(self.side)
        elif self.kind_of_graph == "2d_square":
            self.graph = nx.grid_2d_graph(self.side, self.side, periodic=True)
        elif self.kind_of_graph == "2d_triangular":
            self.graph =  nx.triangular_lattice_graph(self.side, self.side, periodic=True)
        elif self.kind_of_graph == "2d_hexagonal":
            self.graph =  nx.hexagonal_lattice_graph(self.side, self.side, periodic=True)
        elif self.kind_of_graph == "4d":
            self.graph =  nx.grid_graph([self.side, self.side, self.side, self.side])
        self.number_of_nodes = self.graph.number_of_nodes()

        graph_path = self.path_data + '/side' + str(self.side) + '_graph.pickle'
        with open(graph_path, 'wb') as f:
            pickle.dump(self.graph, f)
        
        self.nodes = self.graph.nodes()
        nodes_path = self.path_data + '/side' + str(self.side) + '_nodes.pickle'
        with open(nodes_path, 'wb') as f:
            pickle.dump(self.nodes, f)
            
        self._calculate_neighbours()
        neighbours_path = self.path_data + '/side' + str(self.side) + '_neighbours.pickle'
        with open(neighbours_path, 'wb') as f:
            pickle.dump(self.neighbours, f)

    # Load graph
    def load_graph(self):
        graph_path = self.path_data + '/side' + str(self.side) + '_graph.pickle'
        with open(graph_path, 'rb') as f:
            self.graph = pickle.load(f)
        self.number_of_nodes = self.graph.number_of_nodes()

        nodes_path = self.path_data + '/side' + str(self.side) + '_nodes.pickle'
        with open(nodes_path, 'rb') as f:
            self.nodes = pickle.load(f)
            
        neighbours_path = self.path_data + '/side' + str(self.side) + '_neighbours.pickle'
        with open(neighbours_path, 'rb') as f:
            self.neighbours = pickle.load(f)

    # Initialise spins
    def initialise_spins_hot(self):
        self.spins = np.random.choice([-1, 1], self.number_of_nodes)
        self._calculate_magnetisation()
        self._calculate_energy()

    def initialise_spins_cold(self):
        self.spins = np.full(self.number_of_nodes, 1)
        self._calculate_magnetisation()
        self._calculate_energy()
        
    # Calculate physical quantities
    def _calculate_magnetisation(self):
        self.magnetisation = sum(self.spins)

    def _calculate_energy(self):
        neighbour_spins = 0
        for i in range(self.number_of_nodes):
            for j in self.neighbours[i]:
                neighbour_spins += self.spins[i] * self.spins[j]
        self.energy = - self.j * neighbour_spins / 2 + self.b * sum(self.spins)

    def _calculate_neighbours(self):
        self.neighbours = []
        list_of_nodes = list(self.nodes)
        for node in tqdm(self.nodes):
            indices_of_neighbours = []
            for neighbour_of_node in self.graph.neighbors(node):
                j = list_of_nodes.index(neighbour_of_node)
                indices_of_neighbours.append(j)
            self.neighbours.append(indices_of_neighbours)

    # Set physical quantities   
    def set_temperature(self, temperature):
        self.temperature = temperature

    def set_j(self, j=1):
        self.j = j

    def set_b(self, b=0):
        self.b = b

    # Metropolis algorithm scrools all the spins to flip
    def evolution(self, time):
        self.energies = []
        self.magnetisations = []
        self._calculate_energy()
        for _ in tqdm(range(time)):
            for i in range(self.number_of_nodes):
                neighbour_spins = 0
                for j in self.neighbours[i]:
                    neighbour_spins += self.spins[i] * self.spins[j]
                delta_energy = 2.0 * self.j * neighbour_spins + 2 * self.b * neighbour_spins
                transition_probability = np.exp(- delta_energy / self.temperature)
                if delta_energy < 0 or transition_probability > np.random.rand():
                    self.spins[i] *= -1
                    self.energy += delta_energy
            self.energies.append(self.energy)
            self._calculate_magnetisation()
            self.magnetisations.append(self.magnetisation)
    
    # Glauber algorithm chooses randomly what spin to flip
    def evolution_random(self, time):
        self.energies = []
        self.magnetisations = []
        self._calculate_energy()
        choices = np.random.choice([i for i in range(self.number_of_nodes)], time)
        for n in tqdm(range(time)):
            i = choices[n]
            neighbour_spins = 0
            for j in self.neighbours[i]:
                neighbour_spins += self.spins[i] * self.spins[j]
            delta_energy = 2.0 * self.j * neighbour_spins + 2 * self.b * neighbour_spins
            transition_probability = np.exp(- delta_energy / self.temperature)
            if delta_energy < 0 or transition_probability > np.random.rand():
                self.spins[i] *= -1
                self.energy += delta_energy
            self.energies.append(self.energy)
            self._calculate_magnetisation()
            self.magnetisations.append(self.magnetisation)

    # Save data
    def save_data(self):
        energy_path = self.path_data_eq + '/side' + str(self.side) + '_temp' + str(float(self.temperature))[-5:] + '_energy.pickle'
        with open(energy_path, 'wb') as f:
            pickle.dump(self.energies, f)
        
        magnetisation_path = self.path_data_eq + '/side' + str(self.side) + '_temp' + str(float(self.temperature))[-5:] + '_magnetisation.pickle'
        with open(magnetisation_path, 'wb') as f:
            pickle.dump(self.magnetisations, f)

    # Load data
    def load_data(self):
        energy_path = self.path_data_eq + '/side' + str(self.side) + '_temp' + str(float(self.temperature))[-5:] + '_energy.pickle'
        with open(energy_path, 'rb') as f:
            self.energies = pickle.load(f)
        magnetisation_path = self.path_data_eq + '/side' + str(self.side) + '_temp' + str(float(self.temperature))[-5:] + '_magnetisation.pickle'
        with open(magnetisation_path, 'rb') as f:
            self.magnetisations = pickle.load(f)

    # Plot physics
    def plot_physics(self):
        energies = [energy / self.number_of_nodes for energy in self.energies]
        magnetisations = [magnetisation / self.number_of_nodes for magnetisation in self.magnetisations]

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].plot(energies)
        axs[0].set_title("Energies")
        axs[1].plot(magnetisations)
        axs[1].set_title("Magnetisations")
        fig.tight_layout(pad=2.0)
        plt.savefig(self.path_plot + '/side' + str(self.side) + '_temp' + str(float(self.temperature))[-5:] + '.pdf')
        plt.close()

    # Get physical quantities
    def get_physics(self, average_time):
        energies = [energy / self.number_of_nodes for energy in self.energies]
        magnetisations = [magnetisation / self.number_of_nodes for magnetisation in self.magnetisations]

        magnetisation_data = magnetisations[-average_time:]
        energy_data = energies[-average_time:]

        magnetisation = np.mean(magnetisation_data)
        energy = np.mean(energy_data)

        magnetisation_square = np.mean([magnetisation**2 for magnetisation in magnetisation_data])
        energy_square = np.mean([energy**2 for energy in energy_data])

        specific_heat = (energy_square - energy**2) / self.temperature / self.temperature
        susceptibility = (magnetisation_square - magnetisation**2) / self.temperature

        return energy, magnetisation, specific_heat * self.number_of_nodes, susceptibility 

class IsingPhaseTransition:
    def __init__(self, kind_of_graph, side):
        self.kind_of_graph = kind_of_graph
        self.side = side

        self.path_data = './data'
        if not os.path.exists(self.path_data):
            os.mkdir(self.path_data)
        self.path_data = './data/data_phase_transition'
        if not os.path.exists(self.path_data):
            os.mkdir(self.path_data)
        self.path_data = './data/data_phase_transition/' + self.kind_of_graph
        if not os.path.exists(self.path_data):
            os.mkdir(self.path_data)

        self.path_plot = './plot'
        if not os.path.exists(self.path_plot):
            os.mkdir(self.path_plot)    
        self.path_plot = './plot/plot_phase_transition'
        if not os.path.exists(self.path_plot):
            os.mkdir(self.path_plot)
        self.path_plot = './plot/plot_phase_transition/' + self.kind_of_graph
        if not os.path.exists(self.path_plot):
            os.mkdir(self.path_plot)
    
    # Set physical quantities   
    def set_temperatures(self, temperatures):
        self.temperatures = temperatures

    # Compute phase transition
    def phase_transition(self, average_time):
        self.energies = []
        self.magnetisations = []
        self.specific_heats = []
        self.susceptibilities = []
        for temperature in tqdm(self.temperatures):
            ising = IsingNetwork(self.kind_of_graph, self.side)
            ising.load_graph()
            ising.set_temperature(temperature)
            ising.load_data()
            e, m, h, s = ising.get_physics(average_time)
            self.energies.append(e)
            self.magnetisations.append(abs(m))
            self.specific_heats.append(h)
            self.susceptibilities.append(s)

    # Save data
    def save_data(self):
        energy_path = self.path_data + '/side' + str(self.side) + '_energies.pickle'
        with open(energy_path, 'wb') as f:
            pickle.dump(self.energies, f)
        
        magnetisation_path = self.path_data + '/side' + str(self.side) + '_magnetisations.pickle'
        with open(magnetisation_path, 'wb') as f:
            pickle.dump(self.magnetisations, f)

        specific_heat_path = self.path_data + '/side' + str(self.side) + '_specific_heats.pickle'
        with open(specific_heat_path, 'wb') as f:
            pickle.dump(self.specific_heats, f)

        susceptibility_path = self.path_data + '/side' + str(self.side) + '_susceptibilities.pickle'
        with open(susceptibility_path, 'wb') as f:
            pickle.dump(self.susceptibilities, f)

    # Load data
    def load_data(self):
        energy_path = self.path_data + '/side' + str(self.side)  + '_energies.pickle'
        with open(energy_path, 'rb') as f:
            self.energies = pickle.load(f)

        magnetisation_path = self.path_data + '/side' + str(self.side) + '_magnetisations.pickle'
        with open(magnetisation_path, 'rb') as f:
            self.magnetisations = pickle.load(f)

        specific_heat_path = self.path_data + '/side' + str(self.side) + '_specific_heats.pickle'
        with open(specific_heat_path, 'rb') as f:                      
            self.specific_heats = pickle.load(f)

        susceptibility_path = self.path_data + '/side' + str(self.side) + '_susceptibilities.pickle'
        with open(susceptibility_path, 'rb') as f:
            self.susceptibilities = pickle.load(f)

    def plot_physics(self):    
        data = ExactSolutionIsing(self.kind_of_graph)
        T, F, U, M, C = data.load_data()

        plt.scatter(self.temperatures, self.energies)
        plt.plot(T, U, color='red')
        plt.title("Energies")
        plt.savefig(self.path_plot + '/side' + str(self.side) + '_energies.pdf')
        plt.close()

        plt.scatter(self.temperatures, self.magnetisations)
        plt.plot(T, M, color='red')
        plt.title("Magnetisations")
        plt.savefig(self.path_plot + '/side' + str(self.side) + '_magnetisations.pdf')
        plt.close()

        plt.scatter(self.temperatures, self.specific_heats)
        plt.plot(T[2:-2], C, color='red')
        plt.title("Specific heats")
        plt.savefig(self.path_plot + '/side' + str(self.side) + '_specific_heats.pdf')
        plt.close()

        plt.scatter(self.temperatures, self.susceptibilities)
        plt.title("Susceptibilities")
        plt.savefig(self.path_plot + '/side' + str(self.side) + '_susceptibilities.pdf')
        plt.close()