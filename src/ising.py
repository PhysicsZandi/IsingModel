import matplotlib.pyplot as plt
import matplotlib.animation as ani
import matplotlib.colors as col
import scipy as sci
import networkx as nx
import numpy as np
import os 
import pickle
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

cmap = col.ListedColormap(['teal', 'black'])

class IsingSystem:
    def __init__(self, kind_of_graph, size, temperature, J=1, B=0):
        # Implement feature of the graph
        self.kind_of_graph = kind_of_graph
        self.size = size

        # Implement physical quantities
        self.temperature = temperature
        self.J = J
        self.B = B

        # Implement paths to save/load data
        self.path_data = './data'
        if not os.path.exists(self.path_data):
            os.mkdir(self.path_data)
        self.path_data = './data/' + self.kind_of_graph
        if not os.path.exists(self.path_data):
            os.mkdir(self.path_data)
        self.path_data = './data/' + self.kind_of_graph + '/' + str(self.size)
        if not os.path.exists(self.path_data):
            os.mkdir(self.path_data)

        # Implement paths to save/load animation
        self.path_animation = './plot'
        if not os.path.exists(self.path_animation):
            os.mkdir(self.path_animation)
        self.path_animation = './plot/animation'
        if not os.path.exists(self.path_animation):
            os.mkdir(self.path_animation)
        self.path_animation = './plot/animation/' + self.kind_of_graph
        if not os.path.exists(self.path_animation):
            os.mkdir(self.path_animation)
        self.path_animation = './plot/animation/' + self.kind_of_graph + '/' + str(self.size)
        if not os.path.exists(self.path_animation):
            os.mkdir(self.path_animation)

        # Implement paths to save/load graph
        self.path_graph = './plot'
        if not os.path.exists(self.path_graph):
            os.mkdir(self.path_graph)
        self.path_graph = './plot/graph'
        if not os.path.exists(self.path_graph):
            os.mkdir(self.path_graph)
        self.path_graph = './plot/graph/' + self.kind_of_graph
        if not os.path.exists(self.path_graph):
            os.mkdir(self.path_graph)
        self.path_graph = './plot/graph/' + self.kind_of_graph + '/' + str(self.size)
        if not os.path.exists(self.path_graph):
            os.mkdir(self.path_graph)

    ##### GRAPH #####

    # Choose which graph and calculate all its features: number of nodes, nodes, neighbours, spins (and radii)
    def _initialise_graph(self, probability = 0.4):
        if self.kind_of_graph == "erdos_renyi":
            self.graph = nx.erdos_renyi_graph(self.size, probability)
        elif self.kind_of_graph == "1d":
            self.graph = nx.cycle_graph(self.size)
        elif self.kind_of_graph == "2d_square":
            self.graph = nx.grid_2d_graph(self.size, self.size, periodic=True)
        elif self.kind_of_graph == "2d_triangular":
            self.graph =  nx.triangular_lattice_graph(self.size, self.size, periodic=True)
        elif self.kind_of_graph == "2d_hexagonal":
            self.graph =  nx.hexagonal_lattice_graph(self.size, self.size, periodic=True)

        self.number_of_nodes = self.graph.number_of_nodes()
        self.nodes = self.graph.nodes()
        self._calculate_neighbours()
        self.initialise_spins_hot()
        self._calculate_radii()

    # Save indices of neighbours in the list self.neighbours
    def _calculate_neighbours(self):
        self.neighbours = []
        node_indices = {node: idx for idx, node in enumerate(self.nodes)}  
        for node in self.nodes:
            neighbours_of_node = []
            for neighbour_of_node in self.graph.neighbors(node):
                neighbour_index = node_indices.get(neighbour_of_node)
                if neighbour_index is not None:
                    neighbours_of_node.append(neighbour_index)
            self.neighbours.append(neighbours_of_node)

    # Save indices of radii in the list self.radii
    def _calculate_radii(self):
        radii_dictionary = dict(nx.all_pairs_shortest_path_length(self.graph))
        num_nodes = len(self.graph.nodes)
        self.radii  = np.full((num_nodes, num_nodes), np.inf)
        node_to_index = {node: i for i, node in enumerate(self.graph.nodes())}
        for source, paths in radii_dictionary.items():
            source_index = node_to_index[source]
            for target, distance in paths.items():
                target_index = node_to_index[target]
                self.radii[source_index, target_index] = distance

    # Save randomly aligned spins in the list self.spins
    def initialise_spins_hot(self):
        self.spins = np.random.choice([-1, 1], self.number_of_nodes)
        self._calculate_magnetisation()
        self._calculate_energy()

    # Save upwards aligned spins in the list self.spins
    def initialise_spins_cold(self):
        self.spins = np.full(self.number_of_nodes, 1)
        self._calculate_magnetisation()
        self._calculate_energy()
  
    ##### PHYSICAL QUANTITIES #####

    # Calculate the total energy using the Hamiltonian
    def _calculate_energy(self):
        neighbour_spins = 0
        for i in range(self.number_of_nodes):
            for j in self.neighbours[i]:
                neighbour_spins += self.spins[i] * self.spins[j]
        self.energy = - self.J * neighbour_spins / 2 - self.B * np.sum(self.spins)

    # Calculate the total magnetisation using the definition as the sum of all spins
    def _calculate_magnetisation(self):
        self.magnetisation = np.sum(self.spins)

    # Calculate the total entropy in the microcanonical ensemble with the Boltzmann formula
    def _calculate_entropy(self):
        spin_up = np.sum(self.spins > 0)
        spin_down = np.sum(self.spins < 0)
        # if spin_up or spin_down is 0, set them to 1 in order to have log(1) = 0
        spin_up = max(spin_up, 1) 
        spin_down = max(spin_down, 1)
        self.entropy = self.number_of_nodes * np.log(self.number_of_nodes) - spin_up * np.log(spin_up) - spin_down * np.log(spin_down)
        
    # Calculate correlation function as a function of the radius
    def _calculate_correlation_function(self):
        max_radii = int(np.max(self.radii))
        self.correlation_function = []
        list_of_nodes = list(self.nodes)
        node_indices = {node: i for i, node in enumerate(list_of_nodes)}
        radius_spin = [[] for _ in range(max_radii + 1)]
        for node1 in self.nodes:
            i = node_indices[node1]
            for node2 in self.nodes:
                j = node_indices[node2]
                radius = int(self.radii[i][j])
                radius_spin[radius].append(self.spins[i] * self.spins[j])
        for i in range(max_radii):
            self.correlation_function.append(np.mean(radius_spin[i]))

    # Calculate the correlation lenght fitting a negative exponential in the correlation function  
    def fit_function(self, x, xi, a, b):
            return a * np.exp(- x / xi) + b
    
    def _calculate_correlation_length(self):
        if np.std(self.correlation_function) < 0.05:
            self.correlation_length = 0
        else:
            r = np.arange(int(np.max(self.radii)))
            popt, pcov = sci.optimize.curve_fit(self.fit_function, r, self.correlation_function, p0 = [1, 1, 0], maxfev=10000)
            self.correlation_length = popt[0]

    # Calculate the two points correlation function using its definition as mean af products of spins
    def _calculate_two_points_function(self):
        list_of_nodes = list(self.nodes)
        node_indices = {node: i for i, node in enumerate(list_of_nodes)}
        for node1 in self.nodes:
            i = node_indices[node1]
            self.two_points_function_part2[i] += self.spins[i] 
            for node2 in self.nodes:
                j = node_indices[node2]
                if i > j:
                    self.two_points_function_part1[i][j] += self.spins[i] * self.spins[j] 

    def _calculate_two_points_function_average(self, average_time):
        self.two_points_function_part1 = self.two_points_function_part1 / average_time
        self.two_points_function_part2 = self.two_points_function_part2 / average_time / average_time

        self.two_points_function = np.zeros([self.number_of_nodes, self.number_of_nodes])
        for i in range(self.number_of_nodes):
            for j in range(self.number_of_nodes):
                if i > j:
                    self.two_points_function[i][j] = self.two_points_function_part1[i][j] - self.two_points_function_part2[i] * self.two_points_function_part2[j]
        self.two_points_function = self.two_points_function + self.two_points_function.T

    ##### EVOLUTION #####

    # Evolve the system using the Metropolis algorithm scrooling all spins
    def evolution(self, evolution_time, average_time):
        self._initialise_graph()

        self.energies = []
        self.magnetisations = []
        self.entropies = []
        self.correlation_lengths = []
        self.two_points_function_part1 = np.zeros([self.number_of_nodes, self.number_of_nodes])
        self.two_points_function_part2 = np.zeros(self.number_of_nodes)
        self._calculate_energy()

        for _ in range(evolution_time):
            random_numbers = np.random.rand(self.number_of_nodes)
            for i in range(self.number_of_nodes):
                neighbour_spins = self.spins[i] * sum([self.spins[j] for j in self.neighbours[i]])
                delta_energy = 2.0 * self.J * neighbour_spins + 2 * self.B * neighbour_spins
                transition_probability = np.exp(- delta_energy / self.temperature)
                if delta_energy < 0 or transition_probability > random_numbers[i]:
                    self.spins[i] *= -1
                    self.energy += delta_energy

        for _ in range(average_time):
            random_numbers = np.random.rand(self.number_of_nodes)
            for i in range(self.number_of_nodes):
                neighbour_spins = self.spins[i] * sum([self.spins[j] for j in self.neighbours[i]])
                delta_energy = 2.0 * self.J * neighbour_spins + 2 * self.B * neighbour_spins
                transition_probability = np.exp(- delta_energy / self.temperature)
                if delta_energy < 0 or transition_probability > random_numbers[i]:
                    self.spins[i] *= -1
                    self.energy += delta_energy
            self.energies.append(self.energy)
            self._calculate_magnetisation()
            self.magnetisations.append(self.magnetisation)
            self._calculate_entropy()
            self.entropies.append(self.entropy)
            self._calculate_correlation_function()
            self._calculate_correlation_length()
            self.correlation_lengths.append(self.correlation_length)
            self._calculate_two_points_function()
        self._calculate_two_points_function_average(average_time)

    # Evolve the system using the Glauber algorithm randomly choose one spin
    def evolution_random(self, evolution_time, average_time):
        self._initialise_graph()

        self.energies = []
        self.magnetisations = []
        self.entropies = []
        self.correlation_lengths = []
        self._calculate_energy()

        choices = np.random.choice([i for i in range(self.number_of_nodes)], evolution_time)
        random_numbers = np.random.rand(evolution_time)
        for n in range(evolution_time):
            i = choices[n]
            neighbour_spins = self.spins[i] * sum([self.spins[j] for j in self.neighbours[i]])
            delta_energy = 2.0 * self.J * neighbour_spins + 2 * self.B * neighbour_spins
            transition_probability = np.exp(- delta_energy / self.temperature)
            if delta_energy < 0 or transition_probability > random_numbers[i]:
                self.spins[i] *= -1
                self.energy += delta_energy

        choices = np.random.choice([i for i in range(self.number_of_nodes)], average_time)
        random_numbers = np.random.rand(average_time)
        for n in range(average_time):
            i = choices[n]
            neighbour_spins = self.spins[i] * sum([self.spins[j] for j in self.neighbours[i]])
            delta_energy = 2.0 * self.J * neighbour_spins + 2 * self.B * neighbour_spins
            transition_probability = np.exp(- delta_energy / self.temperature)
            if delta_energy < 0 or transition_probability > random_numbers[i]:
                self.spins[i] *= -1
                self.energy += delta_energy
            if (n % 1000 == 0):
                self.energies.append(self.energy)
                self._calculate_magnetisation()
                self.magnetisations.append(self.magnetisation)
                self._calculate_entropy()
                self.entropies.append(self.entropy)

    def get_two_points_function(self):
        return self.two_points_function
    
    ##### AVERAGES #####

    # Once reached the equilibrium, get the thermal average
    def get_thermal_average(self):
        energies = [energy / self.number_of_nodes for energy in self.energies]
        magnetisations = [magnetisation / self.number_of_nodes for magnetisation in self.magnetisations]
        entropies = [entropy / self.number_of_nodes for entropy in self.entropies]

        magnetisation_data = magnetisations
        magnetisation = np.mean(magnetisation_data)
        magnetisation_square = np.mean([magnetisation**2 for magnetisation in magnetisation_data])

        energy_data = energies
        energy = np.mean(energy_data)
        energy_square = np.mean([energy**2 for energy in energy_data])

        specific_heat = (energy_square - energy**2) / self.temperature / self.temperature * self.number_of_nodes
        
        susceptibility = (magnetisation_square - magnetisation**2) / self.temperature * self.number_of_nodes

        entropy = np.mean(entropies)

        free_energy = energy - self.temperature * entropy

        correlation_length = np.mean(self.correlation_lengths)

        return energy, magnetisation, specific_heat, susceptibility, entropy, free_energy, correlation_length

    def get_thermal_average_random(self):
        energies = [energy / self.number_of_nodes for energy in self.energies]
        magnetisations = [magnetisation / self.number_of_nodes for magnetisation in self.magnetisations]
        entropies = [entropy / self.number_of_nodes for entropy in self.entropies]

        magnetisation_data = magnetisations
        magnetisation = np.mean(magnetisation_data)
        magnetisation_square = np.mean([magnetisation**2 for magnetisation in magnetisation_data])

        energy_data = energies
        energy = np.mean(energy_data)
        energy_square = np.mean([energy**2 for energy in energy_data])

        specific_heat = (energy_square - energy**2) / self.temperature / self.temperature * self.number_of_nodes
        
        susceptibility = (magnetisation_square - magnetisation**2) / self.temperature * self.number_of_nodes

        entropy = np.mean(entropies)

        free_energy = energy - self.temperature * entropy

        return energy, magnetisation, specific_heat, susceptibility, entropy, free_energy

    # In an ensemble of systems at the same temperature, get the statistical average
    def get_statistical_average(self, number_of_ensemble):
        set_energies = []
        set_magnetisations = []
        set_entropies = []
        set_free_energies = []
        set_specific_heats = []
        set_susceptibilities = []
        set_correlation_lengths = []

        set_betweenness_centralities = []
        set_clustering_coefficients = []
        set_shortest_paths = []
        set_diameters = []
        set_densities = []
        set_disparities = []

        for _ in tqdm(range(number_of_ensemble)):
            ising = IsingSystem(self.kind_of_graph, self.size, self.temperature)
            ising.evolution(4000, 1000)
            energy, magnetisation, specific_heat, susceptibility, entropy, free_energy, correlation_length = ising.get_thermal_average()
            set_energies.append(energy)
            set_magnetisations.append(abs(magnetisation))
            set_entropies.append(entropy)
            set_free_energies.append(free_energy)
            set_specific_heats.append(specific_heat)
            set_susceptibilities.append(susceptibility)
            set_correlation_lengths.append(correlation_length)

            two_points = ising.get_two_points_function()
            weighted_graph = WeightedGraph(self.kind_of_graph, self.size, self.temperature)
            weighted_graph.create_graph(two_points)
            betweenness_centrality, clustering_coefficient, shortest_path, diameter, density, disparity = weighted_graph.get_properties()
            set_betweenness_centralities.append(betweenness_centrality)
            set_clustering_coefficients.append(clustering_coefficient)
            set_shortest_paths.append(shortest_path)
            set_diameters.append(diameter)
            set_densities.append(density)
            set_disparities.append(disparity)

        self.energy = np.mean(set_energies)
        self.magnetisation = np.mean(set_magnetisations)
        self.entropy = np.mean(set_entropies)
        self.free_energy = np.mean(set_free_energies)
        self.specific_heat = np.mean(set_specific_heats)
        self.susceptibility = np.mean(set_susceptibilities)
        self.correlation_length = np.mean(set_correlation_lengths)
        self.betweenness_centrality = np.mean(set_betweenness_centralities)
        self.clustering_coefficient = np.mean(set_clustering_coefficients)
        self.shortest_path = np.mean(set_shortest_paths)
        self.diameter = np.mean(set_diameters)
        self.density = np.mean(set_densities)
        self.disparity = np.mean(set_disparities)

    def get_statistical_average_random(self, number_of_ensemble):
        set_energies = []
        set_magnetisations = []
        set_entropies = []
        set_free_energies = []
        set_specific_heats = []
        set_susceptibilities = []

        for _ in tqdm(range(number_of_ensemble)):
            ising = IsingSystem(self.kind_of_graph, self.size, self.temperature)
            ising.evolution_random(2000000, 1000000)
            energy, magnetisation, specific_heat, susceptibility, entropy, free_energy = ising.get_thermal_average_random()
            set_energies.append(energy)
            set_magnetisations.append(abs(magnetisation))
            set_entropies.append(entropy)
            set_free_energies.append(free_energy)
            set_specific_heats.append(specific_heat)
            set_susceptibilities.append(susceptibility)

        self.energy = np.mean(set_energies)
        self.magnetisation = np.mean(set_magnetisations)
        self.entropy = np.mean(set_entropies)
        self.free_energy = np.mean(set_free_energies)
        self.specific_heat = np.mean(set_specific_heats)
        self.susceptibility = np.mean(set_susceptibilities)

    # Save and load data using pickle
    def save_data(self):
        self.get_statistical_average(20)

        energy_path = self.path_data + '/energy_temp' + str(float(self.temperature))[-5:] + '.pickle'
        with open(energy_path, 'wb') as f:
            pickle.dump(self.energy, f)

        magnetisation_path = self.path_data + '/magnetisation_temp' + str(float(self.temperature))[-5:] + '.pickle'
        with open(magnetisation_path, 'wb') as f:
            pickle.dump(self.magnetisation, f)

        specific_heat_path = self.path_data + '/specific_heat_temp' + str(float(self.temperature))[-5:] + '.pickle'
        with open(specific_heat_path, 'wb') as f:
            pickle.dump(self.specific_heat, f)

        susceptibility_path = self.path_data + '/susceptibility_temp' + str(float(self.temperature))[-5:] + '.pickle'
        with open(susceptibility_path, 'wb') as f:
            pickle.dump(self.susceptibility, f)

        entropy_path = self.path_data + '/entropy_temp' + str(float(self.temperature))[-5:] + '.pickle'
        with open(entropy_path, 'wb') as f:
            pickle.dump(self.entropy, f)

        free_energy_path = self.path_data + '/free_energy_temp' + str(float(self.temperature))[-5:] + '.pickle'
        with open(free_energy_path, 'wb') as f:
            pickle.dump(self.free_energy, f)
        
        correlation_length_path = self.path_data + '/correlation_length_temp' + str(float(self.temperature))[-5:] + '.pickle'
        with open(correlation_length_path, 'wb') as f:
            pickle.dump(self.correlation_length, f)

        betweeness_centrality_path = self.path_data + '/betweeness_centrality_temp' + str(float(self.temperature))[-5:] + '.pickle'
        with open(betweeness_centrality_path, 'wb') as f:
            pickle.dump(self.betweenness_centrality, f)

        clustering_coefficient_path = self.path_data + '/clustering_coefficient_temp' + str(float(self.temperature))[-5:] + '.pickle'
        with open(clustering_coefficient_path, 'wb') as f:
            pickle.dump(self.clustering_coefficient, f)

        shortest_path_path = self.path_data + '/shortest_path_temp' + str(float(self.temperature))[-5:] + '.pickle'
        with open(shortest_path_path, 'wb') as f:
            pickle.dump(self.shortest_path, f)  
        
        diameter_path = self.path_data + '/diameter_temp' + str(float(self.temperature))[-5:] + '.pickle'
        with open(diameter_path, 'wb') as f:
            pickle.dump(self.diameter, f)   
        
        density_path = self.path_data + '/density_temp' + str(float(self.temperature))[-5:] + '.pickle'
        with open(density_path, 'wb') as f:
            pickle.dump(self.density, f)    
        
        disparity_path = self.path_data + '/disparity_temp' + str(float(self.temperature))[-5:] + '.pickle'
        with open(disparity_path, 'wb') as f:
            pickle.dump(self.disparity, f)

    def load_data(self):
        energy_path = self.path_data + '/energy_temp' + str(float(self.temperature))[-5:] + '.pickle'
        with open(energy_path, 'rb') as f:
            self.energy = pickle.load(f)

        magnetisation_path = self.path_data + '/magnetisation_temp' + str(float(self.temperature))[-5:] + '.pickle'
        with open(magnetisation_path, 'rb') as f:
            self.magnetisation = pickle.load(f)

        specific_heat_path = self.path_data + '/specific_heat_temp' + str(float(self.temperature))[-5:] + '.pickle'
        with open(specific_heat_path, 'rb') as f:
            self.specific_heat = pickle.load(f)

        susceptibility_path = self.path_data + '/susceptibility_temp' + str(float(self.temperature))[-5:] + '.pickle'
        with open(susceptibility_path, 'rb') as f:
            self.susceptibility = pickle.load(f)

        entropy_path = self.path_data + '/entropy_temp' + str(float(self.temperature))[-5:] + '.pickle'
        with open(entropy_path, 'rb') as f:
            self.entropy = pickle.load(f)

        free_energy_path = self.path_data + '/free_energy_temp' + str(float(self.temperature))[-5:] + '.pickle'
        with open(free_energy_path, 'rb') as f:
            self.free_energy = pickle.load(f)

        
        correlation_length_path = self.path_data + '/correlation_length_temp' + str(float(self.temperature))[-5:] + '.pickle'
        with open(correlation_length_path, 'rb') as f:
            self.correlation_length = pickle.load(f)


        betweeness_centrality_path = self.path_data + '/betweeness_centrality_temp' + str(float(self.temperature))[-5:] + '.pickle'
        with open(betweeness_centrality_path, 'rb') as f:
            self.betweenness_centrality = pickle.load(f)

        clustering_coefficient_path = self.path_data + '/clustering_coefficient_temp' + str(float(self.temperature))[-5:] + '.pickle'
        with open(clustering_coefficient_path, 'rb') as f:
            self.clustering_coefficient = pickle.load(f)

        shortest_path_path = self.path_data + '/shortest_path_temp' + str(float(self.temperature))[-5:] + '.pickle'
        with open(shortest_path_path, 'rb') as f:
            self.shortest_path = pickle.load(f)

        diameter_path = self.path_data + '/diameter_temp' + str(float(self.temperature))[-5:] + '.pickle'
        with open(diameter_path, 'rb') as f:
            self.diameter = pickle.load(f)

        density_path = self.path_data + '/density_temp' + str(float(self.temperature))[-5:] + '.pickle'
        with open(density_path, 'rb') as f:
            self.density = pickle.load(f)

        disparity_path = self.path_data + '/disparity_temp' + str(float(self.temperature))[-5:] + '.pickle'
        with open(disparity_path, 'rb') as f:
            self.disparity = pickle.load(f)

        return self.energy, self.magnetisation, self.entropy, self.free_energy, self.specific_heat, self.susceptibility, self.correlation_length, self.betweenness_centrality, self.clustering_coefficient, self.shortest_path, self.diameter, self.density, self.disparity

    def save_data_random(self):
        self.get_statistical_average_random(20)

        energy_path = self.path_data + '/energy_temp' + str(float(self.temperature))[-5:] + '.pickle'
        with open(energy_path, 'wb') as f:
            pickle.dump(self.energy, f)

        magnetisation_path = self.path_data + '/magnetisation_temp' + str(float(self.temperature))[-5:] + '.pickle'
        with open(magnetisation_path, 'wb') as f:
            pickle.dump(self.magnetisation, f)

        specific_heat_path = self.path_data + '/specific_heat_temp' + str(float(self.temperature))[-5:] + '.pickle'
        with open(specific_heat_path, 'wb') as f:
            pickle.dump(self.specific_heat, f)

        susceptibility_path = self.path_data + '/susceptibility_temp' + str(float(self.temperature))[-5:] + '.pickle'
        with open(susceptibility_path, 'wb') as f:
            pickle.dump(self.susceptibility, f)

        entropy_path = self.path_data + '/entropy_temp' + str(float(self.temperature))[-5:] + '.pickle'
        with open(entropy_path, 'wb') as f:
            pickle.dump(self.entropy, f)

        free_energy_path = self.path_data + '/free_energy_temp' + str(float(self.temperature))[-5:] + '.pickle'
        with open(free_energy_path, 'wb') as f:
            pickle.dump(self.free_energy, f)
        
    def load_data_random(self):
        energy_path = self.path_data + '/energy_temp' + str(float(self.temperature))[-5:] + '.pickle'
        with open(energy_path, 'rb') as f:
            self.energy = pickle.load(f)

        magnetisation_path = self.path_data + '/magnetisation_temp' + str(float(self.temperature))[-5:] + '.pickle'
        with open(magnetisation_path, 'rb') as f:
            self.magnetisation = pickle.load(f)

        specific_heat_path = self.path_data + '/specific_heat_temp' + str(float(self.temperature))[-5:] + '.pickle'
        with open(specific_heat_path, 'rb') as f:
            self.specific_heat = pickle.load(f)

        susceptibility_path = self.path_data + '/susceptibility_temp' + str(float(self.temperature))[-5:] + '.pickle'
        with open(susceptibility_path, 'rb') as f:
            self.susceptibility = pickle.load(f)

        entropy_path = self.path_data + '/entropy_temp' + str(float(self.temperature))[-5:] + '.pickle'
        with open(entropy_path, 'rb') as f:
            self.entropy = pickle.load(f)

        free_energy_path = self.path_data + '/free_energy_temp' + str(float(self.temperature))[-5:] + '.pickle'
        with open(free_energy_path, 'rb') as f:
            self.free_energy = pickle.load(f)

        return self.energy, self.magnetisation, self.entropy, self.free_energy, self.specific_heat, self.susceptibility

    ##### ANIMATION in 2D #####

    # Evolve saving the spins
    def _compute_field(self):
        size = int(np.sqrt(self.number_of_nodes))
        field = np.zeros((size, size))
        for n in range(self.number_of_nodes):
            field[n // size][n % size] = self.spins[n]
        return field
    
    def evolution_animation(self, evolution_time, average_time):
        self._initialise_graph()
        self.field = []
        self._calculate_energy()

        for _ in tqdm(range(evolution_time)):
            random_numbers = np.random.rand(self.number_of_nodes)
            for i in range(self.number_of_nodes):
                neighbour_spins = self.spins[i] * sum([self.spins[j] for j in self.neighbours[i]])
                delta_energy = 2.0 * self.J * neighbour_spins + 2 * self.B * neighbour_spins
                transition_probability = np.exp(- delta_energy / self.temperature)
                if delta_energy < 0 or transition_probability > random_numbers[i]:
                    self.spins[i] *= -1
                    self.energy += delta_energy

        for _ in tqdm(range(average_time)):
            random_numbers = np.random.rand(self.number_of_nodes)
            for i in range(self.number_of_nodes):
                neighbour_spins = self.spins[i] * sum([self.spins[j] for j in self.neighbours[i]])
                delta_energy = 2.0 * self.J * neighbour_spins + 2 * self.B * neighbour_spins
                transition_probability = np.exp(- delta_energy / self.temperature)
                if delta_energy < 0 or transition_probability > random_numbers[i]:
                    self.spins[i] *= -1
                    self.energy += delta_energy
            self.field.append(self._compute_field())

    # Animation representations of the network
    def plot_network_animation(self):
        fig, ax = plt.subplots()
        def animate(i):
            ax.clear() 
            heatmap = ax.imshow(self.field[i], cmap=cmap, interpolation='nearest')
            ax.axis("off")
            ax.set_title(f'Graphical representation of a ' + self.kind_of_graph + ' at temperature ' + str(self.temperature))
        heatmap = ax.imshow([np.random.choice([-1, 1], self.number_of_nodes)], cmap=cmap, interpolation='nearest')
        plt.colorbar(heatmap, shrink=0.3)
        animation = ani.FuncAnimation(fig, animate, frames=len(self.field), interval=100, repeat=False)
        animation.save(self.path_animation + f'/temperature_{self.temperature}.gif')
        plt.close()

    # Plot graph with library networkx
    def plot_graph(self):
        colors = []
        for spin in self.spins:
            if spin == 1:
                colors.append("teal")
            else:
                colors.append("black")

        pos = nx.spring_layout(self.graph) 
        nx.draw(self.graph, pos, node_size=100, node_color=colors)
        plt.savefig(self.path_graph + f'/temperature_{self.temperature}.pdf')
        plt.close()

class WeightedGraph:
    def __init__(self, kind_of_graph, size, temperature):
        # Implement features of the graph
        self.kind_of_graph = kind_of_graph
        self.size = size

        # Implement temperature
        self.temperature = temperature

        # Implement paths to save/load graph
        self.path_graph = './plot'
        if not os.path.exists(self.path_graph):
            os.mkdir(self.path_graph)
        self.path_graph = './plot/weighted_graph'
        if not os.path.exists(self.path_graph):
            os.mkdir(self.path_graph)
        self.path_graph = './plot/weighted_graph/' + self.kind_of_graph
        if not os.path.exists(self.path_graph):
            os.mkdir(self.path_graph)
        self.path_graph = './plot/weighted_graph/' + self.kind_of_graph + '/' + str(self.size)
        if not os.path.exists(self.path_graph):
            os.mkdir(self.path_graph)

    # Create a weighted grapg with two points correlation function as adjacency matrix
    def create_graph(self, two_point_function):
        self.two_point_function = abs(two_point_function)
        self.graph = nx.from_numpy_array(self.two_point_function)
        self.number_of_nodes = self.graph.number_of_nodes()

        if nx.is_connected(self.graph) == False:
            raise Exception(f"Temperature {self.temperature} is not connected")

    # Calculate average betwenness centrality using the library networkx
    def _calculate_betweenness_centrality(self):
        betweenness_centrality = nx.betweenness_centrality(self.graph, weight="weight")
        self.betweenness_centrality = sum(betweenness_centrality.values()) / self.number_of_nodes

    # Calculate average clustering coefficient using the library networkx
    def _calculate_clustering_coefficient(self):
        self.clustering_coefficient = nx.average_clustering(self.graph, weight='weight')

    # Calculate average shortest path using the library networkx
    def _calculate_shortest_path(self):
        self.shortest_path = nx.average_shortest_path_length(self.graph,  weight='weight')

    # Calculate diameter using the library networkx
    def _calculate_diameter(self):
        eccentricity = nx.eccentricity(self.graph, weight="weight")
        self.diameter = np.max(list(eccentricity.values()))

    # Calculate density using the library networkx
    def _calculate_density(self):
        two_point_function_row = np.zeros(self.number_of_nodes)
        for i in range(self.number_of_nodes):
            for j in range(self.number_of_nodes):
                two_point_function_row[i] += self.two_point_function[i][j]

        self.density = sum(two_point_function_row) / self.number_of_nodes**2

    # Calculate disparity using the definition
    def _calculate_disparity(self):
        two_point_function_square = np.zeros([self.number_of_nodes, self.number_of_nodes])
        for i in range(self.number_of_nodes):
            for j in range(self.number_of_nodes):
                two_point_function_square[i][j] = self.two_point_function[i][j]**2

        two_point_function_row = np.zeros(self.number_of_nodes)
        two_point_function_row_square = np.zeros(self.number_of_nodes)
        for i in range(self.number_of_nodes):
            for j in range(self.number_of_nodes):
                two_point_function_row[i] += self.two_point_function[i][j]
                two_point_function_row_square[i] += two_point_function_square[i][j]

        sum_two_point_function = 0
        if two_point_function_row.all() != 0:
            for i in range(self.number_of_nodes):
                sum_two_point_function += two_point_function_row_square[i] / two_point_function_row[i] / two_point_function_row[i]
    
        self.disparity = sum_two_point_function / self.number_of_nodes
    
    # Get all the properties of the graph
    def get_properties(self):
        self._calculate_betweenness_centrality()
        self._calculate_clustering_coefficient()
        self._calculate_shortest_path()
        self._calculate_diameter()
        self._calculate_density()
        self._calculate_disparity()

        return self.betweenness_centrality, self.clustering_coefficient, self.shortest_path, self.diameter, self.density, self.disparity

    # Plot graph
    def plot_graph(self):
        edge_weights = [d['weight'] for u, v, d in self.graph.edges(data=True)]
        edge_colors = [plt.cm.binary(weight) for weight in edge_weights]
        nx.draw(self.graph, pos=nx.spring_layout(self.graph), node_size=100, node_color="teal", edge_color=edge_colors)

        plt.savefig(self.path_graph + '/weighted_graph_temperature' + str(float(self.temperature))[-5:] + '.pdf')
        plt.close()
        
class TemperatureBehaviour:
    def __init__(self, kind_of_graph, size, temperatures):
        # Implement feature of the graph
        self.kind_of_graph = kind_of_graph
        self.size = size

        # Implement temperatures
        self.temperatures = temperatures

        # Implement paths to save/load phase transition plots
        self.path_plot = './plot'
        if not os.path.exists(self.path_plot):
            os.mkdir(self.path_plot)
        self.path_plot = './plot/phase_transition'
        if not os.path.exists(self.path_plot):
            os.mkdir(self.path_plot)
        self.path_plot = './plot/phase_transition/' + self.kind_of_graph
        if not os.path.exists(self.path_plot):
            os.mkdir(self.path_plot)
        self.path_plot = './plot/phase_transition/' + self.kind_of_graph + '/' + str(self.size)
        if not os.path.exists(self.path_plot):
            os.mkdir(self.path_plot)

    # Get physical quantities for each temperature and save them in a list
    def phase_transition(self):
        self.energies = []
        self.magnetisations = []
        self.specific_heats = []
        self.susceptibilities = []
        self.entropies = []
        self.free_energies = []
        self.correlation_lengths = []

        self.betweenness_centralities = []
        self.clustering_coefficients = []
        self.shortest_paths = []
        self.diameters = []
        self.densities = []
        self.disparities = []

        for temperature in tqdm(self.temperatures):
            ising = IsingSystem(self.kind_of_graph, self.size, temperature)
            e, m, s, f, c, chi, xi, bet_cen, clust_coeff, short_path, diam, dens, disp = ising.load_data()
            self.energies.append(e)
            self.magnetisations.append(abs(m))
            self.specific_heats.append(c)
            self.susceptibilities.append(chi)
            self.entropies.append(s)
            self.free_energies.append(f)
            self.correlation_lengths.append(xi)
            self.betweenness_centralities.append(bet_cen)
            self.clustering_coefficients.append(clust_coeff)
            self.shortest_paths.append(short_path)
            self.diameters.append(diam)
            self.densities.append(dens)
            self.disparities.append(disp)

    def phase_transition_random(self):
        self.energies = []
        self.magnetisations = []
        self.specific_heats = []
        self.susceptibilities = []
        self.entropies = []
        self.free_energies = []

        for temperature in tqdm(self.temperatures):
            ising = IsingSystem(self.kind_of_graph, self.size, temperature)
            e, m, s, f, c, chi = ising.load_data_random()
            self.energies.append(e)
            self.magnetisations.append(abs(m))
            self.specific_heats.append(c)
            self.susceptibilities.append(chi)
            self.entropies.append(s)
            self.free_energies.append(f)

    # Plot physical quantities
    def plot_physics(self):
        plt.title("Numerical simulation in a " + self.kind_of_graph + " graph")
        plt.scatter(self.temperatures, self.energies)
        plt.xlabel("Temperature")
        plt.ylabel("Energy")
        plt.savefig(self.path_plot + '/energy.pdf')
        plt.close()

        plt.title("Numerical simulation in a " + self.kind_of_graph + " graph")
        plt.scatter(self.temperatures, self.magnetisations)
        plt.xlabel("Temperature")
        plt.ylabel("Magnetisation")
        plt.savefig(self.path_plot + '/magnetisation.pdf')
        plt.close()

        plt.title("Numerical simulation in a " + self.kind_of_graph + " graph")
        plt.scatter(self.temperatures, self.specific_heats)
        plt.xlabel("Temperature")
        plt.ylabel("Specific heat")
        plt.savefig(self.path_plot + '/specific_heat.pdf')
        plt.close()

        plt.title("Numerical simulation in a " + self.kind_of_graph + " graph")
        plt.scatter(self.temperatures, self.susceptibilities)
        plt.xlabel("Temperature")
        plt.ylabel("Susceptibility")
        plt.savefig(self.path_plot + '/susceptibility.pdf')
        plt.close()

        plt.title("Numerical simulation in a " + self.kind_of_graph + " graph")
        plt.scatter(self.temperatures, self.entropies)
        plt.xlabel("Temperature")
        plt.ylabel("Entropy")
        plt.savefig(self.path_plot + '/entropy.pdf')
        plt.close()

        plt.title("Numerical simulation in a " + self.kind_of_graph + " graph")
        plt.scatter(self.temperatures, self.free_energies)
        plt.xlabel("Temperature")
        plt.ylabel("Free energy")
        plt.savefig(self.path_plot + '/free_energy.pdf')
        plt.close()

        plt.title("Numerical simulation in a " + self.kind_of_graph + " graph")
        plt.scatter(self.temperatures, self.correlation_lengths)
        plt.xlabel("Temperature")
        plt.ylabel("Correlation length")
        plt.savefig(self.path_plot + '/correlation_length.pdf')
        plt.close()

        plt.title("Numerical simulation in a " + self.kind_of_graph + " graph")
        plt.scatter(self.temperatures, self.betweenness_centralities)
        plt.xlabel("Temperature")
        plt.ylabel("Betweeness centrality")
        plt.savefig(self.path_plot + '/betweeness_centrality.pdf')
        plt.close()

        plt.title("Numerical simulation in a " + self.kind_of_graph + " graph")
        plt.scatter(self.temperatures, self.clustering_coefficients)
        plt.xlabel("Temperature")
        plt.ylabel("Clustering coefficient")
        plt.savefig(self.path_plot + '/clustering_coefficients.pdf')
        plt.close()

        plt.title("Numerical simulation in a " + self.kind_of_graph + " graph")
        plt.scatter(self.temperatures, self.shortest_paths)
        plt.xlabel("Temperature")
        plt.ylabel("Shortest path")
        plt.savefig(self.path_plot + '/shortest_paths.pdf')
        plt.close()

        plt.title("Numerical simulation in a " + self.kind_of_graph + " graph")
        plt.scatter(self.temperatures, self.diameters)
        plt.xlabel("Temperature")
        plt.ylabel("Diameter")
        plt.savefig(self.path_plot + '/diameter.pdf')
        plt.close()

        plt.title("Numerical simulation in a " + self.kind_of_graph + " graph")
        plt.scatter(self.temperatures, self.densities)
        plt.xlabel("Temperature")
        plt.ylabel("Density")
        plt.savefig(self.path_plot + '/density.pdf')
        plt.close()

        plt.title("Numerical simulation in a " + self.kind_of_graph + " graph")
        plt.scatter(self.temperatures, self.disparities)
        plt.xlabel("Temperature")
        plt.ylabel("Disparity")
        plt.savefig(self.path_plot + '/disparity.pdf')
        plt.close()

    def plot_physics_random(self):
        plt.title("Numerical simulation in a " + self.kind_of_graph + " graph")
        plt.scatter(self.temperatures, self.energies)
        plt.xlabel("Temperature")
        plt.ylabel("Energy")
        plt.savefig(self.path_plot + '/energy.pdf')
        plt.close()

        plt.title("Numerical simulation in a " + self.kind_of_graph + " graph")
        plt.scatter(self.temperatures, self.magnetisations)
        plt.xlabel("Temperature")
        plt.ylabel("Magnetisation")
        plt.ylim(-0.5, 0.5)
        plt.savefig(self.path_plot + '/magnetisation.pdf')
        plt.close()

        plt.title("Numerical simulation in a " + self.kind_of_graph + " graph")
        plt.scatter(self.temperatures, self.specific_heats)
        plt.xlabel("Temperature")
        plt.ylabel("Specific heat")
        plt.savefig(self.path_plot + '/specific_heat.pdf')
        plt.close()

        plt.title("Numerical simulation in a " + self.kind_of_graph + " graph")
        plt.scatter(self.temperatures, self.susceptibilities)
        plt.xlabel("Temperature")
        plt.ylabel("Susceptibility")
        plt.savefig(self.path_plot + '/susceptibility.pdf')
        plt.close()

        plt.title("Numerical simulation in a " + self.kind_of_graph + " graph")
        plt.scatter(self.temperatures, self.entropies)
        plt.xlabel("Temperature")
        plt.ylabel("Entropy")
        plt.savefig(self.path_plot + '/entropy.pdf')
        plt.close()

        plt.title("Numerical simulation in a " + self.kind_of_graph + " graph")
        plt.scatter(self.temperatures, self.free_energies)
        plt.xlabel("Temperature")
        plt.ylabel("Free energy")
        plt.savefig(self.path_plot + '/free_energy.pdf')
        plt.close()

##### RUN #####

temperatures = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10.0]

# Possible choices are "1d", "2d_square", "2d_triangular", "2d_hexagonal"
kind_of_graph = "2d_triangular" 

# Chosen values are 100 for 1d, 10 for 2d_square, 14 for 2d_triangular, 8 for 2d_hexagonal
size = 14

# Chosen values are evolution_time = 4000, average_time = 1000
evolution_time = 4000
average_time = 1000

##### ANIMATION #####

def animation():
    for t in tqdm(temperatures):
        ising = IsingSystem(kind_of_graph , size, t)
        ising.evolution_animation(evolution_time, average_time)
        ising.plot_network_animation()

##### GRAPH PLOT #####

def graph_plot():
    for t in tqdm(temperatures):
        ising = IsingSystem(kind_of_graph , size, t)
        ising.evolution_animation(evolution_time, average_time)
        ising.plot_graph()

##### WEIGHTED GRAPH PLOT #####

def weighted_graph_plot():
    for t in tqdm(temperatures):
        ising = IsingSystem(kind_of_graph, size, t)
        ising.evolution(evolution_time, average_time)
        two_points = ising.get_two_points_function()
        weighted_graph = WeightedGraph(kind_of_graph, size, t)
        weighted_graph.create_graph(two_points)
        weighted_graph.plot_graph()

##### SAVE DATA #####

def save_data():
    for t in tqdm(temperatures):
        ising = IsingSystem(kind_of_graph, size, t)
        ising.save_data()

def save_data_random():
    for t in tqdm(temperatures):
        ising = IsingSystem(kind_of_graph, size, t)
        ising.save_data_random()
        
##### PHASE TRANSITION ######

def phase_transition():
    ising = TemperatureBehaviour(kind_of_graph, size, temperatures)
    ising.phase_transition()
    ising.plot_physics()

def phase_transition_random():
    ising = TemperatureBehaviour(kind_of_graph, size, temperatures)
    ising.phase_transition_random()
    ising.plot_physics_random()

#####

# For 1d graph

#save_data_random()
#phase_transition_random()

# For 2d graphs

#save_data()
#phase_transition()


