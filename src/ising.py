import matplotlib.pyplot as plt
import scipy as sci
import networkx as nx
import numpy as np
import pickle
import os

from tqdm import tqdm

class IsingSystem:
    def __init__(self, kind_of_graph, size):
        # Implement feature of the graph
        self.kind_of_graph = kind_of_graph
        self.size = size

        # Implement paths to save/load data
        self.path_data = './data'
        if not os.path.exists(self.path_data):
            os.mkdir(self.path_data)
        self.path_data = './data/data_graph'
        if not os.path.exists(self.path_data):
            os.mkdir(self.path_data)
        self.path_data = './data/data_graph/' + self.kind_of_graph
        if not os.path.exists(self.path_data):
            os.mkdir(self.path_data)
        self.path_data = './data/data_graph/' + self.kind_of_graph + '/' + str(self.size)
        if not os.path.exists(self.path_data):
            os.mkdir(self.path_data)

        self.path_data_eq = './data/data_equilibrium'
        if not os.path.exists(self.path_data_eq):
            os.mkdir(self.path_data_eq)
        self.path_data_eq = './data/data_equilibrium/' + self.kind_of_graph
        if not os.path.exists(self.path_data_eq):
            os.mkdir(self.path_data_eq)
        self.path_data_eq = './data/data_equilibrium/' + self.kind_of_graph + '/' + str(self.size)
        if not os.path.exists(self.path_data_eq):
            os.mkdir(self.path_data_eq)

        # Implement paths to save/load plots
        self.path_plot = './plot'
        if not os.path.exists(self.path_plot):
            os.mkdir(self.path_plot)
        self.path_plot = './plot/plot_equilibrium'
        if not os.path.exists(self.path_plot):
            os.mkdir(self.path_plot)
        self.path_plot = './plot/plot_equilibrium/' + self.kind_of_graph
        if not os.path.exists(self.path_plot):
            os.mkdir(self.path_plot)
        self.path_plot = './plot/plot_equilibrium/' + self.kind_of_graph + '/' + str(self.size)
        if not os.path.exists(self.path_plot):
            os.mkdir(self.path_plot)

    ###### GRAPH FEATURES ######

    # Calculate features of the graph
    def _calculate_neighbours(self):
        self.neighbours = []
        node_indices = {node: idx for idx, node in enumerate(self.nodes)}  
        for node in tqdm(self.nodes):
            neighbours_of_node = []
            for neighbour_of_node in self.graph.neighbors(node):
                neighbour_index = node_indices.get(neighbour_of_node)
                if neighbour_index is not None:
                    neighbours_of_node.append(neighbour_index)
            self.neighbours.append(neighbours_of_node)

    def _calculate_radii(self):
        radii_dictionary = dict(nx.all_pairs_shortest_path_length(self.graph))
        num_nodes = len(self.graph.nodes)
        self.radii  = np.full((num_nodes, num_nodes), np.inf)
        node_to_index = {node: i for i, node in enumerate(self.graph.nodes())}
        for source, paths in tqdm(radii_dictionary.items()):
            source_index = node_to_index[source]
            for target, distance in paths.items():
                target_index = node_to_index[target]
                self.radii[source_index, target_index] = distance

    # Save data of the graph
    def save_graph(self, probability = 0.4):
        # Choose among different kind of graphs
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
        elif self.kind_of_graph == "4d":
            self.graph =  nx.grid_graph([self.size, self.size, self.size, self.size])

        # Implement properties of the graph: nodes, neighbours of each node, radii among all nodes
        self.number_of_nodes = self.graph.number_of_nodes()
        self.nodes = self.graph.nodes()
        self._calculate_neighbours()
        self._calculate_radii()

        graph_path = self.path_data + '/graph.pickle'
        with open(graph_path, 'wb') as f:
            pickle.dump(self.graph, f)
        nodes_path = self.path_data + '/nodes.pickle'
        with open(nodes_path, 'wb') as f:
            pickle.dump(self.nodes, f)
        neighbours_path = self.path_data + '/neighbours.pickle'
        with open(neighbours_path, 'wb') as f:
            pickle.dump(self.neighbours, f)
        radii_path = self.path_data + '/radii.pickle'
        with open(radii_path, 'wb') as f:
            pickle.dump(self.radii, f)

    # Load data of the graph
    def load_graph(self):
        graph_path = self.path_data + '/graph.pickle'
        with open(graph_path, 'rb') as f:
            self.graph = pickle.load(f)
        nodes_path = self.path_data + '/nodes.pickle'
        with open(nodes_path, 'rb') as f:
            self.nodes = pickle.load(f)
        neighbours_path = self.path_data + '/neighbours.pickle'
        with open(neighbours_path, 'rb') as f:
            self.neighbours = pickle.load(f)
        radii_path = self.path_data + '/radii.pickle'
        with open(radii_path, 'rb') as f:
            self.radii = pickle.load(f)
        self.number_of_nodes = self.graph.number_of_nodes()

    ###### CALCULATE PHYSICAL QUANTITIES ######

    # Initialise hot spins: randomly aligned
    def initialise_spins_hot(self):
        self.spins = np.random.choice([-1, 1], self.number_of_nodes)
        self._calculate_magnetisation()
        self._calculate_energy()

    # Initialise cold spins: aligned upwards
    def initialise_spins_cold(self):
        self.spins = np.full(self.number_of_nodes, 1)
        self._calculate_magnetisation()
        self._calculate_energy()

    # Set physical quantities in the Hamiltonian: J, B and T
    def set_physics(self, temperature, J=1, B=0):
        self.temperature = temperature
        self.J = J
        self.B = B
        
    # Calculate energy using with the Hamiltonian
    def _calculate_energy(self):
        neighbour_spins = 0
        for i in range(self.number_of_nodes):
            for j in self.neighbours[i]:
                neighbour_spins += self.spins[i] * self.spins[j]
        self.energy = - self.J * neighbour_spins / 2 - self.B * np.sum(self.spins)

    # Calculate magnetisation by definition
    def _calculate_magnetisation(self):
        self.magnetisation = np.sum(self.spins)

    # Calculate entropy using the Stirling approximation 
    def _calculate_entropy(self):
        spin_up = np.sum(self.spins > 0)
        spin_down = np.sum(self.spins < 0)
        # if spin_up or spin_down is 0, set them to 1 in order to have log(1) = 0
        spin_up = max(spin_up, 1) 
        spin_down = max(spin_down, 1)
        self.entropy = self.number_of_nodes * np.log(self.number_of_nodes) - spin_up * np.log(spin_up) - spin_down * np.log(spin_down)
        
    # Calculate correlation function over radius by definition
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
        
    # Calculate correlation length by fitting negative exponential
    def fit_function(self, x, xi, a, b):
            return a * np.exp(- x / xi) + b
    
    def _calculate_correlation_length(self):
        if np.std(self.correlation_function) < 0.05:
            self.correlation_length = 0
        else:
            r = np.arange(int(np.max(self.radii)))
            popt, pcov = sci.optimize.curve_fit(self.fit_function, r, self.correlation_function, p0 = [1, 1, 0])
            self.correlation_length = popt[0]

    # Calculate two points correlation function, where part1 is <spin_i * spin_j> and part1 only <spin_i>
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

    # Metropolis algorithm (scrools all the spins to flip) for evolution 
    def evolution(self, evolution_time):
        self.energies = []
        self.magnetisations = []
        self.entropies = []
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
            self.energies.append(self.energy)
            self._calculate_magnetisation()
            self.magnetisations.append(self.magnetisation)
            self._calculate_entropy()
            self.entropies.append(self.entropy)
        
    def evolution_correlation(self, evolution_time, average_time):
        self._calculate_energy()
        self.correlation_lengths = []

        for _ in range(evolution_time):
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
            self._calculate_correlation_function()
            self._calculate_correlation_length()
            self.correlation_lengths.append(self.correlation_length)
       
    def evolution_weighted_graph(self, evolution_time, average_time):
        self._calculate_energy()
        self.two_points_function_part1 = np.zeros([self.number_of_nodes, self.number_of_nodes])
        self.two_points_function_part2 = np.zeros(self.number_of_nodes)

        for _ in range(evolution_time):
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
            self._calculate_two_points_function()
        self._calculate_two_points_function_average(average_time)
            
    # Glauber algorithm (chooses randomly what spin to flip) for evolution
    def evolution_random(self, evolution_time):
        self.energies = []
        self.magnetisations = []
        self.entropies = []
        self._calculate_energy()

        choices = np.random.choice([i for i in range(self.number_of_nodes)], evolution_time)
        random_numbers = np.random.rand(evolution_time)
        for n in tqdm(range(evolution_time)):
            i = choices[n]
            neighbour_spins = self.spins[i] * sum([self.spins[j] for j in self.neighbours[i]])
            delta_energy = 2.0 * self.J * neighbour_spins + 2 * self.B * neighbour_spins
            transition_probability = np.exp(- delta_energy / self.temperature)
            if delta_energy < 0 or transition_probability > random_numbers[i]:
                self.spins[i] *= -1
                self.energy += delta_energy
            if n % 1000 == 0:
                self.energies.append(self.energy)
                self._calculate_magnetisation()
                self.magnetisations.append(self.magnetisation)
                self._calculate_entropy()
                self.entropies.append(self.entropy)

    def evolution_random_correlation(self, evolution_time, average_time):
        self._calculate_energy()
        self.correlation_lengths = []

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
        for n in tqdm(range(average_time)):
            i = choices[n]
            neighbour_spins = self.spins[i] * sum([self.spins[j] for j in self.neighbours[i]])
            delta_energy = 2.0 * self.J * neighbour_spins + 2 * self.B * neighbour_spins
            transition_probability = np.exp(- delta_energy / self.temperature)
            if delta_energy < 0 or transition_probability > random_numbers[i]:
                self.spins[i] *= -1
                self.energy += delta_energy
            if n % 1000 == 0:
                self._calculate_correlation_function()
                self._calculate_correlation_length()
                self.correlation_lengths.append(self.correlation_length)

    def evolution_random_weighted_graph(self, evolution_time, average_time):
        self._calculate_energy()
        self.two_points_function_part1 = np.zeros([self.number_of_nodes, self.number_of_nodes])
        self.two_points_function_part2 = np.zeros(self.number_of_nodes)

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
        for n in tqdm(range(average_time)):
            i = choices[n]
            neighbour_spins = self.spins[i] * sum([self.spins[j] for j in self.neighbours[i]])
            delta_energy = 2.0 * self.J * neighbour_spins + 2 * self.B * neighbour_spins
            transition_probability = np.exp(- delta_energy / self.temperature)
            if delta_energy < 0 or transition_probability > random_numbers[i]:
                self.spins[i] *= -1
                self.energy += delta_energy
            if n % 1000 == 0:
                self._calculate_two_points_function()
        self._calculate_two_points_function_average(average_time)

    ##### SAVE/PLOT DATA #####

    # Save data of thermalisation: energy, magnetisation and entropy
    def save_data(self):
        energy_path = self.path_data_eq + '/energy_temp' + str(float(self.temperature))[-5:] + '.pickle'
        with open(energy_path, 'wb') as f:
            pickle.dump(self.energies, f)
        
        magnetisation_path = self.path_data_eq + '/magnetisation_temp' + str(float(self.temperature))[-5:] + '.pickle'
        with open(magnetisation_path, 'wb') as f:
            pickle.dump(self.magnetisations, f)

        entropy_path = self.path_data_eq + '/entropy_temp' + str(float(self.temperature))[-5:] + '.pickle'
        with open(entropy_path, 'wb') as f:
            pickle.dump(self.entropies, f)

    # Load data of thermalisation: energy, magnetisation and entropy
    def load_data(self):
        energy_path = self.path_data_eq + '/energy_temp' + str(float(self.temperature))[-5:] + '.pickle'
        with open(energy_path, 'rb') as f:
            self.energies = pickle.load(f)

        magnetisation_path = self.path_data_eq + '/magnetisation_temp' + str(float(self.temperature))[-5:] + '.pickle'
        with open(magnetisation_path, 'rb') as f:
            self.magnetisations = pickle.load(f)

        entropy_path = self.path_data_eq + '/entropy_temp' + str(float(self.temperature))[-5:] + '.pickle'
        with open(entropy_path, 'rb') as f:
            self.entropies = pickle.load(f)

    # Save data of thermalisation: correlation_function and correlation length
    def save_data_correlation(self):
        correlation_function_path = self.path_data_eq + '/correlation_function_temp' + str(float(self.temperature))[-5:] + '.pickle'
        with open(correlation_function_path, 'wb') as f:
            pickle.dump(self.correlation_function, f)

        correlation_length_path = self.path_data_eq + '/correlation_length_temp' + str(float(self.temperature))[-5:] + '.pickle'
        with open(correlation_length_path, 'wb') as f:
            pickle.dump(self.correlation_lengths, f)
            
    # Load data of thermalisation: correlation_function and orrelation length
    def load_data_correlation(self):
        correlation_function_path = self.path_data_eq + '/correlation_function_temp' + str(float(self.temperature))[-5:] + '.pickle'
        with open(correlation_function_path, 'rb') as f:
            self.correlation_function = pickle.load(f)

        correlation_length_path = self.path_data_eq + '/correlation_length_temp' + str(float(self.temperature))[-5:] + '.pickle'
        with open(correlation_length_path, 'rb') as f:
            self.correlation_lengths = pickle.load(f)

    # Save data of thermalisation: two points function
    def save_data_weighted_graph(self):
        two_points_function_path = self.path_data_eq + '/two_points_function_temp' + str(float(self.temperature))[-5:] + '.pickle'
        with open(two_points_function_path, 'wb') as f:
            pickle.dump(self.two_points_function, f)
            
    # Load data of thermalisation: two points function
    def load_data_weighted_graph(self):
        two_points_function_path = self.path_data_eq + '/two_points_function_temp' + str(float(self.temperature))[-5:] + '.pickle'
        with open(two_points_function_path, 'rb') as f:
            self.two_points_function = pickle.load(f)

    # Plot physics on file: : energy, magnetisation and entropy
    def plot_physics(self):
        energies = [energy / self.number_of_nodes for energy in self.energies]
        magnetisations = [magnetisation / self.number_of_nodes for magnetisation in self.magnetisations]
        entropies = [entropy / self.number_of_nodes for entropy in self.entropies]

        plt.title("Energy over time in a " + self.kind_of_graph + " graph")
        plt.plot(energies)
        plt.xlabel(f"Time x {self.number_of_nodes}")
        plt.ylabel("Energy")
        plt.savefig(self.path_plot + '/energy_temp' + str(float(self.temperature))[-5:] + '.pdf')
        plt.close()

        plt.title("Magnetisation over time in a " + self.kind_of_graph + " graph")
        plt.plot(magnetisations)
        plt.xlabel(f"Time x {self.number_of_nodes}")
        plt.ylabel("Magnetisation")
        plt.savefig(self.path_plot + '/magnetisation_temp' + str(float(self.temperature))[-5:] + '.pdf')
        plt.close()

        plt.title("Entropy over time in a " + self.kind_of_graph + " graph")
        plt.plot(entropies)
        plt.xlabel(f"Time x {self.number_of_nodes}")
        plt.ylabel("Entropy")
        plt.savefig(self.path_plot + '/entropy_temp' + str(float(self.temperature))[-5:] + '.pdf')
        plt.close()

    # Plot physics on file:: correlation_function and correlation length
    def plot_physics_correlation(self):
        plt.title("Correlation function over radius in a " + self.kind_of_graph + " graph")
        plt.plot(self.correlation_function)
        plt.xlabel("Radius")
        plt.ylabel("Correlation function")
        plt.savefig(self.path_plot + '/correlation_function_temp' + str(float(self.temperature))[-5:] + '.pdf')
        plt.close()

        plt.title("Correlation length over time in a " + self.kind_of_graph + " graph")
        plt.plot(self.correlation_lengths)
        plt.xlabel(f"Time x {self.number_of_nodes}")
        plt.ylabel("Correlation length")
        plt.savefig(self.path_plot + '/correlation_length_temp' + str(float(self.temperature))[-5:] + '.pdf')
        plt.close()

    # Compute and get physical quantities by means of averages
    def get_average_physics(self, average_time):
        energies = [energy / self.number_of_nodes for energy in self.energies]
        magnetisations = [magnetisation / self.number_of_nodes for magnetisation in self.magnetisations]
        entropies = [entropy / self.number_of_nodes for entropy in self.entropies]

        magnetisation_data = magnetisations[-average_time:]
        magnetisation = np.mean(magnetisation_data)
        magnetisation_square = np.mean([magnetisation**2 for magnetisation in magnetisation_data])

        energy_data = energies[-average_time:]
        energy = np.mean(energy_data)
        energy_square = np.mean([energy**2 for energy in energy_data])

        specific_heat = (energy_square - energy**2) / self.temperature / self.temperature
        
        susceptibility = (magnetisation_square - magnetisation**2) / self.temperature

        entropy = np.mean(entropies[-average_time:])

        free_energy = energy - self.temperature * entropy

        return energy, magnetisation, specific_heat, susceptibility, entropy, free_energy

    def get_average_physics_correlation(self, average_time):
        correlation_length_data = self.correlation_lengths[-average_time:]
        correlation_length = np.mean(correlation_length_data)

        return correlation_length

    def get_two_points_function(self):
        return self.two_points_function

class WeightedGraph:
    def __init__(self, kind_of_graph, size, temperature):
        # Implement feature of the graph
        self.kind_of_graph = kind_of_graph
        self.size = size
        self.temperature = temperature

        # Implement paths to save/load data
        self.path_data = './data'
        if not os.path.exists(self.path_data):
            os.mkdir(self.path_data)
        self.path_data = './data/data_properties'
        if not os.path.exists(self.path_data):
            os.mkdir(self.path_data)
        self.path_data = './data/data_properties/' + self.kind_of_graph
        if not os.path.exists(self.path_data):
            os.mkdir(self.path_data)
        self.path_data = './data/data_properties/' + self.kind_of_graph + '/' + str(self.size)
        if not os.path.exists(self.path_data):
            os.mkdir(self.path_data)

    # Create weighted graph from the two point function
    def create_graph(self, two_point_function):
        self.two_point_function = abs(two_point_function)
        self.graph = nx.from_numpy_array(self.two_point_function)
        self.number_of_nodes = self.graph.number_of_nodes()

    # Calculate average betweeness centrality with networkx
    def _calculate_betweeness_centrality(self):
        betweenness_centrality = nx.betweenness_centrality(self.graph, k=None, weight='weight')
        self.betweenness_centrality = sum(betweenness_centrality.values()) / self.number_of_nodes

    # Calculate average clustering coefficient with networkx
    def _calculate_clustering_coefficient(self):
        self.clustering_coefficient = nx.average_clustering(self.graph, weight='weight')

    # Calculate average shortest path with networkx
    def _calculate_shortest_path(self):
        self.shortest_path = nx.average_shortest_path_length(self.graph,  weight='weight')

    # Calculate diameter with networkx
    def _calculate_diameter(self):
        eccentricity = nx.eccentricity(self.graph, weight="weight")
        self.diameter = np.max(list(eccentricity.values()))

    # Calculate density using the two point function
    def _calculate_density(self):
        two_point_function_row = np.zeros(self.number_of_nodes)
        for i in range(self.number_of_nodes):
            for j in range(self.number_of_nodes):
                two_point_function_row[i] += self.two_point_function[i][j]

        self.density = sum(two_point_function_row) / self.number_of_nodes**2

    # Calculate disparity using the two point function
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
    
    # Compute all properties
    def compute_properties(self):
        self._calculate_betweeness_centrality()
        self._calculate_clustering_coefficient()
        self._calculate_shortest_path()
        self._calculate_diameter()
        self._calculate_density()
        self._calculate_disparity()

        self.properties = [self.betweenness_centrality, self.clustering_coefficient, self.shortest_path, self.diameter, self.density, self.disparity]

    # Save data
    def save_data(self):
        properties_path = self.path_data + '/properties_temp' + str(float(self.temperature))[-5:] + '.pickle'
        with open(properties_path, 'wb') as f:
            pickle.dump(self.properties, f)
            
    # Load data
    def load_data(self):
        properties_path = self.path_data + '/properties_temp' + str(float(self.temperature))[-5:] + '.pickle'
        with open(properties_path, 'rb') as f:
            self.properties = pickle.load(f)

    # Get all properties
    def get_network_properties(self):
        return self.properties

class TemperatureBehaviour:
    def __init__(self, kind_of_graph, size):
        # Implement feature of the graph
        self.kind_of_graph = kind_of_graph
        self.size = size

        # Implement paths to save/load data
        self.path_data = './data/data_phase_transition'
        if not os.path.exists(self.path_data):
            os.mkdir(self.path_data)
        self.path_data = './data/data_phase_transition/' + self.kind_of_graph
        if not os.path.exists(self.path_data):
            os.mkdir(self.path_data)
        self.path_data = './data/data_phase_transition/' + self.kind_of_graph + '/' + str(self.size)
        if not os.path.exists(self.path_data):
            os.mkdir(self.path_data)

        # Implement paths to save/load plots
        self.path_plot = './plot'
        if not os.path.exists(self.path_plot):
            os.mkdir(self.path_plot)
        self.path_plot = './plot/plot_phase_transition'
        if not os.path.exists(self.path_plot):
            os.mkdir(self.path_plot)
        self.path_plot = './plot/plot_phase_transition/' + self.kind_of_graph
        if not os.path.exists(self.path_plot):
            os.mkdir(self.path_plot)
        self.path_plot = './plot/plot_phase_transition/' + self.kind_of_graph + '/' + str(self.size)
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
        self.entropies = []
        self.free_energies = []
        for temperature in self.temperatures:
            ising = IsingSystem(self.kind_of_graph, self.size)
            ising.load_graph()
            ising.set_physics(temperature)
            ising.load_data()
            e, m, c, chi, s, f = ising.get_average_physics(average_time)
            self.energies.append(e)
            self.magnetisations.append(abs(m))
            self.specific_heats.append(c)
            self.susceptibilities.append(chi)
            self.entropies.append(s)
            self.free_energies.append(f)
        
    def phase_transition_correlation(self, average_time):
        self.correlation_lengths = []
        for temperature in self.temperatures:
            ising = IsingSystem(self.kind_of_graph, self.size)
            ising.load_graph()
            ising.set_physics(temperature)
            ising.load_data_correlation()
            xi = ising.get_average_physics_correlation(average_time)
            self.correlation_lengths.append(xi)

    def phase_transition_weighted_graph(self):
        self.betweeness_centralities = []
        self.clustering_coefficients = []
        self.shortest_paths = []
        self.diameters = []
        self.densities = []
        self.disparities = []
        for temperature in self.temperatures:
            weighted_graph = WeightedGraph(self.kind_of_graph, self.size, temperature)
            weighted_graph.load_data()
            bet_cen, clust_coeff, short_path, diam, dens, disp = weighted_graph.get_network_properties()
            self.betweeness_centralities.append(bet_cen)
            self.clustering_coefficients.append(clust_coeff)
            self.shortest_paths.append(short_path)
            self.diameters.append(diam)
            self.densities.append(dens)
            self.disparities.append(disp)
        
    # Save data of phase transition: energy, magnetisation, specific heat, susceptibility, entropy, free energy
    def save_data(self):
        energy_path = self.path_data + '/energy.pickle'
        with open(energy_path, 'wb') as f:
            pickle.dump(self.energies, f)

        magnetisation_path = self.path_data + '/magnetisation.pickle'
        with open(magnetisation_path, 'wb') as f:
            pickle.dump(self.magnetisations, f)

        specific_heat_path = self.path_data + '/specific_heat.pickle'
        with open(specific_heat_path, 'wb') as f:
            pickle.dump(self.specific_heats, f)

        susceptibility_path = self.path_data + '/susceptibility.pickle'
        with open(susceptibility_path, 'wb') as f:
            pickle.dump(self.susceptibilities, f)

        entropy_path = self.path_data + '/entropy.pickle'
        with open(entropy_path, 'wb') as f:
            pickle.dump(self.entropies, f)

        free_energy_path = self.path_data + '/free_energy.pickle'
        with open(free_energy_path, 'wb') as f:
            pickle.dump(self.free_energies, f)

    # Load data of phase transition: energy, magnetisation, specific heat, susceptibility, entropy, free energy
    def load_data(self):
        energy_path = self.path_data + '/energy.pickle'
        with open(energy_path, 'rb') as f:
            self.energies = pickle.load(f)

        magnetisation_path = self.path_data + '/magnetisation.pickle'
        with open(magnetisation_path, 'rb') as f:
            self.magnetisations = pickle.load(f)

        specific_heat_path = self.path_data + '/specific_heat.pickle'
        with open(specific_heat_path, 'rb') as f:
            self.specific_heats = pickle.load(f)

        susceptibility_path = self.path_data + '/susceptibility.pickle'
        with open(susceptibility_path, 'rb') as f:
            self.susceptibilities = pickle.load(f)

        entropy_path = self.path_data + '/entropy.pickle'
        with open(entropy_path, 'rb') as f:
            self.entropies = pickle.load(f)

        free_energy_path = self.path_data + '/free_energy.pickle'
        with open(free_energy_path, 'rb') as f:
            self.free_energies = pickle.load(f)

    # Save data of phase transition: correlation length
    def save_data_correlation(self):
        correlation_length_path = self.path_data + '/correlation_length.pickle'
        with open(correlation_length_path, 'wb') as f:
            pickle.dump(self.correlation_lengths, f)

    # Load data of phase transition: correlation length
    def load_data_correlation(self):
        correlation_length_path = self.path_data + '/correlation_length.pickle'
        with open(correlation_length_path, 'rb') as f:
            self.correlation_lengths = pickle.load(f)

    # Save data of phase transition: weighted graph properties
    def save_data_weighted_graph(self):
        betweeness_centrality_path = self.path_data + '/betweeness_centrality.pickle'
        with open(betweeness_centrality_path, 'wb') as f:
            pickle.dump(self.betweeness_centralities, f)

        clustering_coefficient_path = self.path_data + '/clustering_coefficient.pickle'
        with open(clustering_coefficient_path, 'wb') as f:
            pickle.dump(self.clustering_coefficients, f)

        shortest_path_path = self.path_data + '/shortest_path.pickle'
        with open(shortest_path_path, 'wb') as f:
            pickle.dump(self.shortest_paths, f)

        diameter_path = self.path_data + '/diameter.pickle'
        with open(diameter_path, 'wb') as f:
            pickle.dump(self.diameters, f)

        density_path = self.path_data + '/density.pickle'
        with open(density_path, 'wb') as f:
            pickle.dump(self.densities, f)

        disparity_path = self.path_data + '/disparity.pickle'
        with open(disparity_path, 'wb') as f:
            pickle.dump(self.disparities, f)

    # Load data of phase transition: weighted graph properties
    def load_data_weighted_graph(self):
        betweeness_centrality_path = self.path_data + '/betweeness_centrality.pickle'
        with open(betweeness_centrality_path, 'rb') as f:
            self.betweeness_centralities = pickle.load(f)

        clustering_coefficient_path = self.path_data + '/clustering_coefficient.pickle'
        with open(clustering_coefficient_path, 'rb') as f:
            self.clustering_coefficients = pickle.load(f)

        shortest_path_path = self.path_data + '/shortest_path.pickle'
        with open(shortest_path_path, 'rb') as f:
            self.shortest_paths = pickle.load(f)

        diameter_path = self.path_data + '/diameter.pickle'
        with open(diameter_path, 'rb') as f:
            self.diameters = pickle.load(f)

        density_path = self.path_data + '/density.pickle'
        with open(density_path, 'rb') as f:
            self.densities = pickle.load(f)

        disparity_path = self.path_data + '/disparity.pickle'
        with open(disparity_path, 'rb') as f:
            self.disparities = pickle.load(f)

    # Plot physics on file
    def plot_physics(self):
        plt.title("Energy over temperature in a " + self.kind_of_graph + " graph")
        plt.scatter(self.temperatures, self.energies)
        plt.xlabel("Temperature")
        plt.ylabel("Energy")
        plt.savefig(self.path_plot + '/energy.pdf')
        plt.close()

        plt.title("Magnetisation over temperature in a " + self.kind_of_graph + " graph")
        plt.scatter(self.temperatures, self.magnetisations)
        plt.xlabel("Temperature")
        plt.ylabel("Magnetisation")
        plt.savefig(self.path_plot + '/magnetisation.pdf')
        plt.close()

        plt.title("Specific heat over temperature in a " + self.kind_of_graph + " graph")
        plt.scatter(self.temperatures, self.specific_heats)
        plt.xlabel("Temperature")
        plt.ylabel("Specific heat")
        plt.savefig(self.path_plot + '/specific_heat.pdf')
        plt.close()

        plt.title("Susceptibility over temperature in a " + self.kind_of_graph + " graph")
        plt.scatter(self.temperatures, self.susceptibilities)
        plt.xlabel("Temperature")
        plt.ylabel("Susceptibility")
        plt.savefig(self.path_plot + '/susceptibility.pdf')
        plt.close()

        plt.title("Entropy over temperature in a " + self.kind_of_graph + " graph")
        plt.scatter(self.temperatures, self.entropies)
        plt.xlabel("Temperature")
        plt.ylabel("Entropy")
        plt.savefig(self.path_plot + '/entropy.pdf')
        plt.close()

        plt.title("Free energy over temperature in a " + self.kind_of_graph + " graph")
        plt.scatter(self.temperatures, self.free_energies)
        plt.xlabel("Temperature")
        plt.ylabel("Free energy")
        plt.savefig(self.path_plot + '/free_energy.pdf')
        plt.close()

    def plot_physics_correlation(self):
        plt.title("Correlation length over temperature in a " + self.kind_of_graph + " graph")
        plt.scatter(self.temperatures, self.correlation_lengths)
        plt.xlabel("Temperature")
        plt.ylabel("Correlation length")
        plt.savefig(self.path_plot + '/correlation_length.pdf')
        plt.close()

    def plot_physics_weighted_graph(self):
        plt.title("Betweeness centrality over temperature in a " + self.kind_of_graph + " graph")
        plt.scatter(self.temperatures, self.betweeness_centralities)
        plt.xlabel("Temperature")
        plt.ylabel("Betweeness centrality")
        plt.savefig(self.path_plot + '/betweeness_centrality.pdf')
        plt.close()

        plt.title("Clustering coefficient over temperature in a " + self.kind_of_graph + " graph")
        plt.scatter(self.temperatures, self.clustering_coefficients)
        plt.xlabel("Temperature")
        plt.ylabel("Clustering coefficient")
        plt.savefig(self.path_plot + '/clustering_coefficients.pdf')
        plt.close()

        plt.title("Shortest path over temperature in a " + self.kind_of_graph + " graph")
        plt.scatter(self.temperatures, self.shortest_paths)
        plt.xlabel("Temperature")
        plt.ylabel("Shortest path")
        plt.savefig(self.path_plot + '/shortest_paths.pdf')
        plt.close()

        plt.title("Diameter over temperature in a " + self.kind_of_graph + " graph")
        plt.scatter(self.temperatures, self.diameters)
        plt.xlabel("Temperature")
        plt.ylabel("Diameter")
        plt.savefig(self.path_plot + '/diameter.pdf')
        plt.close()

        plt.title("Density over temperature in a " + self.kind_of_graph + " graph")
        plt.scatter(self.temperatures, self.densities)
        plt.xlabel("Temperature")
        plt.ylabel("Density")
        plt.savefig(self.path_plot + '/density.pdf')
        plt.close()

        plt.title("Disparity over temperature in a " + self.kind_of_graph + " graph")
        plt.scatter(self.temperatures, self.disparities)
        plt.xlabel("Temperature")
        plt.ylabel("Disparity")
        plt.savefig(self.path_plot + '/disparity.pdf')

import multiprocessing
import time

graph = "4d" 
size = 10

temperatures = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 
                2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0]

#temperatures = [2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0,
#                4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0]




def evolution(t):
    ising = IsingSystem(graph, size)
    ising.load_graph()
    ising.set_physics(t)
    ising.initialise_spins_hot()
    evolution_time = 4000
    ising.evolution(evolution_time)
    ising.save_data()
    ising.load_data()
    ising.plot_physics()

def evolution_correlation(t):
    ising = IsingSystem(graph, size)
    ising.load_graph()
    ising.set_physics(t)
    ising.initialise_spins_hot()
    evolution_time = 3000
    average_time = 1000
    ising.evolution_correlation(evolution_time, average_time)
    ising.save_data_correlation()
    ising.plot_physics_correlation()

def evolution_weighted_graph(t):
    ising = IsingSystem(graph, size)
    ising.load_graph()
    ising.set_physics(t)
    ising.initialise_spins_hot()
    evolution_time = 3000
    average_time = 1000
    ising.evolution_weighted_graph(evolution_time, average_time)
    ising.save_data_weighted_graph()

def weighted_graph(t):
    ising = IsingSystem(graph, size)
    ising.load_graph()
    ising.set_physics(t)
    ising.load_data_weighted_graph()
    two_point = ising.get_two_points_function()
    weighted_graph = WeightedGraph(graph, size, t)
    weighted_graph.create_graph(two_point)
    weighted_graph.compute_properties()
    weighted_graph.save_data()

def plot():
    temp = TemperatureBehaviour(graph, size)
    temp.set_temperatures(temperatures)
    average_time = 1000
    temp.phase_transition(average_time)
    temp.save_data()
    temp.plot_physics()

def plot_correlation():
    temp = TemperatureBehaviour(graph, size)
    temp.set_temperatures(temperatures)
    average_time = 1000
    temp.phase_transition_correlation(average_time)
    temp.save_data_correlation()
    temp.plot_physics_correlation()

def plot_weighted_graph():
    temp = TemperatureBehaviour(graph, size)
    temp.set_temperatures(temperatures)
    temp.phase_transition_weighted_graph()
    temp.save_data_weighted_graph()
    temp.plot_physics_weighted_graph()

if __name__ == "__main__":

    # Save graph
    #ising = IsingSystem(graph, size)
    #ising.save_graph()

    # Evolution
    #start = time.time()
    #num_processes = 4
    #with multiprocessing.Pool(processes=num_processes) as pool:
    #    pool.map(evolution, temperatures)
    #end = time.time()
    #print(end - start)
    #
    #start = time.time()
    #num_processes = 4
    #with multiprocessing.Pool(processes=num_processes) as pool:
    #    pool.map(evolution_correlation, temperatures)
    #end = time.time()
    #print(end - start)

    #start = time.time()
    #num_processes = 4
    #with multiprocessing.Pool(processes=num_processes) as pool:
    #    pool.map(evolution_weighted_graph, temperatures)
    #end = time.time()
    #print(end - start) 
#
    #start = time.time()
    #num_processes = 4
    #with multiprocessing.Pool(processes=num_processes) as pool:
    #    pool.map(weighted_graph, temperatures)
    #end = time.time()
    #print(end - start) 

    plot()
    #plot_correlation()
    #plot_weighted_graph()

        


    
