import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib.animation as ani
import numpy as np
from tqdm import tqdm
import os

class IsingNetwork:
    def __init__(self, graph, temperature, j = 1, b = 0):
        # Implement features of the graph
        self.graph = graph
        self.number_of_nodes = graph.number_of_nodes()
        self.nodes = graph.nodes()
        self.side = int(np.sqrt(self.number_of_nodes))
        # Implement physical quantities
        self._calculate_neighbours()
        self.temperature = temperature
        self.j = j
        self.b = b
        self.energies = []
        self.magnetisations = []
        # Implement graphical representation
        self.field = []

    def _calculate_neighbours(self):
        self.neighbours = []
        list_of_nodes = list(self.nodes)
        for node in self.nodes:
            indices_of_neighbours = []
            for neighbour_of_node in self.graph.neighbors(node):
                j = list_of_nodes.index(neighbour_of_node)
                indices_of_neighbours.append(j)
            self.neighbours.append(indices_of_neighbours)

    def initialise_spins_hot(self):
        self.spins = np.random.choice([-1, 1], self.number_of_nodes)
        self._calculate_magnetisation()
        self._calculate_energy()

    def initialise_spins_cold(self):
        self.spins = np.full(self.number_of_nodes, 1)
        self._calculate_magnetisation()
        self._calculate_energy()
        
    # Compute physical quantities
    def _calculate_magnetisation(self):
        self.magnetisation = sum(self.spins)

    def _calculate_energy(self):
        neighbour_spins = 0
        for i in range(self.number_of_nodes):
            for j in self.neighbours[i]:
                neighbour_spins += self.spins[i] * self.spins[j]
        self.energy = - self.j * neighbour_spins / 2 + self.b * sum(self.spins)

    # Metropolis algorithm scrool all the spins to flip
    def evolution(self, time):
        self.field.append(self._compute_field())
        for _ in range(time):
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
            self.field.append(self._compute_field())
    
    # Glauber algorithm chooses randomly what spin to flip
    def evolution_random(self, time):
        self.field.append(self._compute_field())
        choices = np.random.choice([i for i in range(self.number_of_nodes)], time)
        for n in range(time):
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
            self.field.append(self._compute_field())

    # Get physical quantities
    def plot_physics(self):
        energies = [energy / self.number_of_nodes for energy in self.energies]
        magnetisations = [magnetisation / self.number_of_nodes for magnetisation in self.magnetisations]
        name = "side" + str(self.side) + "_temp" + str(self.temperature)[-4:]
        path = './equilibrium'
        if not os.path.exists(path):
            os.mkdir(path)
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].plot(energies)
        axs[0].set_title("Energies")
        axs[1].plot(magnetisations)
        axs[1].set_title("Magnetisations")
        fig.tight_layout(pad=2.0)
        plt.savefig(path + f'/equilibrium_{name}.pdf')
        plt.close()

    def get_physics(self, average_time):
        self.evolution(average_time)
        magnetisation_data = self.magnetisations[-average_time:]
        energy_data = self.energies[-average_time:]
        magnetisation = np.mean(magnetisation_data)
        energy = np.mean(energy_data)
        magnetisation_square = np.mean([magnetisation * magnetisation for magnetisation in magnetisation_data])
        energy_square = np.mean([energy * energy for energy in energy_data])
        specific_heat = (energy_square - energy * energy) / self.temperature / self.temperature
        susceptibility = (magnetisation_square - magnetisation * magnetisation) / self.temperature
        return energy / self.number_of_nodes, magnetisation / self.number_of_nodes, specific_heat / self.number_of_nodes, susceptibility / self.number_of_nodes
    
    def get_physics_random(self, average_time):
        self.evolution_random(average_time)
        magnetisation_data = self.magnetisations[-average_time:]
        energy_data = self.energies[-average_time:]
        magnetisation = np.mean(magnetisation_data)
        energy = np.mean(energy_data)
        magnetisation_square = np.mean([magnetisation * magnetisation for magnetisation in magnetisation_data])
        energy_square = np.mean([energy * energy for energy in energy_data])
        specific_heat = (energy_square - energy * energy) / self.temperature / self.temperature
        susceptibility = (magnetisation_square - magnetisation * magnetisation) / self.temperature
        return energy / self.number_of_nodes, magnetisation / self.number_of_nodes, specific_heat / self.number_of_nodes, susceptibility / self.number_of_nodes

    #Graphical representations of the network (only for square networks)
    def _compute_field(self):
        size = int(np.sqrt(self.number_of_nodes))
        field = np.zeros((size, size))
        for n in range(self.number_of_nodes):
            field[n // size][n % size] = self.spins[n]
        return field

    def plot_network(self, index):
        plt.imshow(self.field[index], cmap='binary')
        plt.axis("off")
        plt.title(f'time: {index * 10}')
        legend_handles = [
            pat.Patch(color='white', label='- 1'),
            pat.Patch(color='black', label='+ 1')
        ]
        plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1))
        plt.show()

    def plot_network_animation(self):
        name = "side" + str(self.side) + "_temp" + str(self.temperature)[-4:]
        path = './sequence'
        if not os.path.exists(path):
            os.mkdir(path)
        fig, ax = plt.subplots()
        def animate(i):
            ax.imshow(self.field[i], cmap='binary')
            ax.axis("off")
            ax.set_title(f'temperature: {str(self.temperature)[:4]}, time: {i}')
            legend_handles = [
                pat.Patch(color='white', label='- 1'),
                pat.Patch(color='black', label='+ 1')
            ]
            ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1))
        animation = ani.FuncAnimation(fig, animate, frames=len(self.field), interval=200, repeat=False)
        animation.save(path + f'/sequence_{name}.gif')

class IsingPhaseTransition:
    def __init__(self, graph, temperatures):
        self.graph = graph
        # Implement temperatures to scroll
        self.temperatures = temperatures
        # Implement physical quantities
        self.energies = []
        self.magnetisations = []
        self.magnetisations_abs = []
        self.specific_heats = []
        self.susceptibilities = []

    # Compute phase transition
    def phase_transition(self, time, average_time):
        for temperature in tqdm(self.temperatures):
            ising = IsingNetwork(self.graph, temperature)
            ising.initialise_spins_hot()
            ising.evolution(time)
            e, m, h, s = ising.get_physics(average_time)
            self.energies.append(e)
            self.magnetisations.append(m)
            self.magnetisations_abs.append(abs(m))
            self.specific_heats.append(h)
            self.susceptibilities.append(s)
            ising.plot_physics()

    def phase_transition_random(self, time, average_time):
        for temperature in tqdm(self.temperatures):
            ising = IsingNetwork(self.graph, temperature)
            ising.initialise_spins_hot()
            ising.evolution_random(time)
            e, m, h, s = ising.get_physics_random(average_time)
            self.energies.append(e)
            self.magnetisations.append(m)
            self.magnetisations_abs.append(abs(m))
            self.specific_heats.append(h)
            self.susceptibilities.append(s)
            ising.plot_physics()

    # Compute ensemble phase transition
    def phase_transition_ensemble(self, time, average_time):
        for temperature in tqdm(self.temperatures):
            energy = []
            magnetisation = []
            specific_heat = []
            susceptibility = []
            for _ in tqdm(range(10)):
                ising = IsingNetwork(self.graph, temperature)
                ising.initialise_spins_hot()
                ising.evolution(time)
                e, m, h, s = ising.get_physics(average_time)
                energy.append(e)
                magnetisation.append(m)
                specific_heat.append(h)
                susceptibility.append(s)
            magnetisation = [abs(m) for m in magnetisation]
            e = np.mean(energy)
            m = np.mean(magnetisation)
            h = np.mean(specific_heat)
            s = np.mean(susceptibility)
            self.energies.append(e)
            self.magnetisations.append(m)
            self.magnetisations_abs.append(abs(m))
            self.specific_heats.append(h)
            self.susceptibilities.append(s)
            ising.plot_physics()

    def phase_transition_ensemble_random(self, time, average_time):
        for temperature in tqdm(self.temperatures):
            energy = []
            magnetisation = []
            specific_heat = []
            susceptibility = []
            for _ in tqdm(range(10)):
                ising = IsingNetwork(self.graph, temperature)
                ising.initialise_spins_hot()
                ising.evolution_random(time)
                e, m, h, s = ising.get_physics_random(average_time)
                energy.append(e)
                magnetisation.append(m)
                specific_heat.append(h)
                susceptibility.append(s)
            magnetisation = [abs(m) for m in magnetisation]
            e = np.mean(energy)
            m = abs(np.mean(magnetisation))
            h = np.mean(specific_heat)
            s = np.mean(susceptibility)
            self.energies.append(e)
            self.magnetisations.append(m)
            self.magnetisations_abs.append(abs(m))
            self.specific_heats.append(h)
            self.susceptibilities.append(s)
            ising.plot_physics()

    # Get physical quantities
    def plot_physics(self):
        self.name = "side" + str(int(np.sqrt(self.graph.number_of_nodes()))) + "_steps" + str(len(self.temperatures))
        path = './phase_transition'
        if not os.path.exists(path):
            os.mkdir(path)
        
        fig, axs = plt.subplots(2, 2, figsize=(10, 5))
        axs[0, 0].scatter(self.temperatures, self.energies)
        axs[0, 0].set_title("Energies")
        def m(x):
            sinh_term = np.sinh(2 / x)
            magnetisation = (1 - sinh_term ** (-4)) ** (1 / 8)
            if x < 2.26918531421302:
                return magnetisation
            else:
                return 0
        x = np.linspace(min(self.temperatures), max(self.temperatures), 100)
        y = list(map(m, x))
        axs[0, 1].plot(x, y, color='red')
        axs[0, 1].scatter(self.temperatures, self.magnetisations_abs)
        axs[0, 1].set_title("Magnetisations")
        axs[1, 0].scatter(self.temperatures, self.specific_heats)
        axs[1, 0].set_title("Specific heats")
        axs[1, 1].scatter(self.temperatures, self.susceptibilities)
        axs[1, 1].set_title("Susceptibilities")

        fig.tight_layout(pad=2.0)
        plt.savefig(path + f'/phase_transition_{self.name}.pdf')
        plt.close()

        print(self.temperatures[self.specific_heats.index(max(self.specific_heats))])
        print(self.temperatures[self.susceptibilities.index(max(self.susceptibilities))])
