import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import matplotlib.patches as pat
from tqdm import tqdm

class IsingNetwork:
    def __init__(self, graph, temperature, j = 1, b = 0):
        # Implement features of the graph
        self.graph = graph
        self.number_of_nodes = graph.number_of_nodes()
        self.nodes = graph.nodes()
        self._calculate_neighbours()
        self._calculate_spins()
        # Implement physical quantities
        self.temperature = temperature
        self.j = j
        self.b = b
        self._calculate_magnetisation()
        self._calculate_energy()
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

    def _calculate_spins(self):
        self.spins = np.random.choice([-1, 1], self.number_of_nodes)

    # Compute physical quantities
    def _calculate_magnetisation(self):
        self.magnetisation = abs(sum(self.spins))

    def _calculate_energy(self):
        neighbour_spins = 0
        for i in range(self.number_of_nodes):
            for j in self.neighbours[i]:
                neighbour_spins += self.spins[i] * self.spins[j]
        self.energy = - self.j * neighbour_spins / 2 + self.b * sum(self.spins)

    # Monte Carlo algorithm for reaching equilibrium
    def reach_equilibrium(self, time_evolution):
        self.field.append(self._compute_field())
        for _ in range(time_evolution):
            for i in range(self.number_of_nodes):
                neighbour_spins = 0
                for j in self.neighbours[i]:
                    neighbour_spins += self.spins[i] * self.spins[j]
                delta_energy = 2.0 * self.j * neighbour_spins
                transition_probability = np.exp(- delta_energy / self.temperature)
                if delta_energy < 0 or transition_probability > np.random.rand():
                    self.spins[i] *= -1
                    self.energy += delta_energy
            self.energies.append(self.energy)
            self._calculate_magnetisation()
            self.magnetisations.append(self.magnetisation)
            self.field.append(self._compute_field())

    # Get physical quantities
    def plot_physics(self, name_file):
        name = str(name_file)[:3]
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].plot(self.energies)
        axs[0].set_title("Energies")
        axs[1].plot(self.magnetisations)
        axs[1].set_title("Magnetisations")
        fig.tight_layout(pad=2.0)
        plt.savefig(f'equilibrium/equilibrium_{name}.pdf')

    def get_physics(self):
        magnetisation_data = self.magnetisations[-50:]
        energy_data = self.energies[-50:]
        magnetisation = np.mean(magnetisation_data)
        energy = np.mean(energy_data)
        magnetisation_square = np.mean([magnetisation * magnetisation for magnetisation in magnetisation_data])
        energy_square = np.mean([energy * energy for energy in energy_data])
        specific_heat = (energy_square - energy * energy) / self.temperature / self.temperature
        susceptibility = (magnetisation_square - magnetisation * magnetisation) / self.temperature
        return energy, magnetisation, specific_heat, susceptibility
    
    # Graphical representations of the network (only for square networks)
    def _compute_field(self):
        size = int(np.sqrt(self.number_of_nodes))
        field = np.zeros((size, size))
        for n in range(self.number_of_nodes):
            field[n // size][n % size] = self.spins[n]
        return field

    def plot_network(self, index):
        plt.imshow(self.field[index], cmap='binary', interpolation='nearest')
        plt.axis("off")
        plt.title(f'time: {index * 10}')
        legend_handles = [
            pat.Patch(color='white', label='- 1'),
            pat.Patch(color='black', label='+ 1')
        ]
        plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1))
        plt.show()

    def plot_network_animation(self, name_file):
      name = str(name_file)[:3]
      fig, ax = plt.subplots()
      def animate(i):
          ax.imshow(self.field[i], cmap='binary', interpolation='nearest')
          ax.axis("off")
          ax.set_title(f'temperature: {str(self.temperature)[:3]}, time: {i}')
          legend_handles = [
              pat.Patch(color='white', label='- 1'),
              pat.Patch(color='black', label='+ 1')
          ]
          ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1))
      animation = ani.FuncAnimation(fig, animate, frames=len(self.field), interval=100, repeat=False)
      animation.save(f'sequence/temperature_{name}.gif')

class IsingEnsemble:
    def __init__(self, number_of_iterations, initial_temperature, final_temperature, number_of_steps, graph):
        self.graph = graph
        # Implement temperatures to scroll
        self.temperatures = np.arange(initial_temperature, final_temperature, (final_temperature - initial_temperature) / number_of_steps)
        self.number_of_iterations = number_of_iterations
        # Implement physical quantities
        self.energies = []
        self.magnetisations = []
        self.specific_heats = []
        self.susceptibilities = []
        self.uncertainty_critical_temperature = (final_temperature - initial_temperature) / number_of_steps

    # Compute phase transition
    def phase_transitions(self):
        for temperature in tqdm(self.temperatures):
            ising = IsingNetwork(self.graph, temperature)
            ising.reach_equilibrium(self.number_of_iterations)
            e, m, h, s = ising.get_physics()
            self.energies.append(e)
            self.magnetisations.append(m)
            self.specific_heats.append(h)
            self.susceptibilities.append(s)

    # Get physical quantities
    def plot_physics(self, name_file):
        fig, axs = plt.subplots(2, 2, figsize=(10, 5))
        axs[0, 0].plot(self.temperatures, self.energies),
        axs[0, 0].scatter(self.temperatures, self.energies),
        axs[0, 0].set_title("Energies")
        axs[0, 1].plot(self.temperatures, self.magnetisations),
        axs[0, 1].scatter(self.temperatures, self.magnetisations),
        axs[0, 1].set_title("Magnetisations")
        axs[1, 0].plot(self.temperatures, self.specific_heats),
        axs[1, 0].scatter(self.temperatures, self.specific_heats),
        axs[1, 0].set_title("Specific heats")
        axs[1, 1].plot(self.temperatures, self.susceptibilities),
        axs[1, 1].scatter(self.temperatures, self.susceptibilities),
        axs[1, 1].set_title("Susceptibilities")
        fig.tight_layout(pad=2.0)
        plt.savefig(f'phase_transition/plot_{name_file}.pdf')

    def get_critical_temperature(self):
        index_max_susceptibility = self.susceptibilities.index(max(self.susceptibilities))
        critical_temperature_susceptibility = self.temperatures[index_max_susceptibility]

        index_max_specific_heat = self.specific_heats.index(max(self.specific_heats))
        critical_temperature_specific_heat = self.temperatures[index_max_specific_heat]

        print("Critical temperature for susceptibilities:", critical_temperature_susceptibility, "+/-", self.uncertainty_critical_temperature)
        print("Critical temperature for specific heats:", critical_temperature_specific_heat, "+/-", self.uncertainty_critical_temperature)
