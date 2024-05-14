import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy as sci
import numpy as np
import sympy as sp
import pickle
import os

class ExactSolutionIsing:
    def __init__(self, kind_of_graph):
        # Implement different graphs
        self.kind_of_graph = kind_of_graph

        # Implement temperatures
        self.temperatures = [0.05 + 0.05 * i for i in range(200)]

        # Implement paths to save/load data
        self.path_data = './data'
        if not os.path.exists(self.path_data):
            os.mkdir(self.path_data)
        self.path_data = './data/exact_solutions'
        if not os.path.exists(self.path_data):
            os.mkdir(self.path_data)
        self.path_data = './data/exact_solutions/' + self.kind_of_graph
        if not os.path.exists(self.path_data):
            os.mkdir(self.path_data)

        # Implement paths to save/load plots
        self.path_plot = './plot'
        if not os.path.exists(self.path_plot):
            os.mkdir(self.path_plot)
        self.path_plot = './plot/exact_solutions'
        if not os.path.exists(self.path_plot):
            os.mkdir(self.path_plot)
        self.path_plot = './plot/exact_solutions/' + self.kind_of_graph
        if not os.path.exists(self.path_plot):
            os.mkdir(self.path_plot)

    # Calculate 1d physical quantities
    def free_energy_1d(self, T):
        free_energy = - T * np.log(2 * np.cosh(1 / T))
        return free_energy

    def internal_energy_1d(self, T):
        internal_energy = - np.tanh(1 / T)
        return internal_energy
    
    def magnetisation_1d(self, T):
        return 0
 
    # Calculate 2d square physical quantities with symbolic derivation and numerical integration
    def symbolic_derivative_square(self):
        x = sp.Symbol("x")
        y = sp.Symbol("y")
        z = sp.Symbol("z")
        T = sp.Symbol("T")

        symbolic_P_square = sp.cosh(2 / T)**2 - sp.sinh(2 / T) * (sp.cos(x) + sp.cos(y))
        symbolic_P_square = sp.simplify(symbolic_P_square)

        symbolic_dP_square = symbolic_P_square.diff(T)
        symbolic_dP_square = sp.simplify(symbolic_dP_square)

        self.P_square = sp.lambdify((x, y, T), symbolic_P_square, 'numpy')
        self.dP_square = sp.lambdify((x, y, T), symbolic_dP_square, 'numpy')

    def free_energy_square(self, T):
        integrand = lambda x, y, T: np.log(self.P_square(x, y, T)) / (2 * np.pi)**2
        integral = sci.integrate.dblquad(integrand, 0, 2 * np.pi, 0, 2 * np.pi, args=(T,))[0]
        free_energy = - T * np.log(2) - T * integral / 2
        return free_energy

    def internal_energy_square(self, T):
        integrand = lambda x, y, T: self.dP_square(x, y, T) / (self.P_square(x, y, T) * (2 * np.pi)**2)
        integral = sci.integrate.dblquad(integrand, 0, 2 * np.pi, 0, 2 * np.pi, args=(T,))[0]
        internal_energy = T**2 * integral / 2
        return internal_energy
    
    def magnetisation_square(self, T):
        magnetisation = (1 - np.sinh(2 / T) ** (-4)) ** (1 / 8)
        if T < 2 / np.log(1 + np.sqrt(2)):
            return magnetisation
        else:
            return 0

    # Calculate 2d triangular physical quantities with symbolic derivation and numerical integration
    def symbolic_derivative_triangular(self):
        x = sp.Symbol("x")
        y = sp.Symbol("y")
        z = sp.Symbol("z")
        T = sp.Symbol("T")

        symbolic_P_triangular = sp.cosh(2 / T)**3 + sp.sinh(2 / T)**3 - sp.sinh(2 / T) * (sp.cos(x) + sp.cos(y) + sp.cos(z))
        symbolic_P_triangular = sp.simplify(symbolic_P_triangular)

        symbolic_dP_triangular = symbolic_P_triangular.diff(T)
        symbolic_dP_triangular = sp.simplify(symbolic_dP_triangular)

        self.P_triangular  = sp.lambdify((x, y, z, T), symbolic_P_triangular , 'numpy')
        self.dP_triangular  = sp.lambdify((x, y, z, T), symbolic_dP_triangular , 'numpy')

    def free_energy_triangular(self, T):
        integrand = lambda x, y, z, T: np.log(self.P_triangular(x, y, z, T)) / (2 * np.pi)**3
        integral = sci.integrate.tplquad(integrand, 0, 2 * np.pi, 0, 2 * np.pi, 0, 2 * np.pi, args=(T,))[0]
        free_energy = - T * np.log(2) - T * integral / 2
        return free_energy

    def internal_energy_triangular(self, T):
        integrand = lambda x, y, z, T: self.dP_triangular(x, y, z, T) / (self.P_triangular(x, y, z, T) * (2 * np.pi)**3)
        integral = sci.integrate.tplquad(integrand, 0, 2 * np.pi, 0, 2 * np.pi, 0, 2 * np.pi, args=(T,))[0]
        internal_energy = T**2 * integral / 2
        return internal_energy

    def magnetisation_triangular(self, T):
        magnetisation = ( (1 + np.exp(4 / T))**3 * (3 - np.exp(4/T)) / (1 - np.exp(4 / T))**3 / (3 + np.exp(4/T)))**(1 /8)
        if T < 2 / np.log(np.sqrt(3)):
            return magnetisation
        else:
            return 0

    # Calculate 2d hexagonal physical quantities with symbolic derivation and numerical integration
    def symbolic_derivative_hexagonal(self):
        x = sp.Symbol("x")
        y = sp.Symbol("y")
        z = sp.Symbol("z")
        T = sp.Symbol("T")

        symbolic_P_hexagonal = 0.5 * (1 + sp.cosh(2 / T)**3 - sp.sinh(2 / T)**2 * (sp.cos(x) + sp.cos(y) + sp.cos(z)))
        symbolic_P_hexagonal = sp.simplify(symbolic_P_hexagonal)

        symbolic_dP_hexagonal  = symbolic_P_hexagonal.diff(T)
        symbolic_dP_hexagonal = sp.simplify(symbolic_dP_hexagonal)

        self.P_hexagonal = sp.lambdify((x, y, z, T), symbolic_P_hexagonal  , 'numpy')
        self.dP_hexagonal = sp.lambdify((x, y, z, T), symbolic_dP_hexagonal , 'numpy')

    def free_energy_hexagonal(self, T):
        integrand = lambda x, y, z, T: np.log(self.P_hexagonal(x, y, z, T)) / (2 * np.pi)**3
        integral = sci.integrate.tplquad(integrand, 0, 2 * np.pi, 0, 2 * np.pi, 0, 2 * np.pi, args=(T,))[0]
        free_energy = - 2 * T * np.log(2) - T * integral / 2
        return free_energy

    def internal_energy_hexagonal(self, T):
        integrand = lambda x, y, z, T: self.dP_hexagonal(x, y, z, T) / (self.P_hexagonal(x, y, z, T) * (2 * np.pi)**3)
        integral = sci.integrate.tplquad(integrand, 0, 2 * np.pi, 0, 2 * np.pi, 0, 2 * np.pi, args=(T,))[0]
        internal_energy = T**2 * integral / 2
        return internal_energy

    def magnetisation_hexagonal(self, T):
        magnetisation = ( (1 + np.exp(4/T))**3 * (1 + np.exp(4 / T) - 4 * np.exp(2 / T)) / (1 - np.exp(2 / T))**6 / (1 + np.exp(2 / T))**2 )**(1/8)
        if T < 2 / np.log(2 + np.sqrt(3)):
            return magnetisation
        else:
            return 0

    # Calculate mean field physical quantities with numerical root finding
    def free_energy_mean_field(self, T):
        d = 4
        z = 2 * d 
        m = self.magnetisation_mean_field(T)
        free_energy = m**2 * z / 2 - T * np.log(2 * np.cosh(z * m / T))
        return free_energy

    def internal_energy_mean_field(self, T):
        d = 4
        z = 2 * d 
        m = self.magnetisation_mean_field(T)
        internal_energy = - z * m**2 / 2
        return internal_energy
    
    def m(self, x):
        d = 4
        z = 2 * d 
        return np.tanh (z * x / self.T) - x
    
    def magnetisation_mean_field(self, T):
        d = 4
        z = 2 * d 
        T_c = z
        if T < T_c:
            self.T = T
            solution = sci.optimize.root_scalar(self.m, bracket=[0.01, 1], method='brentq')
            return solution.root
        else:
            return 0
        
    # Calculate specific heat with five point formula
    def specific_heat(self, temperatures, internal_energies):
        indices = [i + 2 for i in range(len(temperatures) - 4)]
        specific_heats = []
        for i in indices:
            derivative = (internal_energies[i - 2] - 8 * internal_energies[i - 1] + 8 * internal_energies[i + 1] - internal_energies[i + 2]) / 12 / (temperatures[i] - temperatures[i - 1])
            specific_heats.append(derivative)
        return specific_heats
    
    # Calculate entropy
    def entropy(self, temperatures, internal_energies, free_energies):
        index = [int(i) for i in range(len(temperatures))]
        entropy = []
        for i in index:
            s = (internal_energies[i] - free_energies[i]) / temperatures[i]
            entropy.append(s)
        return entropy

    # Calculate physical quantities
    def compute_physics(self):
        if self.kind_of_graph == "1d":
            self.compute_physics_1d()
        elif self.kind_of_graph == "2d_square":
            self.compute_physics_square()
        elif self.kind_of_graph == "2d_triangular":
            self.compute_physics_triangular()
        elif self.kind_of_graph == "2d_hexagonal":
            self.compute_physics_hexagonal()
        elif self.kind_of_graph == "4d":
            self.compute_physics_mean_field()

    def compute_physics_1d(self):
        self.free_energy = []
        for i in tqdm(range(len(self.temperatures))):
            self.free_energy.append(self.free_energy_1d(self.temperatures[i]))

        self.internal_energy = []
        for i in tqdm(range(len(self.temperatures))):
            self.internal_energy.append(self.internal_energy_1d(self.temperatures[i]))

        self.magnetisation = []
        for i in tqdm(range(len(self.temperatures))):
            self.magnetisation.append(self.magnetisation_1d(self.temperatures[i]))

        self.specific_heat = self.specific_heat(self.temperatures, self.internal_energy)

        self.entropy = self.entropy(self.temperatures, self.internal_energy, self.free_energy)

    def compute_physics_square(self):
        self.symbolic_derivative_square()

        self.free_energy = []
        for i in tqdm(range(len(self.temperatures))):
            self.free_energy.append(self.free_energy_square(self.temperatures[i]))

        self.internal_energy = []
        for i in tqdm(range(len(self.temperatures))):
            self.internal_energy.append(self.internal_energy_square(self.temperatures[i]))

        self.magnetisation = []
        for i in tqdm(range(len(self.temperatures))):
            self.magnetisation.append(self.magnetisation_square(self.temperatures[i]))

        self.specific_heat = self.specific_heat(self.temperatures, self.internal_energy)

        self.entropy = self.entropy(self.temperatures, self.internal_energy, self.free_energy)

    def compute_physics_triangular(self):
        self.symbolic_derivative_triangular()

        self.free_energy = []
        for i in tqdm(range(len(self.temperatures))):
            self.free_energy.append(self.free_energy_triangular(self.temperatures[i]))

        self.internal_energy = []
        for i in tqdm(range(len(self.temperatures))):
            self.internal_energy.append(self.internal_energy_triangular(self.temperatures[i]))

        self.magnetisation = []
        for i in tqdm(range(len(self.temperatures))):
            self.magnetisation.append(self.magnetisation_triangular(self.temperatures[i]))

        self.specific_heat = self.specific_heat(self.temperatures, self.internal_energy)

        self.entropy = self.entropy(self.temperatures, self.internal_energy, self.free_energy)

    def compute_physics_hexagonal(self):
        self.symbolic_derivative_hexagonal()

        self.free_energy = []
        for i in tqdm(range(len(self.temperatures))):
            self.free_energy.append(self.free_energy_hexagonal(self.temperatures[i]))

        self.internal_energy = []
        for i in tqdm(range(len(self.temperatures))):
            self.internal_energy.append(self.internal_energy_hexagonal(self.temperatures[i]))
            
        self.magnetisation = []
        for i in tqdm(range(len(self.temperatures))):
            self.magnetisation.append(self.magnetisation_hexagonal(self.temperatures[i]))

        self.specific_heat = self.specific_heat(self.temperatures, self.internal_energy)

        self.entropy = self.entropy(self.temperatures, self.internal_energy, self.free_energy)

    def compute_physics_mean_field(self):
        self.free_energy = []
        for i in tqdm(range(len(self.temperatures))):
            self.free_energy.append(self.free_energy_mean_field(self.temperatures[i]))

        self.internal_energy = []
        for i in tqdm(range(len(self.temperatures))):
            self.internal_energy.append(self.internal_energy_mean_field(self.temperatures[i]))

        self.magnetisation = []
        for i in tqdm(range(len(self.temperatures))):
            self.magnetisation.append(self.magnetisation_mean_field(self.temperatures[i]))

        self.specific_heat = self.specific_heat(self.temperatures, self.internal_energy)

        self.entropy = self.entropy(self.temperatures, self.internal_energy, self.free_energy)

    # Save data with pickle
    def save_data(self):
        temperatures_path = self.path_data + '/temperatures.pickle'
        with open(temperatures_path, 'wb') as f:
            pickle.dump(self.temperatures, f)

        free_energy_path = self.path_data + '/free_energy.pickle'
        with open(free_energy_path, 'wb') as f:
            pickle.dump(self.free_energy, f)

        internal_energy_path = self.path_data + '/energy.pickle'
        with open(internal_energy_path, 'wb') as f:
            pickle.dump(self.internal_energy, f)

        magnetisation_path = self.path_data + '/magnetisation.pickle'
        with open(magnetisation_path, 'wb') as f:
            pickle.dump(self.magnetisation, f)

        specific_heat_path = self.path_data + '/specific_heat.pickle'
        with open(specific_heat_path, 'wb') as f:
            pickle.dump(self.specific_heat, f)

        entropy_path = self.path_data + '/entropy.pickle'
        with open(entropy_path, 'wb') as f:
            pickle.dump(self.entropy, f)

    # Load data with pickle
    def load_data(self):
        temperatures_path = self.path_data + '/temperatures.pickle'
        with open(temperatures_path, 'rb') as f:
            temperatures = pickle.load(f)

        free_energy_path = self.path_data + '/free_energy.pickle'
        with open(free_energy_path, 'rb') as f:
            free_energy = pickle.load(f)

        internal_energy_path = self.path_data + '/energy.pickle'
        with open(internal_energy_path, 'rb') as f:
            internal_energy = pickle.load(f)

        magnetisation_path = self.path_data + '/magnetisation.pickle'
        with open(magnetisation_path, 'rb') as f:
            magnetisation = pickle.load(f)

        specific_heat_path = self.path_data + '/specific_heat.pickle'
        with open(specific_heat_path, 'rb') as f:
            specific_heat = pickle.load(f)

        entropy_path = self.path_data + '/entropy.pickle'
        with open(entropy_path, 'rb') as f:
            entropy = pickle.load(f)

        return temperatures, free_energy, internal_energy, magnetisation, specific_heat, entropy

    # Plot data
    def plot_data(self):
        T, F, E, M, C, S = self.load_data()

        plt.title("Theoretical prediction in a " + self.kind_of_graph + " graph")
        plt.plot(T, F)
        plt.ylabel("Free energy")
        plt.xlabel("Temperature")
        plt.savefig(self.path_plot + '/free_energy.pdf')
        plt.close()

        plt.title("Theoretical prediction in a " + self.kind_of_graph + " graph")
        plt.plot(T, E)
        plt.ylabel("Internal energy")
        plt.xlabel("Temperature")
        plt.savefig(self.path_plot + '/energy.pdf')
        plt.close()

        plt.title("Theoretical prediction in a " + self.kind_of_graph + " graph")
        plt.plot(T, M)
        plt.ylabel("Magnetisation")
        plt.xlabel("Temperature")
        plt.savefig(self.path_plot + '/magnetisation.pdf')
        plt.close()

        plt.title("Theoretical prediction in a " + self.kind_of_graph + " graph")
        plt.plot(T[2:-2], C)
        plt.ylabel("Specific heat")
        plt.xlabel("Temperature")
        plt.savefig(self.path_plot + '/specific_heat.pdf')
        plt.close()

        plt.title("Theoretical prediction in a " + self.kind_of_graph + " graph")
        plt.plot(T, S)
        plt.ylabel("Entropy")
        plt.xlabel("Temperature")
        plt.savefig(self.path_plot + '/entropy.pdf')
        plt.close()

def run():
    ising = ExactSolutionIsing("2d_triangular")
    ising.compute_physics()
    ising.save_data()
    ising.plot_data()

run()
