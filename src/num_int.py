import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy as sci
import numpy as np
import sympy as sp
import pickle
import os

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

class One:
    def __init__(self):
        self.kind_of_graph = "1d"

        self.path_data = './data'
        if not os.path.exists(self.path_data):
            os.mkdir(self.path_data)
        self.path_data = './data/' + self.kind_of_graph
        if not os.path.exists(self.path_data):
            os.mkdir(self.path_data)

    def free_energy_1d(self, T):
        free_energy = - T * np.log(2 * np.cosh(1 / T))
        return free_energy

    def internal_energy_1d(self, T):
        internal_energy = - np.tanh(1 / T)
        return internal_energy
    
    def magnetisation_1d(self, T):
        return 0

    # Save data with pickle
    def save_1d_data(self):
        temperatures_path = self.path_data + '/temperatures.pickle'
        with open(temperatures_path, 'wb') as f:
            pickle.dump(temperatures, f)

        free_energy = []
        for i in tqdm(range(len(temperatures))):
            free_energy.append(self.free_energy_1d(temperatures[i]))
        free_energy_path = self.path_data + '/free_energy.pickle'
        with open(free_energy_path, 'wb') as f:
            pickle.dump(free_energy, f)

        internal_energy = []
        for i in tqdm(range(len(temperatures))):
            internal_energy.append(self.internal_energy_1d(temperatures[i]))
        internal_energy_path = self.path_data + '/internal_energy.pickle'
        with open(internal_energy_path, 'wb') as f:
            pickle.dump(internal_energy, f)
        
        magnetisation = []
        for i in tqdm(range(len(temperatures))):
            magnetisation.append(self.magnetisation_1d(temperatures[i]))
        magnetisation_path = self.path_data + '/magnetisation.pickle'
        with open(magnetisation_path, 'wb') as f:
            pickle.dump(magnetisation, f)

    # Load data with pickle
    def load_1d_data(self):
        temperatures_path = self.path_data + '/temperatures.pickle'
        with open(temperatures_path, 'rb') as f:
            temperatures = pickle.load(f)
        free_energy_path = self.path_data + '/free_energy.pickle'
        with open(free_energy_path, 'rb') as f:
            free_energy = pickle.load(f)
        internal_energy_path = self.path_data + '/internal_energy.pickle'
        with open(internal_energy_path, 'rb') as f:
            internal_energy = pickle.load(f)
        magnetisation_path = self.path_data + '/magnetisation.pickle'
        with open(magnetisation_path, 'rb') as f:
            magnetisation = pickle.load(f)
        return temperatures, free_energy, internal_energy, magnetisation

    # Plot data
    def plot_1d_data(self):
        temperatures, free_energy, internal_energy, magnetisation = self.load_1d_data()
        fig, axs = plt.subplots(1, 3, figsize=(10, 5))
        axs[0].plot(temperatures, free_energy)
        axs[0].set_title("Free energy")
        axs[1].plot(temperatures, internal_energy)
        axs[1].set_title("Internal energy")
        axs[2].plot(temperatures, magnetisation)
        axs[2].set_title("Magnetisation")
        plt.show()

class Square:
    def __init__(self):
        self.kind_of_graph = "2d_square"

        self.path_data = './data'
        if not os.path.exists(self.path_data):
            os.mkdir(self.path_data)
        self.path_data = './data/' + self.kind_of_graph
        if not os.path.exists(self.path_data):
            os.mkdir(self.path_data)

    # Calculate physical quantities with numerical integration
    def symbolic_derivative(self):
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

    # Save data with pickle
    def save_square_data(self):
        temperatures_path = self.path_data + '/temperatures.pickle'
        with open(temperatures_path, 'wb') as f:
            pickle.dump(temperatures, f)

        free_energy = []
        for i in tqdm(range(len(temperatures))):
            free_energy.append(self.free_energy_square(temperatures[i]))
        free_energy_path = self.path_data + '/free_energy.pickle'
        with open(free_energy_path, 'wb') as f:
            pickle.dump(free_energy, f)

        internal_energy = []
        for i in tqdm(range(len(temperatures))):
            internal_energy.append(self.internal_energy_square(temperatures[i]))
        internal_energy_path = self.path_data + '/internal_energy.pickle'
        with open(internal_energy_path, 'wb') as f:
            pickle.dump(internal_energy, f)
        
        magnetisation = []
        for i in tqdm(range(len(temperatures))):
            magnetisation.append(self.magnetisation_square(temperatures[i]))
        magnetisation_path = self.path_data + '/magnetisation.pickle'
        with open(magnetisation_path, 'wb') as f:
            pickle.dump(magnetisation, f)

    # Load data with pickle
    def load_square_data(self):
        temperatures_path = self.path_data + '/temperatures.pickle'
        with open(temperatures_path, 'rb') as f:
            temperatures = pickle.load(f)
        free_energy_path = self.path_data + '/free_energy.pickle'
        with open(free_energy_path, 'rb') as f:
            free_energy = pickle.load(f)
        internal_energy_path = self.path_data + '/internal_energy.pickle'
        with open(internal_energy_path, 'rb') as f:
            internal_energy = pickle.load(f)
        magnetisation_path = self.path_data + '/magnetisation.pickle'
        with open(magnetisation_path, 'rb') as f:
            magnetisation = pickle.load(f)
        return temperatures, free_energy, internal_energy, magnetisation

    # Plot data
    def plot_square_data(self):
        temperatures, free_energy, internal_energy, magnetisation = self.load_square_data()
        fig, axs = plt.subplots(1, 3, figsize=(10, 5))
        axs[0].plot(temperatures, free_energy)
        axs[0].set_title("Free energy")
        axs[1].plot(temperatures, internal_energy)
        axs[1].set_title("Internal energy")
        axs[2].plot(temperatures, magnetisation)
        axs[2].set_title("Magnetisation")
        plt.show()

class Triangular:
    def __init__(self):
        self.kind_of_graph = "2d_triangular"

        self.path_data = './data'
        if not os.path.exists(self.path_data):
            os.mkdir(self.path_data)
        self.path_data = './data/' + self.kind_of_graph
        if not os.path.exists(self.path_data):
            os.mkdir(self.path_data)

    # Calculate physical quantities with numerical integration
    def symbolic_derivative(self):
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
        integrand = lambda x, y, z, T: np.log(self.P_triangular(x, y, z, T)) / (2 * np.pi)**2
        integral = sci.integrate.tplquad(integrand, 0, 2 * np.pi, 0, 2 * np.pi, 0, 2 * np.pi, args=(T,))[0]
        free_energy = - T * np.log(2) - T * integral / 2
        return free_energy

    def internal_energy_triangular(self, T):
        integrand = lambda x, y, z, T: self.dP_triangular(x, y, z, T) / (self.P_triangular(x, y, z, T) * (2 * np.pi)**2)
        integral = sci.integrate.tplquad(integrand, 0, 2 * np.pi, 0, 2 * np.pi, 0, 2 * np.pi, args=(T,))[0]
        internal_energy = T**2 * integral / 2
        return internal_energy

    def magnetisation_triangular(self, T):
        magnetisation = ( (1 + np.exp(4 / T))**3 * (3 - np.exp(4/T)) / (1 - np.exp(4 / T))**3 / (3 + np.exp(4/T)))**(1 /8)
        if T < 2 / np.log(np.sqrt(3)):
            return magnetisation
        else:
            return 0

    # Save data with pickle
    def save_triangular_data(self):
        temperatures_path = self.path_data + '/temperatures.pickle'
        with open(temperatures_path, 'wb') as f:
            pickle.dump(temperatures, f)

        free_energy = []
        for i in tqdm(range(len(temperatures))):
            free_energy.append(self.free_energy_triangular(temperatures[i]))
        free_energy_path = self.path_data + '/free_energy.pickle'
        with open(free_energy_path, 'wb') as f:
            pickle.dump(free_energy, f)

        internal_energy = []
        for i in tqdm(range(len(temperatures))):
            internal_energy.append(self.internal_energy_triangular(temperatures[i]))
        internal_energy_path = self.path_data + '/internal_energy.pickle'
        with open(internal_energy_path, 'wb') as f:
            pickle.dump(internal_energy, f)
        
        magnetisation = []
        for i in tqdm(range(len(temperatures))):
            magnetisation.append(self.magnetisation_triangular(temperatures[i]))
        magnetisation_path = self.path_data + '/magnetisation.pickle'
        with open(magnetisation_path, 'wb') as f:
            pickle.dump(magnetisation, f)

    # Load data with pickle
    def load_triangular_data(self):
        temperatures_path = self.path_data + '/temperatures.pickle'
        with open(temperatures_path, 'rb') as f:
            temperatures = pickle.load(f)
        free_energy_path = self.path_data + '/free_energy.pickle'
        with open(free_energy_path, 'rb') as f:
            free_energy = pickle.load(f)
        internal_energy_path = self.path_data + '/internal_energy.pickle'
        with open(internal_energy_path, 'rb') as f:
            internal_energy = pickle.load(f)
        magnetisation_path = self.path_data + '/magnetisation.pickle'
        with open(magnetisation_path, 'rb') as f:
            magnetisation = pickle.load(f)
        return temperatures, free_energy, internal_energy, magnetisation

    # Plot data
    def plot_triangular_data(self):
        temperatures, free_energy, internal_energy, magnetisation = self.load_triangular_data()
        fig, axs = plt.subplots(1, 3, figsize=(10, 5))
        axs[0].plot(temperatures, free_energy)
        axs[0].set_title("Free energy")
        axs[1].plot(temperatures, internal_energy)
        axs[1].set_title("Internal energy")
        axs[2].plot(temperatures, magnetisation)
        axs[2].set_title("Magnetisation")
        plt.show()

class Hexagonal:
    def __init__(self):
        self.kind_of_graph = "2d_hexagonal"

        self.path_data = './data'
        if not os.path.exists(self.path_data):
            os.mkdir(self.path_data)
        self.path_data = './data/' + self.kind_of_graph
        if not os.path.exists(self.path_data):
            os.mkdir(self.path_data)

    # Calculate physical quantities with numerical integration
    def symbolic_derivative(self):
        x = sp.Symbol("x")
        y = sp.Symbol("y")
        z = sp.Symbol("z")
        T = sp.Symbol("T")

        symbolic_P_hexagonal = 1 / 2 * (1 + sp.cosh(2 / T)**3 - sp.sinh(2 / T)**2 * (sp.cos(x) + sp.cos(y))+ + sp.cos(z))
        symbolic_P_hexagonal = sp.simplify(symbolic_P_hexagonal)
        symbolic_dP_hexagonal  = symbolic_P_hexagonal.diff(T)
        symbolic_dP_hexagonal = sp.simplify(symbolic_dP_hexagonal)

        self.P_hexagonal = sp.lambdify((x, y, z, T), symbolic_P_hexagonal  , 'numpy')
        self.dP_hexagonal = sp.lambdify((x, y, z, T), symbolic_dP_hexagonal , 'numpy')

    def free_energy_hexagonal(self, T):
        integrand = lambda x, y, z, T: np.log(self.P_hexagonal(x, y, z, T)) / (2 * np.pi)**2
        integral = sci.integrate.tplquad(integrand, 0, 2 * np.pi, 0, 2 * np.pi, 0, 2 * np.pi, args=(T,))[0]
        free_energy = - T * np.log(2) - T * integral / 2
        return free_energy

    def internal_energy_hexagonal(self, T):
        integrand = lambda x, y, z, T: self.dP_hexagonal(x, y, z, T) / (self.P_hexagonal(x, y, z, T) * (2 * np.pi)**2)
        integral = sci.integrate.tplquad(integrand, 0, 2 * np.pi, 0, 2 * np.pi, 0, 2 * np.pi, args=(T,))[0]
        internal_energy = T**2 * integral / 2
        return internal_energy

    def magnetisation_hexagonal(self, T):
        magnetisation = ( (1 + np.exp(4/T))**3 * (1 + np.exp(4 / T) - 4 * np.exp(2 / T)) / (1 - np.exp(2 / T))**6 / (1 + np.exp(2 / T))**2 )**(1/8)
        if T < 2 / np.log(2 + np.sqrt(3)):
            return magnetisation
        else:
            return 0

    # Save data with pickle
    def save_hexagonal_data(self):
        temperatures_path = self.path_data + '/temperatures.pickle'
        with open(temperatures_path, 'wb') as f:
            pickle.dump(temperatures, f)

        free_energy = []
        for i in tqdm(range(len(temperatures))):
            free_energy.append(self.free_energy_hexagonal(temperatures[i]))
        free_energy_path = self.path_data + '/free_energy.pickle'
        with open(free_energy_path, 'wb') as f:
            pickle.dump(free_energy, f)

        internal_energy = []
        for i in tqdm(range(len(temperatures))):
            internal_energy.append(self.internal_energy_hexagonal(temperatures[i]))
        internal_energy_path = self.path_data + '/internal_energy.pickle'
        with open(internal_energy_path, 'wb') as f:
            pickle.dump(internal_energy, f)
        
        magnetisation = []
        for i in tqdm(range(len(temperatures))):
            magnetisation.append(self.magnetisation_hexagonal(temperatures[i]))
        magnetisation_path = self.path_data + '/magnetisation.pickle'
        with open(magnetisation_path, 'wb') as f:
            pickle.dump(magnetisation, f)

    # Load data with pickle
    def load_hexagonal_data(self):
        temperatures_path = self.path_data + '/temperatures.pickle'
        with open(temperatures_path, 'rb') as f:
            temperatures = pickle.load(f)
        free_energy_path = self.path_data + '/free_energy.pickle'
        with open(free_energy_path, 'rb') as f:
            free_energy = pickle.load(f)
        internal_energy_path = self.path_data + '/internal_energy.pickle'
        with open(internal_energy_path, 'rb') as f:
            internal_energy = pickle.load(f)
        magnetisation_path = self.path_data + '/magnetisation.pickle'
        with open(magnetisation_path, 'rb') as f:
            magnetisation = pickle.load(f)
        return temperatures, free_energy, internal_energy, magnetisation

    # Plot data
    def plot_hexagonal_data(self):
        temperatures, free_energy, internal_energy, magnetisation = self.load_hexagonal_data()
        fig, axs = plt.subplots(1, 3, figsize=(10, 5))
        axs[0].plot(temperatures, free_energy)
        axs[0].set_title("Free energy")
        axs[1].plot(temperatures, internal_energy)
        axs[1].set_title("Internal energy")
        axs[2].plot(temperatures, magnetisation)
        axs[2].set_title("Magnetisation")
        plt.show()

##########

# Create data

def one():
    one_data = One()
    one_data.save_1d_data()
    one_data.plot_1d_data()
#one()

def square():
    square_data = Square()
    square_data.symbolic_derivative()
    square_data.save_square_data()
    square_data.plot_square_data()
#square()

def triangular():
    triangular_data = Triangular()
    triangular_data.symbolic_derivative()
    triangular_data.save_triangular_data()
    triangular_data.plot_triangular_data()
#triangular()

def hexagonal():
    hexagonal_data = Hexagonal()
    hexagonal_data.symbolic_derivative()
    hexagonal_data.save_hexagonal_data()
    hexagonal_data.plot_hexagonal_data()
#hexagonal()