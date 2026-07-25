import os
from math import sqrt

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy import fft

plt.rcParams["font.family"] = "serif"
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8

greys_without_white = cmr.get_sub_cmap("Greys", 0.25, 1)

def save_data(data: tuple, data_names: list[str]):

    while True:

        data_folder = input("Provide a name for the folder in which your simulation data will be stored (this can be an existing or new folder): ")
        cwd = os.getcwd()

        path = os.path.join(cwd, f"src\\data\\{data_folder}")

        # Try to create a folder with the user inputted name, {data_folder}.

        try:

            os.mkdir(path)

            print("Saving data; for large n_steps or n this may take a few minutes...")

            for dat, name in zip(data, data_names):

                np.save(f"{path}\\{data_folder} {name}", dat)

            print(f"Simulation data saved to {path}.")

            break

        # If a folder with the input name already exists in /data/, handle the FileExistsError by prompting the user to overwrite the files in that folder,
        # create a new folder, or abort the data saving process. 

        except FileExistsError:

            while True:

                overwrite_input = input(f"A folder called \"{data_folder}\" already exists; would you like to overwrite all data in \"{data_folder}\" (y/n)? ")

                if overwrite_input == "y":

                    print("Saving data; for large n_steps or n this may take a few minutes...")

                    for dat, name in zip(data, data_names):

                        np.save(f"{path}\\{data_folder} {name}", dat)

                    print(f"Simulation data saved to {path}.")

                    break

                elif overwrite_input == "n":
                        
                    create_folder_input = input("Would you like to create a new folder (otherwise, the data saving process will be aborted) (y/n)? ")

                    if create_folder_input == "y":

                        break

                    elif create_folder_input == "n":

                        print("Data saving aborted.")

                        break

                    else:

                        print(f"\"{create_folder_input}\" is not a valid response; please try again...")
                    
                else:

                    print(f"\"{overwrite_input}\" is not a valid response; please try again...")
                    
            break

        # Catch any other OSErrors that may come up.

        except OSError as error:

            print(error)
            print("Data saving aborted.")

def get_data(data_folder: str, data_names: list[str]) -> list[np.ndarray]:
    
    '''
    Given the name "data_folder" of a folder in src/data/, returns np.ndarrays of the data stored there.
    '''

    cwd = os.getcwd()
    data_path = f"src\\data\\{data_folder}\\{data_folder}"
    data = []

    for name in data_names:

        path = os.path.join(cwd, f"{data_path} {name}.npy")

        try: 

            data.append(np.load(path))

        except FileNotFoundError as error:

            print(error)
            print("FileNotFoundError: it looks like the file you requested does not exist - perhaps data_folder is incorrect?")
            print("Aborting data retrieval.")

    return data

def get_specific_heat(lattice: list, data: list[np.ndarray], schedule: list[float]):

    '''
    Given energy data, returns the specific heat of the system from which that data was collected.
    '''

    N = lattice.shape[0]

    c_list = []

    for dat, T in zip(data, schedule):

        c = (1/(N * T**2))*(np.mean(dat**2) - np.mean(dat)**2)
        c_list.append(c)

    return c_list

def get_reciprocal_lattice(lattice: list):

    '''
    Returns the reciprocal lattice of the given lattice, ie. gets the discrete Fourier transform of it.
    '''

    N = lattice.shape[0]
    L = int(sqrt(N))

    lattice = np.reshape(lattice, (L, L))

    reciprocal = fft.fft2(lattice, norm = "forward") 

    return np.round(np.abs(reciprocal.flatten()), 10)

def plot_lattice(lattice: list, type = "square"):

    L = int(sqrt(lattice.shape[0]))

    cwd = os.getcwd()
    path = os.path.join(cwd, "src\\visualizations")

    if type == "square":

        # Define primitive vectors.

        a_1 = np.array([1, 0])
        a_2 = np.array([0, 1])

    if type == "triangle":

        a_1 = np.array([1, 0])
        a_2 = np.array([0, 1])
    
    # Generate Bravais lattice coordinates in space.
    
    x = np.array([[i * a_1[0] + j * a_2[0] for i in np.arange(L)] for j in np.arange(L)])
    y = np.array([[i * a_1[1] + j * a_2[1] for i in np.arange(L)] for j in np.arange(L)])  

    # Scatter coordinates (and make it pretty). 

    fig, ax = plt.subplots(1)

    marker_sizes = lattice.copy()
    marker_sizes[marker_sizes == 1] = 20 # Make occupied sites bigger.
    marker_sizes[marker_sizes == 0] = 5

    ax.scatter(x.flatten(), y.flatten(), c = lattice, cmap = greys_without_white, s = marker_sizes)
    
    ax.set_aspect("equal")
    ax.set_xticklabels("")
    ax.set_yticklabels("")
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)

    fig.savefig(f"{path}\\Lattice.pdf")

""" def plot_reciprocal_lattice(lattice: list, type = "square"):

    Q = np.array([[0, -1], [1, 0]]) # 90 degree rotation matrix, for use in calculating reciprocal vectors.

    N = lattice.shape[0]
    L = int(sqrt(N))

    reciprocal = get_reciprocal_lattice(lattice)

    cwd = os.getcwd()
    path = os.path.join(cwd, "visualizations")

    if type == "square":

        # Define primitive real space vectors.

        a_1 = np.array([1, 0])
        a_2 = np.array([0, 1])

        # Define primitive reciprocal space vectors.

        b_1 = (2 * np.pi * (Q @ a_2)) / (np.dot(a_1, Q @ a_2))
        b_2 = (2 * np.pi * (Q @ a_1)) / (np.dot(a_2, Q @ a_1))    

    # Generate Bravais lattice coordinates in space.
    
    x = np.array([[i * b_1[0] + j * b_2[0] for i in np.arange(-L/2, L/2)] for j in np.arange(-L/2, L/2)])/L
    y = np.array([[i * b_1[1] + j * b_2[1] for i in np.arange(-L/2, L/2)] for j in np.arange(-L/2, L/2)])/L

    # Scatter coordinates (and make it pretty). 

    fig, ax = plt.subplots(1)

    marker_sizes = reciprocal.copy()
    marker_sizes[marker_sizes > 0] = 20 # Make occupied sites bigger.
    marker_sizes[marker_sizes <= 0] = 5

    ax.scatter(x.flatten(), y.flatten(), c = reciprocal, cmap = greys_without_white, s = marker_sizes)
    
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(plt.MultipleLocator(4 * np.pi/L))
    ax.yaxis.set_major_locator(plt.MultipleLocator(4 * np.pi/L))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter(L, np.pi)))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter(L, np.pi)))

    fig.savefig(f"{path}\\Reciprocal lattice.pdf") """

def plot_lattice_with_spin_vectors(particle_lattice: list, spin_lattice: list, type = "square"):

    N = particle_lattice.shape[0]
    L = int(sqrt(N)) 

    cwd = os.getcwd()
    path = os.path.join(cwd, "src\\visualizations")

    if type == "square":

        # Define primitive vectors.

        a_1 = np.array([1, 0])
        a_2 = np.array([0, 1])

    if type == "triangle":

        a_1 = np.array([1, 0])
        a_2 = np.array([0, 1])
    
    # Generate Bravais lattice coordinates in space.
    
    x = np.array([[i * a_1[0] + j * a_2[0] for i in np.arange(L)] for j in np.arange(L)]).flatten()
    y = np.array([[i * a_1[1] + j * a_2[1] for i in np.arange(L)] for j in np.arange(L)]).flatten()

    # Scatter coordinates (and make it pretty). 

    fig, ax = plt.subplots(1)

    marker_sizes = particle_lattice.copy()
    marker_sizes[marker_sizes == 1] = 20 # Make occupied sites bigger.
    marker_sizes[marker_sizes == 0] = 5
    
    x_spins = spin_lattice[:, 0]
    y_spins = spin_lattice[:, 1]
    norm_consts = np.linalg.norm([x_spins, y_spins], axis = 0)
    
    x_spins = np.divide(x_spins, norm_consts, out = np.zeros_like(x_spins), where = (norm_consts != 0))
    y_spins = np.divide(y_spins, norm_consts, out = np.zeros_like(x_spins), where = (norm_consts != 0))
    z_spins = spin_lattice[:, 2]

    c_norm = Normalize(vmin = -1, vmax = 1)

    ax.scatter(x, y, c = particle_lattice, cmap = greys_without_white, s = marker_sizes, zorder = 10)
    ax.quiver(x, y, x_spins, y_spins, color = cm.viridis(c_norm(z_spins)), angles = "xy", scale_units = "xy", scale = 1.8, zorder = -10)  
    fig.colorbar(cm.ScalarMappable(norm = c_norm, cmap = cm.viridis), ax = ax)
    
    ax.set_aspect("equal")
    ax.set_xticklabels("")
    ax.set_yticklabels("")
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)

    fig.savefig(f"{path}\\Lattice with spin vectors.pdf")

def plot_specific_heat_data(data: list, schedule: list[float]):

    '''
    Given specific heat data, plots it.
    '''
    
    cwd = os.getcwd()

    path = os.path.join(cwd, "src\\visualizations")

    fig, ax = plt.subplots(1)

    ax.plot(schedule, data, color = "black")
    ax.scatter(schedule, data, color = "black", zorder = 10)

    ax.grid(linestyle = "--")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Specific heat (per site)")

    fig.savefig(f"{path}\\Specific heat plot.pdf")

def plot_annealing_data(lattice: list, data: list[np.ndarray], data_names: list[str], schedule: list[float]):

    '''
    Given data (which can be generated using anneal() functions), as well as the cooling
    schedule of the simulated annealing, outputs visualizations of that data to the /visualizations/ folder.
    '''

    cwd = os.getcwd()

    path = os.path.join(cwd, "src\\visualizations")

    N = lattice.shape[0]
    L = int(sqrt(N))

    i = 0
    mean_d_list = []

    for dat in data:

        dat = np.asarray(dat)/N # Data per lattice site.

        for d in dat:

            mean_d = np.mean(d)
            mean_d_list.append(mean_d)

        fig, ax = plt.subplots(1)

        ax.scatter(schedule, mean_d_list, color = "black", zorder = 10)

        data_name = data_names[i]

        ax.grid(linestyle = "--")
        ax.set_xlabel("Temperature")
        ax.set_ylabel(data_name)

        fig.savefig(f"{path}\\{data_name} annealing plot.pdf")

        i += 1

    # Reshape the 1D indexed lattice to the 2D lattice for saving an image of the lattice state.

    plot_lattice(lattice)

def multiple_formatter(denominator, number = np.pi, latex = '\\pi'):

    '''
    Formatter for non-standard axis labels in Matplotlib, by Scott Centoni
    (see https://stackoverflow.com/questions/40642061/how-to-set-axis-ticks-in-multiples-of-pi-python-matplotlib).
    '''

    def gcd(a, b):
        while b:
            a, b = b, a%b
        return a
    def _multiple_formatter(x, pos):
        den = denominator
        num = int(np.rint(den*x/number))
        com = gcd(num,den)
        (num,den) = (int(num/com),int(den/com))
        if den==1:
            if num==0:
                return r'$0$'
            if num==1:
                return r'$%s$'%latex
            elif num==-1:
                return r'$-%s$'%latex
            else:
                return r'$%s%s$'%(num,latex)
        else:
            if num==1:
                return r'$\\frac{%s}{%s}$'%(latex,den)
            elif num==-1:
                return r'$\\frac{-%s}{%s}$'%(latex,den)
            else:
                return r'$\\frac{%s%s}{%s}$'%(num,latex,den)
    return _multiple_formatter