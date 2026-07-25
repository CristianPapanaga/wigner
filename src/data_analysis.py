import data_utilities as dat_utils

data_folder = "data"

# Get data from the folder with name data_folder in src\data\.
particle_lattice, spin_lattice, E_data, n_sweeps, schedule = dat_utils.get_data(
    f"{data_folder}",
    ["particle_lattice", "spin_lattice", "E_array", "n_sweeps", "schedule"]
)

c_data = dat_utils.get_specific_heat(particle_lattice, E_data, schedule)

dat_utils.plot_annealing_data(particle_lattice, data = [E_data], data_names = ["Energy (per site)"], schedule = schedule)
dat_utils.plot_specific_heat_data(c_data, schedule)
dat_utils.plot_lattice(particle_lattice)
dat_utils.plot_lattice_with_spin_vectors(particle_lattice, spin_lattice)