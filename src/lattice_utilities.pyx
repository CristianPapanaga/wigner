# cython: language_level = 3
# cython: boundscheck = False
# cython: wraparound = False

import numpy as np
cimport numpy as np

cpdef np.ndarray[np.int64_t, ndim=1, mode="c"] generate_filled_square_lattice(int L, double filling = 1):

    '''
    Generates a square lattice of dimension L with a specified filling of particles,
    given as a float between 0 (exclusive) and 1 (inclusive). The lattice is populated by [spin, 0], where spin
    indicates a particle (typically, an electron) with given spin is present and 0 indicates an empty lattice site.

    Note that the filling constant must be chosen such that L*filling is an integer.
    '''

    cdef int N = L**2
    cdef np.ndarray[np.int64_t, ndim = 1, mode = "c"] particle_lattice

    # Fill lattice with particles.
    particle_lattice = np.zeros(N, dtype = np.int64) # All sites filled by 0 (site empty).
    particle_lattice[:int(N*filling)] = 1 # Fill the first N*filling sites of the lattice with 1 (site filled).
    np.random.shuffle(particle_lattice) # Shuffle the particles randomly throughout the lattice.

    return particle_lattice

cpdef np.ndarray[np.int64_t, ndim=2, mode="c"] get_nearest_neighbour_coords(int L, int n):

    '''
    Given the lattice length L and the number of lattice sites N, finds the coordinates of the nth
    nearest neighbours (array i_nn) to every site in the lattice, with periodic boundary conditions
    which work with single-coordinate-indexed lattices. 

    The arguement n specifies that the nth nearest neighbours will be included (max 3rd neighbours).

    Returns lattice_nn, a list of lists where, for eg., the list at index 0 contains the neighbours of site 0.
    '''

    cdef list ij_nn, nn, nn_0, nn_1, nn_2, nn_3
    cdef np.ndarray[np.int64_t, ndim = 2, mode = "c"] lattice_nn
    cdef np.ndarray[np.int64_t, ndim = 1, mode = "c"] i_nn
    cdef int index, k 

    # Changed from dtype=int to explicit np.int64 to satisfy Windows
    lattice_nn = np.empty((L**2, n*4), dtype = np.int64)

    # Get 2D neighbours.
    k = 0
    for i in range(L):
        for j in range(L):
            ij_nn = []

            # 1st nearest neighbours (east, west, north, south).
            if n >= 1:
                nn_0 = [(i + 1) % L, j]
                nn_1 = [(i - 1) % L, j]
                nn_2 = [i, (j + 1) % L]
                nn_3 = [i, (j - 1) % L]
                ij_nn.extend((nn_0, nn_1, nn_2, nn_3))
            
            # 2nd nearest neighbours (north-east, south-west, south-east, north-west).
            if n >= 2:
                nn_0 = [(i + 1) % L, (j + 1) % L]
                nn_1 = [(i - 1) % L, (j - 1) % L]
                nn_2 = [(i + 1) % L, (j - 1) % L]
                nn_3 = [(i - 1) % L, (j + 1) % L]
                ij_nn.extend((nn_0, nn_1, nn_2, nn_3))
            
            # 3rd nearest neighbours (east, west, north, south).
            if n >= 3:
                nn_0 = [(i + 2) % L, j]
                nn_1 = [(i - 2) % L, j]
                nn_2 = [i, (j + 2) % L]
                nn_3 = [i, (j - 2) % L]
                ij_nn.extend((nn_0, nn_1, nn_2, nn_3))

            # Convert to 1D representation and return.
            i_nn = np.array([], dtype = np.int64)
            for nn in ij_nn:
                index = (nn[0] * L) + nn[1]
                i_nn = np.append(i_nn, index)

            lattice_nn[k] = i_nn.copy()
            k += 1

    return lattice_nn