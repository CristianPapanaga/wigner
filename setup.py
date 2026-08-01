import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        "wigner",
        sources=["src/wigner.pyx"],
        include_dirs=[numpy.get_include()],
    ),
    Extension(
        "lattice_utilities",
        sources=["src/lattice_utilities.pyx"],
        include_dirs=[numpy.get_include()],
    ),
]

setup(
    ext_modules=cythonize(extensions, language_level=3),
)