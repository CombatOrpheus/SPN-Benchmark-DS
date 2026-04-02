from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import os

# Create a custom build_ext class to ignore compilation errors
# This makes Cython extensions strictly optional.
class BuildExtOptional(build_ext):
    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as e:
            print(f"Failed to build extension {ext.name}: {e}")
            print("Skipping and falling back to pure Python / Numba implementations.")

try:
    from Cython.Build import cythonize
    import numpy as np

    extensions = [
        Extension(
            "spn_datasets.generator.cy_arrivable_graph",
            ["src/spn_datasets/generator/cy_arrivable_graph.pyx"],
            include_dirs=[np.get_include()],
            language="c++",
        ),
        Extension(
            "spn_datasets.generator.cy_data_transformation",
            ["src/spn_datasets/generator/cy_data_transformation.pyx"],
            include_dirs=[np.get_include()],
            language="c++",
        ),
        Extension(
            "spn_datasets.generator.cy_petri_generate",
            ["src/spn_datasets/generator/cy_petri_generate.pyx"],
            include_dirs=[np.get_include()],
        ),
        Extension(
            "spn_datasets.generator.cy_spn",
            ["src/spn_datasets/generator/cy_spn.pyx"],
            include_dirs=[np.get_include()],
            language="c++",
        ),
    ]
    ext_modules = cythonize(extensions, compiler_directives={'language_level': "3", 'boundscheck': False, 'wraparound': False})
except ImportError:
    print("Cython or NumPy not found. Extensions will not be built.")
    ext_modules = []

setup(
    name="SPN-Benchmarks",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtOptional},
)
