from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "spn_datasets.generator._cython_arrivable_graph",
        ["src/spn_datasets/generator/_cython_arrivable_graph.pyx"],
        include_dirs=[np.get_include()],
        language="c++",
        extra_compile_args=["-O3", "-std=c++11"],
    ),
    Extension(
        "spn_datasets.generator._cython_data_transformation",
        ["src/spn_datasets/generator/_cython_data_transformation.pyx"],
        include_dirs=[np.get_include()],
        language="c++",
        extra_compile_args=["-O3", "-std=c++11"],
    ),
    Extension(
        "spn_datasets.generator._cython_petri_generate",
        ["src/spn_datasets/generator/_cython_petri_generate.pyx"],
        include_dirs=[np.get_include()],
        language="c++",
        extra_compile_args=["-O3", "-std=c++11"],
    ),
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"}
    ),
)
