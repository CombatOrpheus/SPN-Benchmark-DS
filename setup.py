from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import os

here = os.path.dirname(__file__)

extensions = [
    Extension(
        "spn_datasets.generator.arrivable_graph_cy",
        ["src/spn_datasets/generator/arrivable_graph_cy.pyx"],
        include_dirs=[numpy.get_include()],
        language="c++"
    )
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
    # For making package pip-installable seamlessly when `setup.py` is called.
    packages=["spn_datasets", "spn_datasets.generator", "spn_datasets.utils", "spn_datasets.gnns"],
    package_dir={"": "src"},
)
