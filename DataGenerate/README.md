# Data Generation Scripts

This directory contains the core scripts for generating and processing Stochastic Petri Net (SPN) datasets.

## Files

-   `SPN.py`: This module provides the core functionalities for working with SPNs. It includes functions to compute the state equation, solve for steady-state probabilities, calculate average token markings, and filter SPNs based on various criteria.

-   `PetriGenerate.py`: This script is responsible for generating the initial structure of Petri nets. It contains functions to create random Petri net matrices and perform structural modifications like pruning or adding tokens.

-   `DataTransformation.py`: This module handles the augmentation and transformation of existing Petri net data. It can be used to create new data samples from existing ones by applying transformations.

-   `ArrivableGraph.py`: This script is used to generate the reachability graph (also known as the state space) of a Petri net. It uses a Breadth-First Search (BFS) algorithm to find all reachable markings from a given initial state and checks for the boundedness of the net.

-   `wherevec.pyx`: This is a Cython file, likely used to optimize a specific, computationally intensive part of the code. Cython files are compiled into C and can offer significant speed improvements for certain operations.

-   `__init__.py`: This file makes the `DataGenerate` directory a Python package, allowing its modules to be imported elsewhere in the project.
