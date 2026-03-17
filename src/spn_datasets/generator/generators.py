import abc
import random
import numpy as np
from spn_datasets.generator import PetriGenerate

class PetriNetGenerator(abc.ABC):
    """Abstract base class for Petri net generators."""

    @abc.abstractmethod
    def generate(self, num_places: int, num_transitions: int) -> np.ndarray:
        """Generates a Petri net matrix with the specified number of places and transitions.

        Args:
            num_places (int): The number of places.
            num_transitions (int): The number of transitions.

        Returns:
            np.ndarray: The generated Petri net matrix.
        """
        pass


class RandomPetriNetGenerator(PetriNetGenerator):
    """Generates random Petri nets using the original algorithm."""

    def generate(self, num_places: int, num_transitions: int) -> np.ndarray:
        return PetriGenerate.generate_random_petri_net(num_places, num_transitions)


class RuleBasedPetriNetGenerator(PetriNetGenerator):
    """Generates Petri nets using a rule-based growth approach."""

    def generate(self, num_places: int, num_transitions: int) -> np.ndarray:
        # Initialize internal graph structure
        # Places are indexed 0 to num_places-1
        # Transitions are indexed 0 to num_transitions-1

        # Start with a simple cycle: P0 -> T0 -> P0
        current_places = [0]
        current_transitions = [0]
        # Edges: (source_type, source_id, target_id)
        # source_type: 'P' or 'T'
        edges = [('P', 0, 0), ('T', 0, 0)]

        # Grow the network
        while len(current_places) < num_places or len(current_transitions) < num_transitions:
            # Determine available operations based on current counts
            can_add_place = len(current_places) < num_places
            can_add_transition = len(current_transitions) < num_transitions

            # Prioritize balanced growth if possible
            if can_add_place and can_add_transition:
                op = random.choice(['split_arc', 'split_arc', 'parallel_place', 'parallel_transition'])
            elif can_add_place:
                op = 'parallel_place'
            elif can_add_transition:
                op = 'parallel_transition'
            else:
                break

            if op == 'split_arc' and can_add_place and can_add_transition:
                # Split an existing edge (P->T or T->P) by inserting T_new -> P_new (or P_new -> T_new)
                edge_idx = random.randrange(len(edges))
                u_type, u, v = edges[edge_idx]

                # New IDs
                p_new = len(current_places)
                t_new = len(current_transitions)
                current_places.append(p_new)
                current_transitions.append(t_new)

                # Remove old edge
                edges.pop(edge_idx)

                if u_type == 'P':
                    # Old: P(u) -> T(v)
                    # New: P(u) -> T(t_new) -> P(p_new) -> T(v)
                    edges.append(('P', u, t_new))
                    edges.append(('T', t_new, p_new))
                    edges.append(('P', p_new, v))
                else:
                    # Old: T(u) -> P(v)
                    # New: T(u) -> P(p_new) -> T(t_new) -> P(v)
                    edges.append(('T', u, p_new))
                    edges.append(('P', p_new, t_new))
                    edges.append(('T', t_new, v))

            elif op == 'parallel_place' and can_add_place:
                # Add a place between two transitions (or same transition for self-loop)
                # Pick two random transitions (can be the same)
                t1 = random.choice(current_transitions)
                t2 = random.choice(current_transitions)

                p_new = len(current_places)
                current_places.append(p_new)

                edges.append(('T', t1, p_new))
                edges.append(('P', p_new, t2))

            elif op == 'parallel_transition' and can_add_transition:
                # Add a transition between two places
                p1 = random.choice(current_places)
                p2 = random.choice(current_places)

                t_new = len(current_transitions)
                current_transitions.append(t_new)

                edges.append(('P', p1, t_new))
                edges.append(('T', t_new, p2))

        # Convert to matrix format
        # Matrix shape: (num_places, 2 * num_transitions + 1)
        # Columns 0..T-1: P->T edges (Input to transitions)
        # Columns T..2T-1: T->P edges (Output from transitions)
        # Last column: Initial marking

        petri_matrix = np.zeros((num_places, 2 * num_transitions + 1), dtype=np.int32)

        for u_type, u, v in edges:
            if u_type == 'P':
                # P(u) -> T(v). Set M[u, v] = 1
                petri_matrix[u, v] = 1
            else:
                # T(u) -> P(v). Set M[v, u + num_transitions] = 1
                petri_matrix[v, u + num_transitions] = 1

        # Add initial marking (one token in a random place)
        random_place = random.choice(current_places)
        petri_matrix[random_place, -1] = 1

        return petri_matrix
