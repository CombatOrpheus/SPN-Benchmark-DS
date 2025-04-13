#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @File    : PetriGenerate.py
# @Date    : 2020-08-22
# @Author  : mingjian
    描述
"""

from random import choice

import numpy as np


def generate_random_petri_net(num_places: int, num_transitions: int) -> np.ndarray:
    """
    Generates a random Petri net matrix.

    Args:
        num_places (int): The number of places in the Petri net.
        num_transitions (int): The number of transitions in the Petri net.

    Returns:
        np.ndarray: A Petri net matrix of shape (num_places, 2 * num_transitions + 1).
                       The columns represent (pre-transitions, post-transitions, initial_marking).
    """
    remain_node = [i + 1 for i in range(num_places + num_transitions)]
    petri_matrix = np.zeros((num_places, 2 * num_transitions + 1), dtype="int32")
    first_p = choice(range(num_places)) + 1
    first_t = choice(range(num_transitions)) + num_places + 1

    remain_node.remove(first_p)
    remain_node.remove(first_t)
    rand_num = np.random.rand()
    if rand_num <= 0.5:
        petri_matrix[first_p - 1][first_t - num_places - 1] = 1
    else:
        petri_matrix[first_p - 1][first_t - num_places - 1 + num_transitions] = 1
    np.random.shuffle(remain_node)

    sub_graph = np.array(([first_p, first_t]))
    for r_node in np.random.permutation(remain_node):
        subp_list = sub_graph[sub_graph <= num_places]
        subt_list = sub_graph[sub_graph > num_places]

        if r_node <= num_places:  # Is it a place?
            p = r_node
            t = choice(subt_list)
        else:
            p = choice(subp_list)
            t = r_node

        if rand_num <= 0.5:
            petri_matrix[p - 1][t - num_places - 1] = 1
        else:
            petri_matrix[p - 1][t - num_places - 1 + num_transitions] = 1
        sub_graph = np.concatenate((sub_graph, [r_node]))
        remain_node.remove(r_node)

    rand_num = np.random.randint(0, num_places)
    petri_matrix[rand_num][-1] = 1

    rand_nums = np.random.randint(1, 10, np.shape(petri_matrix)) <= 1
    zero_idxs = petri_matrix == 0
    petri_matrix[rand_nums & zero_idxs] = 1

    return petri_matrix


# Suggested name: prune_petri_net
def prune_petri_net(petri_matrix: np.ndarray) -> np.ndarray:
    """
    Prunes the given Petri net matrix by deleting some edges and adding nodes.

    Args:
        petri_matrix (np.ndarray): The Petri net matrix to prune.

    Returns:
        np.ndarray: The pruned Petri net matrix.
    """
    tran_num = (len(petri_matrix[0]) - 1) // 2
    petri_matrix = delete_excess_edges(petri_matrix, tran_num)
    petri_matrix = add_necessary_nodes(petri_matrix, tran_num)

    return petri_matrix


# Suggested name: delete_excess_edges
def delete_excess_edges(gra_matrix: np.ndarray, tran_num: int) -> np.ndarray:
    """
    Deletes excess edges from the Petri net matrix.

    Args:
        gra_matrix (np.ndarray): The Petri net matrix.
        tran_num (int): The number of transitions in the Petri net.

    Returns:
        np.ndarray: The Petri net matrix with some edges deleted.
    """
    for row in range(len(gra_matrix)):
        if np.sum(gra_matrix[row, 0:-1]) >= 3:
            itemindex = np.argwhere(gra_matrix[row, 0:-1] == 1).flatten()
            # print(itemindex)
            rmindex = np.random.choice(itemindex, len(itemindex) - 2, replace=False)
            gra_matrix[row][rmindex] = 0

    for i in range(2 * tran_num):
        if np.sum(gra_matrix[:, i]) >= 3:
            itemindex = np.argwhere(gra_matrix[:, i] == 1).flatten()
            rmindex = np.random.choice(itemindex, len(itemindex) - 2, replace=False)
            for rmidx in rmindex:
                gra_matrix[rmidx][i] = 0

    return gra_matrix


def add_necessary_nodes(petri_matrix: np.ndarray, tran_num: int) -> np.ndarray:
    """
    Adds necessary nodes (edges) to the Petri net matrix to ensure connectivity.

    Args:
        petri_matrix (np.ndarray): The Petri net matrix.
        tran_num (int): The number of transitions in the Petri net.

    Returns:
        np.ndarray: The Petri net matrix with necessary nodes added.
    """
    leftmatrix = petri_matrix[:, 0:tran_num]
    rightmatrix = petri_matrix[:, tran_num:-1]

    # each column must have a 1
    zero_sum_cols = np.where(np.sum(petri_matrix[:, :2 * tran_num], axis=0) < 1)[0]
    random_indices_cols = np.random.randint(0, len(petri_matrix), size=len(zero_sum_cols))
    petri_matrix[random_indices_cols, zero_sum_cols] = 1

    # Each row must have two elements of 1, the left matrix has 1, and the right must also have 1
    rows_with_zero_left_sum = np.where(np.sum(leftmatrix, axis=1) < 1)[0]
    random_indices_left = np.random.randint(0, tran_num, size=len(rows_with_zero_left_sum))
    petri_matrix[rows_with_zero_left_sum, random_indices_left] = 1

    rows_with_zero_right_sum = np.where(np.sum(rightmatrix, axis=1) < 1)[0]
    random_indices_right = np.random.randint(0, tran_num, size=len(rows_with_zero_right_sum))
    petri_matrix[rows_with_zero_right_sum, random_indices_right + tran_num] = 1

    return petri_matrix


def add_token_to_random_place(petri_matrix: np.ndarray) -> np.ndarray:
    """
    Randomly adds tokens to the places in the Petri net matrix (vectorized).

    Args:
        petri_matrix (np.ndarray): The Petri net matrix.

    Returns:
        np.ndarray: The Petri net matrix with potentially added tokens.
    """
    random_values = np.random.randint(0, 10, size=len(petri_matrix))
    petri_matrix[:, -1] += (random_values <= 2).astype(int)
    return petri_matrix
