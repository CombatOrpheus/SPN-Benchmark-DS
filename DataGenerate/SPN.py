#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @File    : SPN.py
# @Date    : 2020-08-22
# @Author  : mingjian
    描述
"""

import numpy as np
from numpy.linalg import solve

from DataGenerate import ArrivableGraph as ArrGra


def state_equation(vertices, edges, arc_transitions, lambda_values):
    num_vertices = len(vertices)
    redundant_state_matrix = np.zeros((num_vertices + 1, num_vertices), dtype=int)
    y_list = np.zeros(num_vertices, dtype=int)
    y_list[-1] = 1
    redundant_state_matrix[-1, :] = 1

    for edge, arc_transition in zip(edges, arc_transitions):
        v1, v2 = edge
        redundant_state_matrix[v1, v1] -= lambda_values[arc_transition]
        redundant_state_matrix[v2, v1] += lambda_values[arc_transition]

    state_matrix = []
    for i in range(num_vertices - 1):
        state_matrix.append(redundant_state_matrix[i])
    state_matrix.append(redundant_state_matrix[-1, :])

    return state_matrix, y_list


def avg_mark_nums(vertices, steady_state_probabilities):
    unique_tokens = []
    for vertex in vertices:
        unique_tokens.extend(np.unique(vertex))
    unique_tokens = np.unique(unique_tokens)

    mark_density_matrix = np.zeros((len(vertices[0]), len(unique_tokens)))
    vertices_array = np.array(vertices)

    for vertex_index in range(len(vertices[0])):
        vertex_tokens = np.unique(vertices_array[:, vertex_index])
        for token in vertex_tokens:
            token_index = np.where(unique_tokens == token)[0][0]
            token_indices_in_vertex = np.where(vertices_array[:, vertex_index] == token)[0]
            mark_density_matrix[vertex_index][token_index] = np.sum(
                steady_state_probabilities[token_indices_in_vertex]
            )

    # Calculate the average mark numbers for each vertex
    average_mark_numbers = np.sum(mark_density_matrix * unique_tokens, axis=1).tolist()

    return mark_density_matrix, average_mark_numbers


def generate_sgn_task(v_list, edage_list, arctrans_list, tran_num):
    labda = np.random.randint(1, 11, size=tran_num)
    state_matrix, y_list = state_equation(v_list, edage_list, arctrans_list, labda)
    sv = None
    try:
        sv = solve(state_matrix, y_list.T)
        mark_dens_list, mu_mark_nums = avg_mark_nums(v_list, sv)
    except np.linalg.linalg.LinAlgError:
        mark_dens_list, mu_mark_nums = None, None
    return sv, mark_dens_list, mu_mark_nums, labda


def convert_data(npdata):
    return np.array(npdata).astype(int).tolist()


def is_connected_graph(petri_matrix):
    petri_matrix = np.array(petri_matrix)
    trans_num = len(petri_matrix[0]) // 2
    flag = True
    for row in range(len(petri_matrix)):
        if np.sum(petri_matrix[row, :-1]) == 0:
            return False
    for col in range(trans_num):
        if np.sum(petri_matrix[:, col]) + np.sum(petri_matrix[:, col + trans_num]) == 0:
            return False
    return flag


def filter_spn(
    petri_matrix, place_upper_bound=10, marks_lower_limit=4, marks_upper_limit=500
):
    v_list, edage_list, arctrans_list, tran_num, bound_flag = ArrGra.get_arr_gra(
        petri_matrix, place_upper_bound, marks_upper_limit
    )
    results_dict = {}
    if not bound_flag or len(v_list) < marks_lower_limit:
        return results_dict, False
    sv, mark_dens_list, mu_mark_nums, labda = generate_sgn_task(
        v_list, edage_list, arctrans_list, tran_num
    )
    if sv is None:
        return results_dict, False

    if not is_connected_graph(petri_matrix):
        return results_dict, False

    results_dict["petri_net"] = convert_data(petri_matrix)
    results_dict["arr_vlist"] = convert_data(v_list)
    results_dict["arr_edge"] = convert_data(edage_list)
    results_dict["arr_tranidx"] = convert_data(arctrans_list)
    results_dict["spn_labda"] = np.array(labda).tolist()
    results_dict["spn_steadypro"] = np.array(sv).tolist()
    results_dict["spn_markdens"] = np.array(mark_dens_list).tolist()
    results_dict["spn_allmus"] = np.array(mu_mark_nums).tolist()
    results_dict["spn_mu"] = np.sum(mu_mark_nums)

    return results_dict, True


def generate_sgn_task_given_labda(v_list, edage_list, arctrans_list, labda):
    state_matrix, y_list = state_equation(v_list, edage_list, arctrans_list, labda)
    sv = None
    try:
        sv = solve(state_matrix, y_list.T)
        mark_dens_list, mu_mark_nums = avg_mark_nums(v_list, sv)
    except np.linalg.linalg.LinAlgError:
        mark_dens_list, mu_mark_nums = None, None
    return sv, mark_dens_list, mu_mark_nums


def get_spn(petri_matrix, v_list, edage_list, arctrans_list, labda):
    results_dict = {}
    sv, mark_dens_list, mu_mark_nums = generate_sgn_task_given_labda(
        v_list, edage_list, arctrans_list, labda
    )
    if sv is None:
        return results_dict, False
    if not is_connected_graph(petri_matrix):
        return results_dict, False
    results_dict["petri_net"] = convert_data(petri_matrix)
    results_dict["arr_vlist"] = convert_data(v_list)
    results_dict["arr_edge"] = convert_data(edage_list)
    results_dict["arr_tranidx"] = convert_data(arctrans_list)
    results_dict["spn_labda"] = np.array(labda).tolist()
    results_dict["spn_steadypro"] = np.array(sv).tolist()
    results_dict["spn_markdens"] = np.array(mark_dens_list).tolist()
    results_dict["spn_allmus"] = np.array(mu_mark_nums).tolist()
    results_dict["spn_mu"] = np.sum(mu_mark_nums)
    return results_dict, True


def get_spnds3(
    petri_matrix,
    labda,
    place_upper_bound=10,
    marks_lower_limit=4,
    marks_upper_limit=500,
):
    v_list, edage_list, arctrans_list, tran_num, bound_flag = ArrGra.get_arr_gra(
        petri_matrix, place_upper_bound, marks_upper_limit
    )
    results_dict = {}
    if not bound_flag or len(v_list) < marks_lower_limit:
        return results_dict, False
    sv, mark_dens_list, mu_mark_nums = generate_sgn_task_given_labda(
        v_list, edage_list, arctrans_list, labda
    )
    if sv is None:
        return results_dict, False

    if not is_connected_graph(petri_matrix):
        return results_dict, False
    mu_sums = np.sum(mu_mark_nums)
    if mu_sums < -100 and mu_sums > 100:
        return results_dict, False
    results_dict["petri_net"] = convert_data(petri_matrix)
    results_dict["arr_vlist"] = convert_data(v_list)
    results_dict["arr_edge"] = convert_data(edage_list)
    results_dict["arr_tranidx"] = convert_data(arctrans_list)
    results_dict["spn_labda"] = np.array(labda).tolist()
    results_dict["spn_steadypro"] = np.array(sv).tolist()
    results_dict["spn_markdens"] = np.array(mark_dens_list).tolist()
    results_dict["spn_allmus"] = np.array(mu_mark_nums).tolist()
    results_dict["spn_mu"] = np.sum(mu_mark_nums)
    return results_dict, True
