#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @File    : ArrivableGraph.py
# @Date    : 2020-08-22
# @Author  : mingjian
    描述
"""
import numpy as np
import numba
from random import choice
from .wherevec import wherevec_cython

@numba.jit(nopython=True, cache=True)
def wherevec(vec, matrix):
    for row in range(len(matrix)):
        if np.allclose(matrix[row, :], vec):
            return row
    return -1


def enabled_sets(pre_set, post_set, M):
    # Find enabled transitions
    M = np.expand_dims(M, axis=1)
    enabled_transitions = np.where(np.all(M >= pre_set, axis=0))[0]

    # Calculate new markings
    new_markings = M - pre_set[:, enabled_transitions] + post_set[:, enabled_transitions]

    return new_markings, enabled_transitions



def get_arr_gra(petri_matrix, place_upper_limit=10, marks_upper_limit=500):
    """
    Obtain the reachable graph of the petri net.

    :param
        petri_matrix: petri net matrix
        upper_limit: The upper bound of the place in petri net, if > upper_limit : unbound petri , else : bound petri
    :return:
        v_list : The set of all vertices of the reachable graph.
        edage_list : The set of all edges of the reachable graph.
        arctrans_list : The set of arc transitions of the reachable graph.
        tran_num : Number of transitions.
        bound_flag : Whether it is a bounded net. If yes, return True, otherwise False.
    """

    petri_matrix = np.array(petri_matrix)
    bound_flag = True
    tran_num = int(petri_matrix.shape[1] / 2)
    # place_num = petri_matrix.shape[0]
    leftmatrix = petri_matrix[:, 0:tran_num]
    rightmatrix = petri_matrix[:, tran_num:-1]
    M0 = np.array(petri_matrix[:, -1], dtype=int)
    counter = 0
    v_list = [M0]
    new_list = [counter]
    edage_list = []
    arctrans_list = []
    C = (rightmatrix - leftmatrix)
    while len(new_list) > 0:

        if counter > marks_upper_limit:
            bound_flag = False
            return v_list, edage_list, arctrans_list, tran_num, bound_flag

        new_m = choice(new_list)
        gra_en_sets, tran_sets = enabled_sets(leftmatrix, rightmatrix, v_list[new_m])
        if np.any(np.asarray(gra_en_sets) > place_upper_limit):
            bound_flag = False
            return v_list, edage_list, arctrans_list, tran_num, bound_flag

        if len(gra_en_sets) == 0:
            new_list.remove(new_m)
        else:
            # Traverse all enable marks
            for en_m, ent_idx in zip(gra_en_sets, tran_sets):
                # Calculate the current enable transition, generate a new mark and save it in M_new.
                M_new = np.asarray(v_list[new_m] + C[:, ent_idx], dtype=int)
                M_newidx = wherevec_cython(M_new, np.asarray(v_list))
                if M_newidx == -1:
                    counter += 1
                    v_list.append(M_new)
                    new_list.append(counter)
                    edage_list.append([new_m, counter])
                else:
                    edage_list.append([new_m, M_newidx])
                    
                arctrans_list.append(ent_idx)
            new_list.remove(new_m)

    return v_list, edage_list, arctrans_list, tran_num, bound_flag
