#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @File    : PetriGenerate.py
# @Date    : 2020-08-22
# @Author  : mingjian
    描述
"""
import numba
import numpy as np


@numba.jit(nopython=True, cache=True)
def split_pt(node_list, place_num):
    p_list = [x for x in node_list if x <= place_num]
    t_list = [x for x in node_list if x > place_num]

    return p_list, t_list


@numba.jit(nopython=True, cache=True)
def dele_edage(gra_matrix, tran_num):
    for row in range(len(gra_matrix)):
        if np.sum(gra_matrix[row, 0:-1]) >= 3:
            itemindex = np.argwhere(gra_matrix[row, 0:-1] == 1).flatten()
            # print(itemindex)
            rmindex = np.random.choice(itemindex, len(itemindex) - 2)
            gra_matrix[row][rmindex] = 0

    for i in range(2*tran_num):
        if np.sum(gra_matrix[:, i]) >= 3:
            itemindex = np.argwhere(gra_matrix[:, i] == 1).flatten()
            rmindex = np.random.choice(itemindex, len(itemindex) - 2)
            for rmidx in rmindex:
                gra_matrix[rmidx][i] = 0

    return gra_matrix


@numba.jit(nopython=True, cache=True)
def add_node(petri_matrix, tran_num):
    leftmatrix = petri_matrix[:, 0:tran_num]
    rightmatrix = petri_matrix[:, tran_num:-1]

    # each column must have a 1
    for i in range(2*tran_num):
        if np.sum(petri_matrix[:, i]) < 1:
            rand_idx = np.random.randint(0, len(petri_matrix))
            petri_matrix[rand_idx][i] = 1

    # Each row must have two elements of 1, the left matrix has 1, and the right must also have 1
    for i in range(len(petri_matrix)):
        if np.sum(leftmatrix[i]) < 1:
            rand_idx = np.random.randint(0, tran_num)
            petri_matrix[i][rand_idx] = 1

        if np.sum(rightmatrix[i]) < 1:
            rand_idx = np.random.randint(0, tran_num)
            petri_matrix[i][rand_idx + tran_num] = 1

    return petri_matrix


def rand_generate_petri(place_num, tran_num):
    sub_graph = []
    remain_node = [i + 1 for i in range(place_num + tran_num)]
    petri_matrix = np.zeros((place_num, 2 * tran_num + 1), dtype=int)
    # The first selected point in the picture, randomly find other points for him to connect
    p_list, t_list = split_pt(remain_node, place_num)
    first_p = np.random.choice(p_list)
    first_t = np.random.choice(t_list)

    sub_graph.extend([first_p, first_t])
    remain_node.remove(first_p)
    remain_node.remove(first_t)
    rand_num = np.random.rand(0, 1)
    if rand_num <= 0.5:
        petri_matrix[first_p - 1][first_t - place_num - 1] = 1
    else:
        petri_matrix[first_p - 1][first_t - place_num - 1 + tran_num] = 1
    np.random.shuffle(remain_node)

    for r_node in np.random.permutation(remain_node):
        subp_list, subt_list = split_pt(sub_graph, place_num)

        if r_node <= place_num:  # Is it a place?
            p = r_node
            t = np.random.choice(subt_list)
        else:
            p = np.random.choice(subp_list)
            t = r_node
        
        if rand_num <= 0.5:
            petri_matrix[p - 1][t - place_num - 1] = 1
        else:
            petri_matrix[p - 1][t - place_num - 1 + tran_num] = 1
        sub_graph.append(r_node)
        remain_node.remove(r_node)

    # The front is to prevent isolated subgraphs

    # token
    rand_num = np.random.randint(0, place_num)
    petri_matrix[rand_num][-1] = 1

    rand_nums = np.random.randint(1, 10, np.shape(petri_matrix)) <= 1
    zero_idxs = petri_matrix == 0
    petri_matrix[rand_nums & zero_idxs] = 1

    return petri_matrix


@numba.jit(nopython=True, cache=True)
def prune_petri(petri_matrix):
    tran_num = (len(petri_matrix[0]) - 1) // 2
    petri_matrix = dele_edage(petri_matrix, tran_num)
    petri_matrix = add_node(petri_matrix, tran_num)

    return petri_matrix


@numba.jit(nopython=True, cache=True)
def add_token(petri_matrix):
    for i in range(len(petri_matrix)):
        rand_num = np.random.randint(0, 10)
        # print(rand_num)
        if rand_num <= 2:
            petri_matrix[i][-1] += 1

    return petri_matrix
