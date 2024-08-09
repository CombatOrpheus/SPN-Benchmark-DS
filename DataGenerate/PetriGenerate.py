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
from random import choice


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
    remain_node = [i + 1 for i in range(place_num + tran_num)]
    petri_matrix = np.zeros((place_num, 2 * tran_num + 1), dtype=int)
    # The first selected point in the picture, randomly find other points for him to connect
    first_p = choice([x for x in remain_node if x <= place_num])
    first_t = choice([x for x in remain_node if x > place_num])

    remain_node.remove(first_p)
    remain_node.remove(first_t)
    rand_num = np.random.rand(0, 1)
    if rand_num <= 0.5:
        petri_matrix[first_p - 1][first_t - place_num - 1] = 1
    else:
        petri_matrix[first_p - 1][first_t - place_num - 1 + tran_num] = 1
    np.random.shuffle(remain_node)

    sub_graph = np.array(([first_p, first_t]))
    for r_node in np.random.permutation(remain_node):
        subp_list = sub_graph[sub_graph <= place_num]
        subt_list = sub_graph[sub_graph > place_num]

        if r_node <= place_num:  # Is it a place?
            p = r_node
            t = choice(subt_list)
        else:
            p = choice(subp_list)
            t = r_node
        
        if rand_num <= 0.5:
            petri_matrix[p - 1][t - place_num - 1] = 1
        else:
            petri_matrix[p - 1][t - place_num - 1 + tran_num] = 1
        sub_graph = np.concatenate((sub_graph, [r_node]))
        remain_node.remove(r_node)

    
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
