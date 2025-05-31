#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @File    : SPNGenerate.py
# @Date    : 2020-08-23
# @Author  : mingjian
    描述
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm, trange

from DataGenerate import SPN, DataTransformation
from DataGenerate import PetriGenerate as PeGen
from DataVisualization import Visual
from utils import DataUtil as DU


def generate_spn(config, write_location, data_index):
    place_upper_bound = config["place_upper_bound"]
    marks_lower_limit = config["marks_lower_limit"]
    marks_upper_limit = config["marks_upper_limit"]
    prune_flag = config["prune_flag"]
    add_token = config["add_token"]
    max_place_number = config["max_place_num"]
    min_place_number = config["min_place_num"]
    spn_generation_finished = False
    while not spn_generation_finished:
        place_number = np.random.randint(min_place_number, max_place_number + 1)
        transition_number = place_number + np.random.randint(-3, 1)
        petri_matrix = PeGen.generate_random_petri_net(place_number, transition_number)
        if prune_flag:
            petri_matrix = PeGen.prune_petri_net(petri_matrix)
        if add_token:
            petri_matrix = PeGen.add_token_to_random_place(petri_matrix)
        results_dict, spn_generation_finished = SPN.filter_stochastic_petri_net(
            petri_matrix, place_upper_bound, marks_lower_limit, marks_upper_limit
        )
    DU.save_data_to_json(
        os.path.join(write_location, "data%s.json" % str(data_index)), results_dict
    )


def augment_single_data(data, place_upper_bound, marks_lower_limit, marks_upper_limit, maxtransform_num):
    all_extended_data = DataTransformation.transformation(
        np.array(data["petri_net"], dtype="long"),
        place_upper_bound,
        marks_lower_limit,
        marks_upper_limit,
    )
    if len(all_extended_data) >= maxtransform_num:
        data_range = np.arange(len(all_extended_data))
        sample_indices = np.random.choice(
            data_range, maxtransform_num, replace=False
        )
    else:
        sample_indices = np.arange(len(all_extended_data))
    transformed_data_list = []
    for selected_index in sample_indices:
        transformed_data_list.append(all_extended_data[selected_index])
    return transformed_data_list


def visualize_single_data(item):
    data_index, data, writable_picture_location = item
    Visual.plot_petri(
        data["petri_net"],
        os.path.join(writable_picture_location, f"data(petri){data_index + 1}"),
    )
    Visual.plot_spn(
        data["arr_vlist"],
        data["arr_edge"],
        data["arr_tranidx"],
        data["spn_labda"],
        data["spn_steadypro"],
        data["spn_markdens"],
        data["spn_allmus"],
        os.path.join(writable_picture_location, f"data(arr){data_index + 1}"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="Please give a config.json file ",
        default="config/DataConfig/SPNGenerate.json",
    )
    args = parser.parse_args()
    config = DU.load_json(args.config)
    print(config)
    write_data_location = config["write_data_loc"]
    parallel_job = config["parallel_job"]
    place_upper_bound = config["place_upper_bound"]
    marks_lower_limit = config["marks_lower_limit"]
    marks_upper_limit = config["marks_upper_limit"]
    data_number = config["data_num"]
    visual_flag = config["visual_flag"]
    picture_location = config["pic_loc"]
    transformation_flag = config["transformation_flag"]
    maximum_transformation_number = config["maxtransform_num"]
    DU.mkdir(write_data_location)
    temporary_write_directory = os.path.join(write_data_location, "tmp")
    writable_picture_location = os.path.join(write_data_location, picture_location)

    print(temporary_write_directory)
    DU.mkdir(temporary_write_directory)
    Parallel(n_jobs=parallel_job, backend="loky")(
        delayed(generate_spn)(config, temporary_write_directory, i + 1)
        for i in trange(data_number, desc="Data Generation")
    )
    #
    # for i in trange(data_number, desc="Data Generation"):
    #     generate_spn(config, temporary_write_directory, i + 1)

    all_data = DU.load_alldata_from_json(temporary_write_directory)
    original_data_location = "ori_data"
    DU.mkdir(os.path.join(write_data_location, original_data_location))
    path = os.path.join(write_data_location, original_data_location, "all_data.json")

    transformed_data = {}
    counter = 0
    if transformation_flag:
        augmented_data_list = Parallel(n_jobs=parallel_job, backend="loky", return_as="generator")(
            delayed(augment_single_data)(
                data, place_upper_bound, marks_lower_limit, marks_upper_limit, maximum_transformation_number
            )
            for data in tqdm(all_data, desc="Data Augmentation", total=data_number)
        )

        with open(path, 'w') as file:
            for sublist in augmented_data_list:
                for data in sublist:
                    json.dump(data, file)
                    file.write('\n')
                    counter += 1

    if not transformation_flag:
        all_data = list(all_data)
        counter = len(all_data)
        DU.save_data_to_json(
            os.path.join(write_data_location, original_data_location, "all_data.json"), all_data
        )

    print(f"total data number : {counter}")

    if visual_flag:
        DU.mkdir(writable_picture_location)
        Parallel(n_jobs=parallel_job, backend="loky")(
            delayed(visualize_single_data)((i, data, writable_picture_location))
            for i, data in tqdm(enumerate(all_data.values()), desc="Visual", total=len(all_data))
        )
    shutil.rmtree(temporary_write_directory)