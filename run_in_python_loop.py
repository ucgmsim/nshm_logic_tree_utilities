from toshi_hazard_post.aggregation_args import AggregationArgs
from toshi_hazard_post.aggregation import run_aggregation
from nzshm_model.logic_tree.correlation import LogicTreeCorrelations
import os
import logging
import numpy as np
import copy
import matplotlib.pyplot as plt
import toml
from pathlib import Path
import subprocess
import shutil
import logic_tree_tools
from dataclasses import dataclass
from nzshm_model.logic_tree import GMCMLogicTree, SourceBranchSet, SourceLogicTree
from typing import Optional
import pandas as pd
import time
import itertools


def run_with_modified_logic_trees(args, output_dir, run_counter, custom_logic_tree_set, locations, output_staging_dir):

    run_start_time = time.time()

    modified_slt = copy.deepcopy(custom_logic_tree_set.slt)
    modified_glt = copy.deepcopy(custom_logic_tree_set.glt)

    logic_tree_tools.print_info_about_logic_tree_sets(custom_logic_tree_set)

    # check the validity of the weights
    logic_tree_tools.check_weight_validity(custom_logic_tree_set.slt)
    logic_tree_tools.check_weight_validity(custom_logic_tree_set.glt)

    modified_slt.to_json(output_staging_dir / f"slt_{run_counter}.json")
    modified_glt.to_json(output_staging_dir / f"glt_{run_counter}.json")

    custom_logic_tree_set.notes_to_toml(output_staging_dir / f"run_{run_counter}_notes.toml")

    if "model_version" in toml_dict["logic_trees"]:
        ## delete the key-value pair specifying that the logic tree should be the full built in logic tree
        del toml_dict["logic_trees"]["model_version"]

    for location in locations:
        print(f'doing run {run_counter} and location {location}')

        args.locations = [location]
        #toml_dict['site']['locations'] = [location]
        #toml_dict["general"]["hazard_model_id"] = f'run_{run_counter}'
        args.hazard_model_id = f'run_{run_counter}'

        ## set the key-value pair specifying that the logic tree should come from the custom logic tree files previously written
        # toml_dict["logic_trees"]["srm_file"] = str(output_staging_dir / f"slt_{run_counter}.json")
        # toml_dict["logic_trees"]["gmcm_file"] = str(output_staging_dir / f"glt_{run_counter}.json")

        args.srm_logic_tree = modified_slt
        args.gmcm_logic_tree = modified_glt


        # result = subprocess.run("python cli.py aggregate --config-file .env_home temp_input.toml",
        #                         shell=True, capture_output=True, text=True)

        # print(result.stdout)
        # print(result.stderr)
        run_aggregation(args)

    run_output_dir = output_dir / f"run_{run_counter}"
    run_output_dir.mkdir(parents=True, exist_ok=False)

    for file in output_staging_dir.iterdir():
        shutil.move(file, run_output_dir)

    run_end_time = time.time()
    print(f"Time taken for run {run_counter}: {(run_end_time - run_start_time)/60} mins")



def OLD_run_with_modified_logic_trees(output_dir, run_counter, custom_logic_tree_set, locations, toml_dict, output_staging_dir):

    run_start_time = time.time()

    #run_group_name = output_dir.name

    modified_slt = copy.deepcopy(custom_logic_tree_set.slt)
    modified_glt = copy.deepcopy(custom_logic_tree_set.glt)

    logic_tree_tools.print_info(custom_logic_tree_set)

    # check the validity of the weights
    logic_tree_tools.check_weight_validity(custom_logic_tree_set.slt)
    logic_tree_tools.check_weight_validity(custom_logic_tree_set.glt)

    modified_slt.to_json(output_staging_dir / f"slt_{run_counter}.json")
    modified_glt.to_json(output_staging_dir / f"glt_{run_counter}.json")

    custom_logic_tree_set.notes_to_toml(output_staging_dir / f"run_{run_counter}_notes.toml")

    if "model_version" in toml_dict["logic_trees"]:
        ## delete the key-value pair specifying that the logic tree should be the full built in logic tree
        del toml_dict["logic_trees"]["model_version"]

    for location in locations:
        print(f'doing run {run_counter} and location {location}')

        toml_dict['site']['locations'] = [location]
        toml_dict["general"]["hazard_model_id"] = f'run_{run_counter}'

        ## set the key-value pair specifying that the logic tree should come from the custom logic tree files previously written
        toml_dict["logic_trees"]["srm_file"] = str(output_staging_dir / f"slt_{run_counter}.json")
        toml_dict["logic_trees"]["gmcm_file"] = str(output_staging_dir / f"glt_{run_counter}.json")

        with open("temp_input.toml", "w") as f:
            toml.dump(toml_dict, f)



        result = subprocess.run("python cli.py aggregate --config-file .env_home temp_input.toml",
                                shell=True, capture_output=True, text=True)

        print(result.stdout)
        print(result.stderr)

        print()


    run_output_dir = output_dir / f"run_{run_counter}"
    run_output_dir.mkdir(parents=True, exist_ok=False)

    for file in output_staging_dir.iterdir():
        shutil.move(file, run_output_dir)

    run_end_time = time.time()
    print(f"Time taken for run {run_counter}: {(run_end_time - run_start_time)/60} mins")

def make_logic_tree_combinations_list_branch_sets(full_logic_tree, logic_tree_highest_weighted_branches):
    #from nzshm_model.logic_tree import GMCMLogicTree, SourceBranchSet, SourceLogicTree

    logic_tree_permutation_list = []

    for branch_set_index, branch_set in enumerate(full_logic_tree.branch_sets):

        modified_logic_tree = copy.deepcopy(full_logic_tree)

        modified_logic_tree.branch_sets[branch_set_index] = logic_tree_highest_weighted_branches.branch_sets[branch_set_index]
        modified_logic_tree.correlations = LogicTreeCorrelations()

        if isinstance(full_logic_tree, SourceLogicTree):
            custom_logic_tree_entry = logic_tree_tools.CustomLogicTreeSet(slt = modified_logic_tree,
                        slt_note = f"branch_set {branch_set.long_name} ({branch_set.short_name}) reduced to its single highest weighted branch. No other changes.")

        elif isinstance(full_logic_tree, GMCMLogicTree):
            custom_logic_tree_entry = logic_tree_tools.CustomLogicTreeSet(glt = modified_logic_tree,
                         glt_note = f"branch_set {branch_set.long_name} ({branch_set.short_name}) reduced to its single highest weighted branch. No other changes.")

        logic_tree_permutation_list.append(custom_logic_tree_entry)

    return logic_tree_permutation_list

def combine_logic_tree_combinations(slt_combinations, glt_combinations):

    combination_list = []

    for custom_slt_entry in slt_combinations:

        for custom_glt_entry in glt_combinations:

            slt_glt_entry = logic_tree_tools.CustomLogicTreeSet(slt=custom_slt_entry.slt,
                                               slt_note=custom_slt_entry.slt_note,
                                               glt=custom_glt_entry.glt,
                                               glt_note=custom_glt_entry.glt_note)

            combination_list.append(slt_glt_entry)

    # check that all required parameters are present
    check_validity_of_combinations(combination_list)
    return combination_list

def check_validity_of_combinations(logic_tree_permutation_list):

    for custom_logic_tree_entry in logic_tree_permutation_list:
        if custom_logic_tree_entry.slt is None:
            raise ValueError("slt is None")
        if custom_logic_tree_entry.slt_note is None:
            raise ValueError("slt_note is None")
        if custom_logic_tree_entry.glt is None:
            raise ValueError("glt is None")
        if custom_logic_tree_entry.glt_note is None:
            raise ValueError("glt_note is None")

    return True

def count_strings(string_list):
    count_dict = {}
    for string in string_list:
        if string in count_dict:
            count_dict[string] += 1
        else:
            count_dict[string] = 1
    return count_dict

start_time = time.time()

## copying logging from scripts/cli.py
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('toshi_hazard_post').setLevel(logging.INFO)
logging.getLogger('toshi_hazard_post.aggregation_calc').setLevel(logging.DEBUG)
logging.getLogger('toshi_hazard_post.aggregation').setLevel(logging.DEBUG)
logging.getLogger('toshi_hazard_post.aggregation_calc').setLevel(logging.DEBUG)
logging.getLogger('toshi_hazard_post.logic_tree').setLevel(logging.DEBUG)
logging.getLogger('toshi_hazard_post.parallel').setLevel(logging.DEBUG)
logging.getLogger('toshi_hazard_post').setLevel(logging.INFO)

delete_exisiting_output = False

input_file_dir = Path("custom_input_files")
output_dir = Path("/home/arr65/data/nshm/auto_output/auto22")

if delete_exisiting_output:
    shutil.rmtree(output_dir, ignore_errors=True)

output_dir.mkdir(parents=True, exist_ok=False)

os.environ['THP_ENV_FILE'] = str(input_file_dir / ".env_home")

initial_input_file = input_file_dir / "simple_input.toml"

with open(input_file_dir / ".env_home", 'r') as file:
    env_lines = file.readlines()

output_staging_dir = Path(env_lines[-1].split('=')[1].strip("\n \' \" "))
if delete_exisiting_output:
    shutil.rmtree(output_staging_dir, ignore_errors=True)
    output_staging_dir.mkdir(parents=True, exist_ok=True)


toml_dict = toml.load(initial_input_file)

# All locations can be specified in the same input file but this uses more memory than doing one location at a time
locations = ["AKL","WLG","CHC"]
#locations = ["WLG"]

args = AggregationArgs(initial_input_file)

args.locations = locations
args.output_individual_realizations = True

slt_full = args.srm_logic_tree
glt_full = args.gmcm_logic_tree

def print_branch_set_total_weight(logic_tree):

    logic_tree = copy.deepcopy(logic_tree)
    total_weight = 0.0

    for branch_set in logic_tree.branch_sets:
        for branch in branch_set.branches:
            total_weight += branch.weight
        print(f"{branch_set.short_name} total weight: {total_weight}")
        total_weight = 0.0


full_lt_set = logic_tree_tools.CustomLogicTreeSet(
    slt = copy.deepcopy(slt_full),
    glt = copy.deepcopy(glt_full),
    slt_note = "full > ",
    glt_note = 'full > ')

# logic_tree_set_list = logic_tree_tools.get_logic_tree_sets_for_individual_ground_motion_models(
#     initial_logic_tree_set = full_lt_set,
#     tectonic_region_type_sets=[["Active Shallow Crust"],["Subduction Interface"], ["Subduction Intraslab"]],
#     which_interfaces = ["only_HIK", "only_PUY", "HIK_and_PUY"])


logic_tree_set_list = logic_tree_tools.get_logic_tree_sets_for_individual_source_models(
    initial_logic_tree_set = full_lt_set,
    tectonic_region_type_sets = [["Active Shallow Crust"], ["Subduction Interface"], ["Subduction Intraslab"]],
    which_interfaces = ["only_HIK", "only_PUY", "HIK_and_PUY"])


logic_tree_tools.print_info_about_logic_tree_sets(logic_tree_set_list)

run_notes_df = pd.DataFrame()
for run_counter, custom_logic_tree_set in enumerate(logic_tree_set_list):
    notes_df = custom_logic_tree_set.notes_to_pandas_df()
    notes_df['run_counter'] = [run_counter]
    run_notes_df = pd.concat([run_notes_df, notes_df], ignore_index=True)

# move the "run_counter" column to the left-most position
run_notes_df.insert(0, "run_counter", run_notes_df.pop("run_counter"))
run_notes_df.to_csv(output_dir / "run_notes.csv")

print()

for run_counter, custom_logic_tree_set in enumerate(logic_tree_set_list):
    run_with_modified_logic_trees(args, output_dir, run_counter, custom_logic_tree_set, locations, output_staging_dir)

end_time = time.time()
print(f"Time taken: {(end_time - start_time)/60} mins for {len(logic_tree_set_list)} runs")

# def run_with_modified_logic_trees(output_dir, run_counter, custom_logic_tree_set, locations, toml_dict, output_staging_dir):