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

delete_exisiting_output = True

input_file_dir = Path("custom_input_files")
output_dir = Path("/home/arr65/data/nshm/auto_output/auto14")

if delete_exisiting_output:
    shutil.rmtree(output_dir, ignore_errors=True)

output_dir.mkdir(parents=True, exist_ok=True)

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
#locations = ["AKL","WLG","CHC"]
locations = ["WLG"]

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

print()

count_dict = {}

for branch_set in glt_full.branch_sets:

    count_dict[branch_set.short_name] = {}

    branch_gsim_name_list = []

    for branch in branch_set.branches:

        branch_gsim_name_list.append(branch.gsim_name)

    count_dict[branch_set.short_name] = count_strings(branch_gsim_name_list)


        #print(f"{branch_set.short_name}, {branch.gsim_name})

#["Active Shallow Crust", "Subduction Interface", "Subduction Intraslab"]
# options are "both", "HIK", "PUY"

# slt_crust = logic_tree_tools.select_trt_branch_sets(slt_full, ["Active Shallow Crust"])
# glt_crust = logic_tree_tools.select_trt_branch_sets(glt_full, ["Active Shallow Crust"])

slt_highest_entry_list = logic_tree_tools.get_custom_logic_tree_entry_for_nth_highest_branch(slt_full,1)
glt_highest_entry_list = logic_tree_tools.get_custom_logic_tree_entry_for_nth_highest_branch(glt_full,1)

print()

trt_select_input_entry = logic_tree_tools.CustomLogicTreeSet(
    slt = copy.deepcopy(slt_highest_entry_list[0].slt),
    glt = copy.deepcopy(glt_full),
    slt_note = slt_highest_entry_list[0].slt_note,
    glt_note = 'full > ')

full_full_input_entry = logic_tree_tools.CustomLogicTreeSet(
    slt = copy.deepcopy(slt_full),
    glt = copy.deepcopy(glt_full),
    slt_note = "full > ",
    glt_note = 'full > ')

highest_weighted_input_entry = logic_tree_tools.CustomLogicTreeSet(
    slt = copy.deepcopy(slt_highest_entry_list[0].slt),
    glt = copy.deepcopy(glt_highest_entry_list[0].glt),
    slt_note = slt_highest_entry_list[0].slt_note,
    glt_note = glt_highest_entry_list[0].glt_note)

# available_trts = ["Active Shallow Crust", "Subduction Interface", "Subduction Intraslab"]
# which_interfaces = ["both", "HIK", "PUY"]

available_trts = ["Active Shallow Crust"]
which_interfaces = ["both"]

trt_combinations_to_process = [["Active Shallow Crust"]]

## For getting all ground motion models

logic_tree_list = []

for trt_combination_list in trt_combinations_to_process:

    for trt in trt_combination_list:

        if trt == "Subduction Interface":

            for which_interface in which_interfaces:

                trt_list = [trt_combination_list]

                lt_entry_for_trts = logic_tree_tools.get_trt_set(trt_select_input_entry, trts = trt_list, which_interface=which_interface)

                logic_tree_list.append(lt_entry_for_trts)

        else:

            trt_list = [trt_combination_list]

            lt_entry_for_trts = logic_tree_tools.get_trt_set(trt_select_input_entry, trts=trt_list,
                                                             which_interface=None)

            logic_tree_list.append(lt_entry_for_trts)

print()

logic_tree_list2 = []

all_glt_gsim_names = []

for lt_set in logic_tree_list:

    assert len(lt_set.glt.branch_sets) == 1

    glt_gsim_names = [branch.gsim_name for branch in lt_set.glt.branch_sets[0].branches]
    all_glt_gsim_names.append(glt_gsim_names)

    unique_gsim_names = list(set(glt_gsim_names))

    unique_gsim_names = ['Bradley2013']

    for gsim_name in unique_gsim_names:

        selected_glt_branches = [copy.deepcopy(branch) for branch in lt_set.glt.branch_sets[0].branches if branch.gsim_name == gsim_name]

        selected_glt_branch_weights = np.array([copy.deepcopy(branch.weight) for branch in lt_set.glt.branch_sets[0].branches if
                                 branch.gsim_name == gsim_name])

        needed_scaling_factor = 1.0 / np.sum(selected_glt_branch_weights)

        scaled_weights = selected_glt_branch_weights * needed_scaling_factor

        for i, branch in enumerate(selected_glt_branches):
            branch.weight = scaled_weights[i]

        modified_lt_set = copy.deepcopy(lt_set)

        modified_lt_set.glt.branch_sets[0].branches = selected_glt_branches
        modified_lt_set.glt_note += f"[{gsim_name}*{needed_scaling_factor:.2f}] > "

        logic_tree_list2.append(modified_lt_set)



logic_tree_list = logic_tree_list2

## end code for getting all ground motion models

#logic_tree_list = [full_full_input_entry]

######################################################################

#logic_tree_list = [full_full_input_entry]

#lt_entry_for_trts = logic_tree_tools.get_trt_set(trt_select_input_entry, trts = ["Active Shallow Crust", "Subduction Interface"], which_interface="HIK")

#highest = logic_tree_tools.reduce_to_highest_weighted_branch(glt_full)

#nth_highest_list = logic_tree_tools.get_custom_logic_tree_entry_for_nth_highest_branch(glt_full,list(range(1,6)))

# slt_highest_weighted_branch_list = logic_tree_tools.get_custom_logic_tree_entry_for_nth_highest_branch(slt_full,1)
# glt_highest_weighted_branch_list = logic_tree_tools.get_custom_logic_tree_entry_for_nth_highest_branch(glt_full,1)
#
# slt_single_branch_set_list = logic_tree_tools.combinations_of_n_branch_sets(slt_full, 1)
# glt_single_branch_set_list = logic_tree_tools.combinations_of_n_branch_sets(glt_full, 1)
#
# slt_list = [logic_tree_tools.CustomLogicTreeSet(slt = slt_full, slt_note = "slt full")]
# glt_list = [logic_tree_tools.CustomLogicTreeSet(glt = glt_full, glt_note = "glt full")]

# slt_list = [logic_tree_tools.CustomLogicTreeSet(slt = slt_crust, slt_note = "slt crust")]
# glt_list = [logic_tree_tools.CustomLogicTreeSet(glt = glt_crust, glt_note = "glt crust")]

# print(len(glt_single_branch_set_list))
#
# for i in range(len(glt_single_branch_set_list)):
#
#     print(f"{len(glt_single_branch_set_list[i].glt.branch_sets)} branch sets")
#
#     print(f"{glt_single_branch_set_list[i].glt.branch_sets[0].short_name}")
#
#     print(f"has {len(glt_single_branch_set_list[i].glt.branch_sets[0].branches)} branches")
#
#
#
# print()

#logic_tree_list = combine_logic_tree_combinations(slt_highest_weighted_branch_list, glt_single_branch_set_list)
#logic_tree_list = combine_logic_tree_combinations(slt_highest_weighted_branch_list, glt_highest_weighted_branch_list)

#logic_tree_list = combine_logic_tree_combinations(slt_list, glt_list)
#logic_tree_list = combine_logic_tree_combinations(slt_entry_for_trts, glt_entry_for_trts)

#logic_tree_list = [lt_entry_for_trts]



#logic_tree_list = combine_logic_tree_combinations(slt_single_branch_set_list, glt_list)
#logic_tree_list = combine_logic_tree_combinations(slt_list, glt_single_branch_set_list)

#logic_tree_list = combine_logic_tree_combinations(slt_list, glt_single_branch_set_list)

## Trying the first 5 highest weighted branches from the SRM to see if the selected branch makes any difference

# slt_combinations = logic_tree_tools.get_custom_logic_tree_entry_for_nth_highest_branch(slt_full,list(range(1,6)))
#
# glt_combinations = [
#     logic_tree_tools.CustomLogicTreeSet(
#         glt = glt_full,
#         glt_note = f"GMCM full")
# ]
#
# logic_tree_list = combine_logic_tree_combinations(slt_combinations, glt_combinations)

### End

logic_tree_tools.print_info(logic_tree_list)

run_notes_df = pd.DataFrame()
for run_counter, custom_logic_tree_set in enumerate(logic_tree_list):
    notes_df = custom_logic_tree_set.notes_to_pandas_df()
    notes_df['run_counter'] = [run_counter]
    run_notes_df = pd.concat([run_notes_df, notes_df], ignore_index=True)

# move the "run_counter" column to the left-most position
run_notes_df.insert(0, "run_counter", run_notes_df.pop("run_counter"))
run_notes_df.to_csv(output_dir / "run_notes.csv")

for run_counter, custom_logic_tree_set in enumerate(logic_tree_list):
    run_with_modified_logic_trees(args, output_dir, run_counter, custom_logic_tree_set, locations, output_staging_dir)

end_time = time.time()
print(f"Time taken: {(end_time - start_time)/60} mins for {len(logic_tree_list)} runs")

# def run_with_modified_logic_trees(output_dir, run_counter, custom_logic_tree_set, locations, toml_dict, output_staging_dir):