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
import modify_logic_tree_in_python


def run_with_modified_logic_trees(run_group_name, run_counter, slt, glt, locations, toml_dict, temp_input_file, staging_output_dir):

    modified_slt = copy.deepcopy(slt)
    modified_glt = copy.deepcopy(glt)

    # check the validity of the weights
    modify_logic_tree_in_python.check_weight_validity(slt)
    modify_logic_tree_in_python.check_weight_validity(glt)

    modified_slt.to_json(staging_output_dir / f"slt_{run_counter}.json")
    modified_glt.to_json(staging_output_dir / f"glt_{run_counter}.json")

    for location in locations:
        print(f'doing run {run_counter} and location {location}')

        toml_dict['site']['locations'] = [location]
        toml_dict["general"]["hazard_model_id"] = f'run_{run_counter}'
        toml_dict["logic_trees"]["srm_file"] = str(staging_output_dir / f"slt_{run_counter}.json")
        toml_dict["logic_trees"]["gmcm_file"] = str(staging_output_dir / f"glt_{run_counter}.json")

        with open((temp_input_file), "w") as f:
            toml.dump(toml_dict, f)

        result = subprocess.run("python cli.py aggregate --config-file .env_home temp_input.toml",
                                shell=True, capture_output=True, text=True)

    output_staging_dir = Path("/home/arr65/data/nshm/nshm_output_staging")

    run_output_dir = Path(f"/home/arr65/data/nshm/auto_output/{run_group_name}/run_{run_counter}")
    run_output_dir.mkdir(parents=True, exist_ok=True)

    for file in output_staging_dir.iterdir():
        shutil.move(file, run_output_dir)

def make_logic_tree_permutation_list_branch_sets(full_logic_tree, logic_tree_highest_weighted_branches):

    logic_tree_permutation_list = []

    for branch_set_index, branch_set in enumerate(full_logic_tree.branch_sets):

        modified_logic_tree = copy.deepcopy(full_logic_tree)

        modified_logic_tree.branch_sets[branch_set_index] = logic_tree_highest_weighted_branches.branch_sets[branch_set_index]
        modified_logic_tree.correlations = LogicTreeCorrelations()

        logic_tree_permutation_list.append(modified_logic_tree)

    return logic_tree_permutation_list

def combine_logic_tree_permutations(slt_permutations, glt_permutations):

    combined_permutations = []

    for slt in slt_permutations:
        for glt in glt_permutations:
            combined_permutations.append([slt, glt])

    return combined_permutations

def transpose_lists(lists):
    # Use zip to combine the lists element-wise and convert to a list of lists
    transposed = list(map(list, zip(*lists)))
    return transposed

## copying logging from scripts/cli.py
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('toshi_hazard_post').setLevel(logging.INFO)
logging.getLogger('toshi_hazard_post.aggregation_calc').setLevel(logging.DEBUG)
logging.getLogger('toshi_hazard_post.aggregation').setLevel(logging.DEBUG)
logging.getLogger('toshi_hazard_post.aggregation_calc').setLevel(logging.DEBUG)
logging.getLogger('toshi_hazard_post.logic_tree').setLevel(logging.DEBUG)
logging.getLogger('toshi_hazard_post.parallel').setLevel(logging.DEBUG)
logging.getLogger('toshi_hazard_post').setLevel(logging.INFO)

#os.environ['THP_ENV_FILE'] = str("/home/arr65/src/gns/toshi-hazard-post/scripts/.env_home")

working_dir = Path("/home/arr65/src/gns/toshi-hazard-post/scripts")
os.chdir(working_dir)

run_group_name = "auto2"

with open("/home/arr65/src/gns/toshi-hazard-post/scripts/.env_home", 'r') as file:
    env_lines = file.readlines()

staging_output_dir = Path(env_lines[-1].split('=')[1].strip("\n \' \" "))

initial_input_file = Path("/home/arr65/src/gns/toshi-hazard-post/scripts/simple_input.toml")
temp_input_file = Path("/home/arr65/src/gns/toshi-hazard-post/scripts/temp_input.toml")

toml_dict = toml.load(initial_input_file)

# All locations can be specified in the same input file but this uses more memory than doing one location at a time
locations = ["AKL","WLG","CHC"]
#locations = ["WLG"]

args = AggregationArgs(initial_input_file)

slt_full = args.srm_logic_tree
glt_full = args.gmcm_logic_tree

slt_full_copy = copy.deepcopy(slt_full)
glt_full_copy = copy.deepcopy(glt_full)

values_dict = {}


# for branch_set in slt_full.branch_sets:
#
#     for branch in branch_set.branches:
#

for branch_set_index, branch_set in enumerate(slt_full.branch_sets):

    values_list = []

    for branch_index, branch in enumerate(branch_set.branches):
        values_list.append(branch.values)

    values_dict[branch_set_index] = values_list

transpose_dict = copy.deepcopy(values_dict)

for key, value in values_dict.items():
    transpose_dict[key] = transpose_lists(value)


unique_values_dict = copy.deepcopy(transpose_dict)

print()

for branch_set_index, list_of_branch_values in transpose_dict.items():

    print()

    for value_idx, values in enumerate(list_of_branch_values):

        print(value_idx, values)

        unique_values_dict[branch_set_index][value_idx] = list(set(strvalues))



print()













print()








####




slt_highest_weighted_branch = modify_logic_tree_in_python.reduce_to_highest_weighted_branch(slt_full_copy)
glt_highest_weighted_branch = modify_logic_tree_in_python.reduce_to_highest_weighted_branch(glt_full_copy)


logic_tree_list = [
    [slt_full, glt_full],
    [slt_highest_weighted_branch, glt_full],
    [slt_full, glt_highest_weighted_branch],
    [slt_highest_weighted_branch, glt_highest_weighted_branch]
    ]


slt_perm = make_logic_tree_permutation_list_branch_sets(slt_full, slt_highest_weighted_branch)
glt_perm = make_logic_tree_permutation_list_branch_sets(glt_full, glt_highest_weighted_branch)

logic_tree_list.extend(combine_logic_tree_permutations(
    make_logic_tree_permutation_list_branch_sets(slt_full, slt_highest_weighted_branch),
    make_logic_tree_permutation_list_branch_sets(glt_full, glt_highest_weighted_branch)
))

for run_counter, [slt, glt] in enumerate(logic_tree_list):
    run_with_modified_logic_trees(run_group_name, run_counter, slt, glt, locations, toml_dict, temp_input_file, staging_output_dir)
