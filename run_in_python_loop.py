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
from dataclasses import dataclass
from nzshm_model.logic_tree import GMCMLogicTree, SourceBranchSet, SourceLogicTree
from typing import Optional
import pandas as pd
import time



def run_with_modified_logic_trees(output_dir, run_counter, custom_logic_tree_set, locations, toml_dict, output_staging_dir):

    run_start_time = time.time()

    #run_group_name = output_dir.name

    modified_slt = copy.deepcopy(custom_logic_tree_set.slt)
    modified_glt = copy.deepcopy(custom_logic_tree_set.glt)

    # check the validity of the weights
    modify_logic_tree_in_python.check_weight_validity(custom_logic_tree_set.slt)
    modify_logic_tree_in_python.check_weight_validity(custom_logic_tree_set.glt)

    print()

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

    run_output_dir = output_dir / f"run_{run_counter}"
    run_output_dir.mkdir(parents=True, exist_ok=True)

    for file in output_staging_dir.iterdir():
        shutil.move(file, run_output_dir)

    run_end_time = time.time()
    print(f"Time taken for run {run_counter}: {(run_end_time - run_start_time)} mins")

def make_logic_tree_combinations_list_branch_sets(full_logic_tree, logic_tree_highest_weighted_branches):
    #from nzshm_model.logic_tree import GMCMLogicTree, SourceBranchSet, SourceLogicTree

    logic_tree_permutation_list = []

    for branch_set_index, branch_set in enumerate(full_logic_tree.branch_sets):

        modified_logic_tree = copy.deepcopy(full_logic_tree)

        modified_logic_tree.branch_sets[branch_set_index] = logic_tree_highest_weighted_branches.branch_sets[branch_set_index]
        modified_logic_tree.correlations = LogicTreeCorrelations()

        if isinstance(full_logic_tree, SourceLogicTree):
            custom_logic_tree_entry = modify_logic_tree_in_python.CustomLogicTreeSet(slt = modified_logic_tree,
                        slt_note = f"branch_set {branch_set.long_name} ({branch_set.short_name}) reduced to its single highest weighted branch. No other changes.")

        elif isinstance(full_logic_tree, GMCMLogicTree):
            custom_logic_tree_entry = modify_logic_tree_in_python.CustomLogicTreeSet(glt = modified_logic_tree,
                         glt_note = f"branch_set {branch_set.long_name} ({branch_set.short_name}) reduced to its single highest weighted branch. No other changes.")

        logic_tree_permutation_list.append(custom_logic_tree_entry)

    return logic_tree_permutation_list

def combine_logic_tree_combinations(slt_permutations, glt_permutations):

    combined_permutations = []

    for custom_slt_entry in slt_permutations:

        for custom_glt_entry in glt_permutations:

            slt_glt_entry = modify_logic_tree_in_python.CustomLogicTreeSet(slt=custom_slt_entry.slt,
                                               slt_note=custom_slt_entry.slt_note,
                                               glt=custom_glt_entry.glt,
                                               glt_note=custom_glt_entry.glt_note)

            combined_permutations.append(slt_glt_entry)

    # check that all required parameters are present
    check_validity_of_combinations(combined_permutations)
    return combined_permutations

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

#os.environ['THP_ENV_FILE'] = str("/home/arr65/src/gns/toshi-hazard-post/scripts/.env_home")

toshi_hazard_post_scripts_dir = Path("/home/arr65/src/gns/toshi-hazard-post/scripts")

output_dir = Path(f"/home/arr65/data/nshm/auto_output/auto5")
output_dir.mkdir(parents=True, exist_ok=True)

initial_input_file = toshi_hazard_post_scripts_dir / "simple_input.toml"
if not initial_input_file.exists():
    shutil.copy("custom_input_files/simple_input.toml", initial_input_file)

with open(toshi_hazard_post_scripts_dir / ".env_home", 'r') as file:
    env_lines = file.readlines()

output_staging_dir = Path(env_lines[-1].split('=')[1].strip("\n \' \" "))

toml_dict = toml.load(initial_input_file)

# All locations can be specified in the same input file but this uses more memory than doing one location at a time
#locations = ["AKL","WLG","CHC"]
locations = ["WLG"]

args = AggregationArgs(initial_input_file)

slt_full = args.srm_logic_tree
glt_full = args.gmcm_logic_tree

glt_full.to_json("/home/arr65/data/nshm/auto_output/glt_full.json")

print()

#slt_comb = modify_logic_tree_in_python.combinations_of_n_branch_sets(slt_full, 1)

slt_comb = [modify_logic_tree_in_python.CustomLogicTreeSet(
    slt = modify_logic_tree_in_python.reduce_to_nth_highest_weighted_branch(
        logic_tree = slt_full,
        nth_highest = 1),
    slt_note = "SRM h.w.b.")]

glt_comb = modify_logic_tree_in_python.combinations_of_n_branch_sets(glt_full, 1)

logic_tree_list = combine_logic_tree_combinations(slt_comb, glt_comb)

print()

# slt_highest_weighted_branch = modify_logic_tree_in_python.reduce_to_highest_weighted_branch(slt_full)
# glt_highest_weighted_branch = modify_logic_tree_in_python.reduce_to_highest_weighted_branch(glt_full)


# slt_perm = make_logic_tree_combinations_list_branch_sets(slt_full, slt_highest_weighted_branch)
# glt_perm = make_logic_tree_combinations_list_branch_sets(glt_full, glt_highest_weighted_branch)
#
# slt_full_and_highest = [modify_logic_tree_in_python.CustomLogicTreeSet(slt = slt_full,
#                                            slt_note = "Full SRM logic tree."),
#                         modify_logic_tree_in_python.CustomLogicTreeSet(slt = slt_highest_weighted_branch,
#                                              slt_note = "SRM logic tree reduced to its single highest weighted branch. No other changes.")]
#
# glt_full_and_highest = [modify_logic_tree_in_python.CustomLogicTreeSet(glt = glt_full,
#                                            glt_note = "Full GMCM logic tree."),
#                         modify_logic_tree_in_python.CustomLogicTreeSet(glt = glt_highest_weighted_branch,
#                                            glt_note = "GMCM logic tree reduced to its single highest weighted branch. No other changes.")]


## Trying the first 5 highest weighted branches from the SRM to see if the selected branch makes any difference

# slt_combinations = [
#     modify_logic_tree_in_python.CustomLogicTreeSet(
#         slt = modify_logic_tree_in_python.reduce_to_nth_highest_weighted_branch(logic_tree = slt_full, nth_highest = n),
#         slt_note = f"SRM {n}th h.w.b.")
#     for n in range(1, 6)
# ]
#
# glt_combinations = [
#     modify_logic_tree_in_python.CustomLogicTreeSet(
#         glt = glt_full,
#         glt_note = f"GMCM full")
# ]
#
# logic_tree_list = combine_logic_tree_combinations(slt_combinations, glt_combinations)

### End





run_notes_df = pd.DataFrame()
for run_counter, custom_logic_tree_set in enumerate(logic_tree_list):
    notes_df = custom_logic_tree_set.notes_to_pandas_df()
    notes_df['run_counter'] = [run_counter]
    run_notes_df = pd.concat([run_notes_df, notes_df], ignore_index=True)

# move the "run_counter" column to the left-most position
run_notes_df.insert(0, "run_counter", run_notes_df.pop("run_counter"))
run_notes_df.to_csv(output_dir / "run_notes.csv")

os.chdir(toshi_hazard_post_scripts_dir)
for run_counter, custom_logic_tree_set in enumerate(logic_tree_list):
    run_with_modified_logic_trees(output_dir, run_counter, custom_logic_tree_set, locations, toml_dict, output_staging_dir)

end_time = time.time()
print(f"Time taken: {(end_time - start_time)} mins for {len(logic_tree_list)} runs")