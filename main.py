import copy
import logging
import os
import shutil
import time
from pathlib import Path

import pandas as pd
from toshi_hazard_post.aggregation_args import (
    AggregationArgs,
)

import config as cfg
import logic_tree_tools
import run_toshi_hazard_post_from_Python

config = cfg.Config()

start_time = time.time()

## copying logging from toshi_hazard_post scripts/cli.py
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logging.getLogger("toshi_hazard_post").setLevel(logging.INFO)
logging.getLogger("toshi_hazard_post.aggregation_calc").setLevel(logging.DEBUG)
logging.getLogger("toshi_hazard_post.aggregation").setLevel(logging.DEBUG)
logging.getLogger("toshi_hazard_post.aggregation_calc").setLevel(logging.DEBUG)
logging.getLogger("toshi_hazard_post.logic_tree").setLevel(logging.DEBUG)
logging.getLogger("toshi_hazard_post.parallel").setLevel(logging.DEBUG)
logging.getLogger("toshi_hazard_post").setLevel(logging.INFO)

input_file_dir = Path(config.get_value("input_directory"))
output_dir = Path(config.get_value("output_directory"))

if config.get_value("delete_exisiting_output"):
    shutil.rmtree(output_dir, ignore_errors=True)
output_dir.mkdir(parents=True, exist_ok=False)

os.environ["THP_ENV_FILE"] = str(input_file_dir / config.get_value("env_file_name"))
initial_input_file = input_file_dir / config.get_value("initial_input_file_name")

with open(input_file_dir / config.get_value("env_file_name"), "r") as file:
    env_lines = file.readlines()

### output_staging_dir is toshi_hazard_post's output directory.
### After toshi_hazard_post has finished, these files are moved to the output_dir
output_staging_dir = Path(env_lines[-1].split("=")[1].strip("\n ' \" "))
if config.get_value("delete_existing_output"):
    shutil.rmtree(output_staging_dir, ignore_errors=True)
    output_staging_dir.mkdir(parents=True, exist_ok=True)

## Define most of the arguments by loading from the initial input file
args = AggregationArgs(initial_input_file)
args.output_individual_realizations = config.get_value("output_individual_realizations")

### An initial pair of source and ground motion logic trees
full_lt_set = logic_tree_tools.CustomLogicTreeSet(
    slt=copy.deepcopy(args.srm_logic_tree),
    glt=copy.deepcopy(args.gmcm_logic_tree),
    slt_note="full > ",
    glt_note="full > ",
)

## This script will iterate over the CustomLogicTreeSet objects in logic_tree_set_list
# and run toshi_hazard_post with each CustomLogicTreeSet object

#### Option 1 ####

### This constructs a logic_tree_set_list of:
## Index 0: The full source logic tree and full ground motion logic tree
## Index 1: Only the highest weighted branch of the source logic tree and the full ground motion logic tree
## Index 2: The full source logic tree and only the highest weighted branch of the ground motion logic tree

logic_tree_set_list = [
    full_lt_set,
    logic_tree_tools.reduce_lt_set_to_nth_highest_branches(
        full_lt_set, slt_nth_highest=1, glt_nth_highest=None
    ),
    logic_tree_tools.reduce_lt_set_to_nth_highest_branches(
        full_lt_set, slt_nth_highest=None, glt_nth_highest=1
    ),
]

#### Option 2 ####

### This constructs a logic_tree_set_list of individual ground motion models,
### paired with the highest weighted branch of the source logic tree

# logic_tree_set_list = logic_tree_tools.get_logic_tree_sets_for_individual_ground_motion_models(
#     initial_logic_tree_set = full_lt_set,
#     tectonic_region_type_sets=[["Active Shallow Crust"],["Subduction Interface"], ["Subduction Intraslab"]],
#     which_interfaces = ["only_HIK", "only_PUY", "HIK_and_PUY"])

#### Option 3 ####

### This constructs a logic_tree_set_list of individual source models,
### paired with the highest weighted branch of the ground motion models logic tree

# logic_tree_set_list = logic_tree_tools.get_logic_tree_sets_for_individual_source_models(
#     initial_logic_tree_set = full_lt_set,
#     tectonic_region_type_sets = [["Active Shallow Crust"], ["Subduction Interface"], ["Subduction Intraslab"]],
#     which_interfaces = ["only_HIK", "only_PUY", "HIK_and_PUY"])

### Print info about the logic trees
logic_tree_tools.print_info_about_logic_tree_sets(logic_tree_set_list)

## Write notes about the modified logic trees
run_notes_df = pd.DataFrame()
for run_counter, custom_logic_tree_set in enumerate(logic_tree_set_list):
    notes_df = custom_logic_tree_set.notes_to_pandas_df()
    notes_df["run_counter"] = [run_counter]
    run_notes_df = pd.concat([run_notes_df, notes_df], ignore_index=True)

# move the "run_counter" column to the left-most position
run_notes_df.insert(0, "run_counter", run_notes_df.pop("run_counter"))
run_notes_df.to_csv(output_dir / config.get_value("run_notes_file_name"))

## Run toshi_hazard_post with the modified logic trees
for run_counter, custom_logic_tree_set in enumerate(logic_tree_set_list):
    run_toshi_hazard_post_from_Python.run_with_modified_logic_trees(
        args,
        output_dir,
        run_counter,
        custom_logic_tree_set,
        config.get_value("locations"),
        output_staging_dir,
    )

end_time = time.time()
print(
    f"Time taken: {(end_time - start_time)/60} mins for {len(logic_tree_set_list)} runs"
)
