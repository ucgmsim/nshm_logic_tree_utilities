"""
This script modifies the logic tree used in the 2022 revision of New Zealand's National Seismic Hazard Model (NZNSHM)
and generates hazard curves using the modified logic trees.
"""

import copy
import logging
import os
import shutil
import time
from pathlib import Path

import pandas as pd
from toshi_hazard_post.aggregation_args import AggregationArgs

from nshm_logic_tree_utilities.lib import config as cfg
from nshm_logic_tree_utilities.lib import logic_tree_tools as logic_tree_tools
from nshm_logic_tree_utilities.lib import param_options as param_options
from nshm_logic_tree_utilities.lib import (
    run_toshi_hazard_post_utilities as run_toshi_hazard_post_utilities,
)

config = cfg.Config()

start_time = time.time()

## copied logging code from toshi_hazard_post scripts/cli.py
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

if config.get_value("overwrite_if_output_directory_exists"):
    shutil.rmtree(output_dir, ignore_errors=True)
output_dir.mkdir(parents=True, exist_ok=False)

os.environ["THP_ENV_FILE"] = str(input_file_dir / config.get_value("env_file_name"))
initial_input_file = input_file_dir / config.get_value("initial_input_file_name")

with open(input_file_dir / config.get_value("env_file_name"), "r") as file:
    env_lines = file.readlines()

### output_staging_dir is toshi_hazard_post's output directory.
### After toshi_hazard_post has finished, these files are moved to the output_dir
output_staging_dir = Path(env_lines[-1].split("=")[1].strip("\n ' \" "))
if config.get_value("overwrite_if_output_directory_exists"):
    shutil.rmtree(output_staging_dir, ignore_errors=True)
    output_staging_dir.mkdir(parents=True, exist_ok=True)

## Define most of the arguments by loading from the initial input file
args = AggregationArgs(initial_input_file)
args.output_individual_realizations = config.get_value("output_individual_realizations")

### An initial pair of source and ground motion logic trees
full_logic_tree_pair = logic_tree_tools.CustomLogicTreePair(
    source_logic_tree=copy.deepcopy(args.srm_logic_tree),
    ground_motion_logic_tree=copy.deepcopy(args.gmcm_logic_tree),
    source_logic_tree_note="full > ",
    ground_motion_logic_tree_note="full > ",
)

## This script will iterate over the CustomLogicTreePair objects in logic_tree_pair_list
# and run toshi_hazard_post with each CustomLogicTreePair object.
# some examples are shown below.

#### Example 1 ####

### This constructs a logic_tree_pair_list of:
## Index 0: The full source logic tree and full ground motion logic tree
## Index 1: Only the highest weighted branch of the source logic tree and the full ground motion logic tree
## Index 2: The full source logic tree and only the highest weighted branch of the ground motion logic tree

logic_tree_pair_list1 = [
    full_logic_tree_pair,
    logic_tree_tools.reduce_logic_tree_pair_to_nth_highest_branches(
        full_logic_tree_pair,
        source_logic_tree_nth_highest=1,
        ground_motion_logic_tree_nth_highest=None,
    ),
    logic_tree_tools.reduce_logic_tree_pair_to_nth_highest_branches(
        full_logic_tree_pair,
        source_logic_tree_nth_highest=None,
        ground_motion_logic_tree_nth_highest=1,
    ),
]

#### Example 2 ####

### This constructs a logic_tree_pair_list of individual ground motion models,
### paired with the highest weighted branch of the source logic tree

logic_tree_pair_list2 = (
    logic_tree_tools.get_logic_tree_pairs_for_individual_ground_motion_models(
        initial_logic_tree_pair=full_logic_tree_pair,
        tectonic_region_type_groups=[
            [param_options.TectonicRegionTypeName.Active_Shallow_Crust],
            [param_options.TectonicRegionTypeName.Subduction_Interface],
            [param_options.TectonicRegionTypeName.Subduction_Intraslab],
        ],
        which_interfaces=[
            param_options.InterfaceName.only_HIK,
            param_options.InterfaceName.only_PUY,
            param_options.InterfaceName.HIK_and_PUY,
        ],
    )
)

#### Example 3 ####

### This constructs a logic_tree_pair_list of individual source models,
### paired with the highest weighted branch of the ground motion models logic tree

logic_tree_pair_list3 = (
    logic_tree_tools.get_logic_tree_pairs_for_individual_source_models(
        initial_logic_tree_pair=full_logic_tree_pair,
        tectonic_region_type_groups=[
            [param_options.TectonicRegionTypeName.Active_Shallow_Crust],
            [param_options.TectonicRegionTypeName.Subduction_Interface],
            [param_options.TectonicRegionTypeName.Subduction_Intraslab],
        ],
        which_interfaces=[
            param_options.InterfaceName.only_HIK,
            param_options.InterfaceName.only_PUY,
            param_options.InterfaceName.HIK_and_PUY,
        ],
    )
)

### concatenate the logic_tree_pair_lists
logic_tree_pair_list = (
    logic_tree_pair_list1 + logic_tree_pair_list2 + logic_tree_pair_list3
)

### Print info about the logic trees
logic_tree_tools.print_info_about_logic_tree_pairs(logic_tree_pair_list)

## Write notes about the modified logic trees
collated_notes_df = pd.DataFrame()
for logic_tree_index, custom_logic_tree_pair in enumerate(logic_tree_pair_list):
    notes_df = custom_logic_tree_pair.notes_to_pandas_df()
    notes_df["logic_tree_index"] = [logic_tree_index]
    collated_notes_df = pd.concat([collated_notes_df, notes_df], ignore_index=True)

# move the "logic_tree_index" column to the left-most position
collated_notes_df.insert(
    0, "logic_tree_index", collated_notes_df.pop("logic_tree_index")
)
collated_notes_df.to_csv(output_dir / config.get_value("collated_notes_file_name"))
## Run toshi_hazard_post with the modified logic trees
for logic_tree_index, custom_logic_tree_pair in enumerate(logic_tree_pair_list):
    run_toshi_hazard_post_utilities.run_with_modified_logic_trees(
        args,
        output_dir,
        logic_tree_index,
        custom_logic_tree_pair,
        config.get_value("locations"),
        output_staging_dir,
    )

end_time = time.time()
print(
    f"Time taken: {(end_time - start_time)/60} mins for {len(logic_tree_pair_list)} logic trees"
)
