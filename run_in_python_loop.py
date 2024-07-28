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

run_group_name = "auto1"

working_dir = Path("/home/arr65/src/gns/toshi-hazard-post/scripts")
os.chdir(working_dir)


with open("/home/arr65/src/gns/toshi-hazard-post/scripts/.env_home", 'r') as file:
    env_lines = file.readlines()


staging_output_dir = Path(env_lines[-1].split('=')[1].strip("\n \' \" "))

#input_file = Path("/home/arr65/src/gns/toshi-hazard-post/scripts/test_input.toml")
initial_input_file = Path("/home/arr65/src/gns/toshi-hazard-post/scripts/test_input.toml")
temp_input_file = Path("/home/arr65/src/gns/toshi-hazard-post/scripts/temp_input.toml")


toml_dict = toml.load(initial_input_file)


# All locations can be specified in the same input file but this uses more memory than doing one location at a time
#locations = ["AKL","WLG","CHC"]
locations = ["CHC"]

#

num_runs_to_do = 4

args = AggregationArgs(initial_input_file)

slt_full = args.srm_logic_tree
glt_full = args.gmcm_logic_tree

slt_full_copy = copy.deepcopy(slt_full)
glt_full_copy = copy.deepcopy(glt_full)

slt_highest_weighted_branch = modify_logic_tree_in_python.reduce_to_highest_weighted_branch(slt_full_copy)
glt_highest_weighted_branch = modify_logic_tree_in_python.reduce_to_highest_weighted_branch(glt_full_copy)

print()

for run_counter in range(num_runs_to_do):

    if run_counter == 0:
        modified_slt = copy.deepcopy(slt_full)
        modified_glt = copy.deepcopy(glt_full)

    if run_counter == 1:
        modified_slt = copy.deepcopy(slt_full)
        modified_glt = copy.deepcopy(glt_highest_weighted_branch)

    if run_counter == 2:
        modified_slt = copy.deepcopy(slt_highest_weighted_branch)
        modified_glt = copy.deepcopy(glt_full)

    if run_counter == 3:
        modified_slt = copy.deepcopy(slt_highest_weighted_branch)
        modified_glt = copy.deepcopy(glt_highest_weighted_branch)

    # check the validity of the weights
    modify_logic_tree_in_python.check_weight_validity(modified_slt)
    modify_logic_tree_in_python.check_weight_validity(modified_glt)

    modified_slt.to_json(staging_output_dir / f"slt_{run_counter}.json")
    modified_glt.to_json(staging_output_dir / f"glt_{run_counter}.json")

    for location in locations:
        print(f'doing run {run_counter} and location {location}')

        toml_dict['site']['locations'] = [location]
        toml_dict["general"]["hazard_model_id"] = f'run_{run_counter}'
        toml_dict["logic_trees"]["srm_file"] = staging_output_dir / f"slt_{run_counter}.json"
        toml_dict["logic_trees"]["gmcm_file"] = staging_output_dir / f"glt_{run_counter}.json"

        with open((temp_input_file) , "w") as f:
            toml.dump(toml_dict, f)

        result = subprocess.run("python cli.py aggregate --config-file .env_home test_input.toml",
                                shell=True, capture_output=True, text=True)

    output_staging_dir = Path("/home/arr65/data/nshm/nshm_output_staging")

    run_output_dir = Path(f"/home/arr65/data/nshm/auto_output/{run_group_name}/run_{run_counter}")
    run_output_dir.mkdir(parents=True, exist_ok=True)

    for file in output_staging_dir.iterdir():
        shutil.move(file, run_output_dir)


# run model


# extract logic trees
# slt = args.srm_logic_tree
# glt = args.gmcm_logic_tree
