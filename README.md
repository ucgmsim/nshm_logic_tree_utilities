### Overview

This package has utilities for working with the logic tree used in [New Zealand's National Seismic Hazard Model 
(NSHM) 2022](https://www.gns.cri.nz/research-projects/national-seismic-hazard-model/). 
With this package, you can create modified logic trees and use them to calculate new hazard curves.

### Installation
1. Clone the repository. One way to do this is to open a terminal and run the following command: 
    
   1. `git clone git@github.com:ucgmsim/nshm_logic_tree_utilities.git`

2. Navigate to the repository directory and create a virtual environment. One way to do this is to run the following 
   commands in the terminal:

   1. `cd nshm_logic_tree_utilities`
   2. `python -m venv .venv`

3. Activate the new virtual environment by running the following command in the terminal:
   1. `source .venv/bin/activate`

4. Install the required packages by running the following command in the terminal (without changing directories):
   1. `pip install -e .`

### Usage

Paths and commands below are relative to the repository root (the outer `nshm_logic_tree_utilities` directory).

1. To modify the standard NZ NSHM 2022 logic tree and generate hazard curves:

   1. Open `nshm_logic_tree_utilities/scripts/run_toshi_hazard_post_config.yaml` and set `output_directory` to the
      directory where you want the results to be saved
   2. Open `nshm_logic_tree_utilities/input_files/.env_home` and set `THP_RLZ_LOCAL_DIR` to your local copy of the
      pre-computed NSHM 2022 realization datastore, and `THP_AGG_LOCAL_DIR` to a directory for toshi_hazard_post's
      intermediate output
   3. Run the code with `python nshm_logic_tree_utilities/scripts/run_toshi_hazard_post_script.py`

   Note that while `overwrite_if_output_directory_exists` is `True` in the config file (as provided), the 
   `output_directory` and `THP_AGG_LOCAL_DIR` directories are deleted at the start of each run, so make sure they 
   don't contain anything you want to keep.

2. To plot the results:
   1. Open `nshm_logic_tree_utilities/scripts/plotting_script.py` and change the directory paths (such as
      `results_directory` and `plot_output_directory`) to your own
   2. Run `python nshm_logic_tree_utilities/scripts/plotting_script.py`

Note that the examples provided in `run_toshi_hazard_post_script.py` and `plotting_script.py`
can be modified for your purposes.

For more background information and example output, please refer to the 
accompanying [wiki](https://github.com/ucgmsim/nshm_logic_tree_utilities/wiki).
