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

1. To modify the standard NZ NSHM 2022 logic tree and generate hazard curves:

   1. Go to the `nshm_logic_tree_utilities/scripts` directory (where this `nshm_logic_tree_utilities` directory is inside the first)
   2. Open `nshm_logic_tree_utilities/run_toshi_hazard_post_config.yaml`
   3. Modify the line containing `output_directory : "/home/arr65/data/nshm/output"` to use your directory
   4. Run the code with `python nshm_logic_tree_utilities/scripts/run_toshi_hazard_post_script.py`

2. To plot the results:
   1. Run `python nshm_logic_tree_utilities/scripts/plotting_script.py`

Note that the examples provided in `run_toshi_hazard_post_script.py` and `plotting_script.py`
can be modified for your purposes.

For more background information and example output, please refer to the 
accompanying [wiki](https://github.com/ucgmsim/nshm_logic_tree_utilities/wiki).
