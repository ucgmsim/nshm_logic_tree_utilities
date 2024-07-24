from toshi_hazard_post.aggregation_args import AggregationArgs
from toshi_hazard_post.aggregation import run_aggregation
from nzshm_model.logic_tree.correlation import LogicTreeCorrelations
import os
import logging
import numpy as np
import copy
import matplotlib.pyplot as plt

## copying logging from scripts/cli.py
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('toshi_hazard_post').setLevel(logging.INFO)
logging.getLogger('toshi_hazard_post.aggregation_calc').setLevel(logging.DEBUG)
logging.getLogger('toshi_hazard_post.aggregation').setLevel(logging.DEBUG)
logging.getLogger('toshi_hazard_post.aggregation_calc').setLevel(logging.DEBUG)
logging.getLogger('toshi_hazard_post.logic_tree').setLevel(logging.DEBUG)
logging.getLogger('toshi_hazard_post.parallel').setLevel(logging.DEBUG)
logging.getLogger('toshi_hazard_post').setLevel(logging.INFO)

os.environ['THP_ENV_FILE'] = str("/home/arr65/src/gns/toshi-hazard-post/scripts/.env_home")

# starting model
input_file = "/home/arr65/src/gns/toshi-hazard-post/scripts/test_input.toml"
args = AggregationArgs(input_file)



print()

# run model
run_aggregation(args)

# extract logic trees
slt = args.srm_logic_tree
glt = args.gmcm_logic_tree
