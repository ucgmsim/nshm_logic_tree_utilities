from toshi_hazard_post.aggregation_args import AggregationArgs
from toshi_hazard_post.aggregation import run_aggregation
from nzshm_model.logic_tree.correlation import LogicTreeCorrelations
import os
import logging

#os.environ['THP_ENV_FILE'] = str(config_file)
os.environ['THP_ENV_FILE'] = str("/home/arr65/src/gns/toshi-hazard-post/scripts/.env_home")

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('toshi_hazard_post').setLevel(logging.INFO)
logging.getLogger('toshi_hazard_post.aggregation_calc').setLevel(logging.DEBUG)
logging.getLogger('toshi_hazard_post.aggregation').setLevel(logging.DEBUG)
logging.getLogger('toshi_hazard_post.aggregation_calc').setLevel(logging.DEBUG)
logging.getLogger('toshi_hazard_post.logic_tree').setLevel(logging.DEBUG)
logging.getLogger('toshi_hazard_post.parallel').setLevel(logging.DEBUG)
logging.getLogger('toshi_hazard_post').setLevel(logging.INFO)

# starting model
#input_file = "demo/hazard_mini.toml"
input_file = "/home/arr65/src/gns/toshi-hazard-post/demo/hazard_mini.toml"
args = AggregationArgs(input_file)

# run model
run_aggregation(args)

# extract logic trees
slt = args.srm_logic_tree
glt = args.gmcm_logic_tree

# modify SRM logic tree
for branch_set in slt.branch_sets:
    print(branch_set.tectonic_region_types)
slt.branch_sets = [slt.branch_sets[0]]
slt.branch_sets[0].branches = [slt.branch_sets[0].branches[0]]

# weights of branches of each branch set must sum to 1.0
slt.branch_sets[0].branches[0].weight = 1.0

# remove correlations
slt.correlations = LogicTreeCorrelations()

# modify GMCM logic tree to match the TRT of the new SRM logic tree
for branch_set in glt.branch_sets:
    print(branch_set.tectonic_region_type)
glt.branch_sets = [glt.branch_sets[1]]

# write logic trees to json for later use
slt.to_json('slt_one_branch.json')
glt.to_json('glt_crust_only.json')

args.srm_logic_tree = slt
args.gmcm_logic_tree = glt
args.hazard_model_id = 'ONE_SRM_BRANCH'
run_aggregation(args)