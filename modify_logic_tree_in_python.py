from toshi_hazard_post.aggregation_args import AggregationArgs
from toshi_hazard_post.aggregation import run_aggregation
from nzshm_model.logic_tree.correlation import LogicTreeCorrelations
import os
import logging
import numpy as np
import copy

def reduce_to_highest_weighted_branch(logic_tree):

    modified_logic_tree = copy.deepcopy(logic_tree)

    for branch_set_idx, branch_set in enumerate(logic_tree.branch_sets):

        highest_weight = 0.0

        for branch in branch_set.branches:

            if branch.weight > highest_weight:
                highest_weight_branch = branch
                highest_weight = branch.weight

        highest_weight_branch.weight = 1.0
        modified_logic_tree.branch_sets[branch_set_idx].branches = [highest_weight_branch]

    modified_logic_tree.correlations = LogicTreeCorrelations()

    return modified_logic_tree


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

# run model
run_aggregation(args)

# extract logic trees
slt = args.srm_logic_tree
glt = args.gmcm_logic_tree

slt_only_highest_weight_branches = reduce_to_highest_weighted_branch(slt)
slt_only_highest_weight_branches.to_json('/home/arr65/src/logic_tree_study/custom_logic_trees/slt_only_highest_weighted_branches.json')

glt_only_highest_weight_branches = reduce_to_highest_weighted_branch(glt)
glt_only_highest_weight_branches.to_json('/home/arr65/src/logic_tree_study/custom_logic_trees/glt_only_highest_weighted_branches.json')


# print()
#
# slt2 = copy.deepcopy(slt)
# glt2 = copy.deepcopy(glt)
#
# print()
#
#
#
# branch_weights = []
#
# test_lt = copy.deepcopy(glt)
# set_idx = 3
#
# for branch in test_lt.branch_sets[set_idx].branches:
#     branch_weights.append(branch.weight)
# branch_weights = np.array(branch_weights)
# highest_weighted_branch = test_lt.branch_sets[set_idx].branches[np.argmax(branch_weights)]
#
# gl3.branch_sets[set_idx].branches[0]


# highest_weighted_branch_slt = reduce_to_highest_weighted_branch(slt)
# highest_weighted_branch_glt = reduce_to_highest_weighted_branch(glt)

# for branch_set_idx, branch_set in enumerate(slt2.branch_sets):
#
#     branch_weights = []
#
#     for branch_idx, branch in enumerate(branch_set.branches):
#
#         branch_weights.append(branch.weight)
#
#     branch_weights = np.array(branch_weights)
#     highest_weighted_branch = branch_set.branches[np.argmax(branch_weights)]
#     highest_weighted_branch.weight = 1.0
#
#     slt2.branch_sets[branch_set_idx] = [highest_weighted_branch]
#
# slt2.correlations = LogicTreeCorrelations()

# for branch_set_idx, branch_set in enumerate(slt2.branch_sets):
#
#     branch_weights = []
#
#     for branch_idx, branch in enumerate(branch_set.branches):
#
#         branch_weights.append(branch.weight)
#
#     branch_weights = np.array(branch_weights)
#     highest_weighted_branch = branch_set.branches[np.argmax(branch_weights)]
#     highest_weighted_branch.weight = 1.0
#
#     slt2.branch_sets[branch_set_idx] = [highest_weighted_branch]
#
# slt2.correlations = LogicTreeCorrelations()

# for set_idx in range(len(slt2.branch_sets)):
#     highest_weight_branch_idx = 0
#     highest_weight = 0.0
#     for branch_idx in range(len(slt2.branch_sets[set_idx].branches)):
#         if slt2.branch_sets[set_idx].branches[branch_idx].weight > highest_weight:
#             highest_weight_branch_idx = branch_idx
#
#     slt2.branch_sets[set_idx].branches = [slt2.branch_sets[set_idx].branches[highest_weight_branch_idx]]
#     slt2.branch_sets[set_idx].branches[0].weight = 1.0


# def reduce_to_highest_weighted_branch(logic_tree):
#
#     modified_logic_tree = copy.copy(logic_tree)
#
#     for set_idx in range(len(logic_tree.branch_sets)):
#         highest_weight_branch_idx = 0
#         highest_weight = 0.0
#         for branch_idx in range(len(logic_tree.branch_sets[set_idx].branches)):
#             if logic_tree.branch_sets[set_idx].branches[branch_idx].weight > highest_weight:
#                 highest_weight_branch_idx = branch_idx
#                 logic_tree.branch_sets[set_idx].branches[branch_idx].weight
#
#         modified_logic_tree.branch_sets[set_idx].branches = [logic_tree.branch_sets[set_idx].branches[highest_weight_branch_idx]]
#
#         # The weight of all branches in a branch_set must sum to 1.0.
#         # As we haved removed all branches except the highest weighted branch,
#         # we set its weight to 1.0
#         modified_logic_tree.branch_sets[set_idx].branches[0].weight = 1.0
#
#     # remove correlations
#     modified_logic_tree.correlations = LogicTreeCorrelations()

    # return modified_logic_tree





# def get_highest_weighted_branch(branch_set: list):
#
#     branch_weights = []
#     for branch in branch_set.branches:
#         branch_weights.append(branch.weight)
#     branch_weights = np.array(branch_weights)
#     return np.argmax(branch_weights)



##################################################################################
################################################
#### Example code below


# # write logic trees to json for later use
# slt.to_json('/home/arr65/src/logic_tree_study/custom_logic_trees/slt_full_NSHM_v1.0.4.json')
# glt.to_json('/home/arr65/src/logic_tree_study/custom_logic_trees/glt_full_NSHM_v1.0.4.json')

# print("slt")
# print(slt)
# print()
# print("glt")
# print(glt)

# print()
#
# # modify SRM logic tree
# for branch_set in slt.branch_sets:
#     print(branch_set.tectonic_region_types)
# slt.branch_sets = [slt.branch_sets[0]]
# slt.branch_sets[0].branches = [slt.branch_sets[0].branches[0]]
#
# # weights of branches of each branch set must sum to 1.0
# slt.branch_sets[0].branches[0].weight = 1.0
#
# # remove correlations
# slt.correlations = LogicTreeCorrelations()
#
# # modify GMCM logic tree to match the TRT of the new SRM logic tree
# for branch_set in glt.branch_sets:
#     print(branch_set.tectonic_region_type)
# glt.branch_sets = [glt.branch_sets[1]]
#
# # write logic trees to json for later use
# slt.to_json('slt_one_branch.json')
# glt.to_json('glt_crust_only.json')
#
# args.srm_logic_tree = slt
# args.gmcm_logic_tree = glt
# args.hazard_model_id = 'ONE_SRM_BRANCH'
# run_aggregation(args)