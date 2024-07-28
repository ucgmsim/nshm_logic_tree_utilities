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

def select_source_branch_sets(logic_tree, branch_set_short_names_to_select):

    modified_logic_tree = copy.deepcopy(logic_tree)

    available_branch_set_short_names = [branch_set.short_name for branch_set in logic_tree.branch_sets]

    if not set(branch_set_short_names_to_select).issubset(set(available_branch_set_short_names)):
        # find which branch set short names are not found in the logic tree
        branch_set_short_names_not_found = set(branch_set_short_names_to_select) - set(available_branch_set_short_names)
        raise ValueError(f"Branch set short names {branch_set_short_names_not_found} are not found in logic tree")

    selected_branch_sets = [ branch_set for branch_set in logic_tree.branch_sets if branch_set.short_name in branch_set_short_names_to_select ]

    modified_logic_tree.branch_sets = selected_branch_sets
    
    if ("PUY" in branch_set_short_names_to_select) & ("HIK" in branch_set_short_names_to_select):
        # retain the HIK to PUY correlations
        pass
    else:
        # remove correlations
        modified_logic_tree.correlations = LogicTreeCorrelations()

    return modified_logic_tree


def logic_tree_single_source_type(source_logic_tree, selected_source_type: str):
    if selected_source_type not in ['distributed', 'inversion']:
        raise ValueError(f"source_type must be either 'distributed' or 'inversion'")

    modified_source_logic_tree = copy.deepcopy(source_logic_tree)

    for branch_set_idx, branch_set in enumerate(slt.branch_sets):
        for branch_idx, branch in enumerate(branch_set.branches):

            if len(branch.sources) > 1:

                for source_index, source in enumerate(branch.sources):
                    if source.type != selected_source_type:
                        del modified_source_logic_tree.branch_sets[branch_set_idx].branches[branch_idx].sources[source_index]

            # if there is only one source in the branch (so removing it would leave no sources)
            if len(branch.sources) == 1:

                # if the only source type is the selected source, leave it as is
                if branch.sources[0].type == selected_source_type:
                    pass

                # if the only source type is not the selected source type, see if there are other branches in the branch_set
                if branch.sources[0].type != selected_source_type:

                    # if the branch_set has no other branches, remove the branch_set
                    if len(branch_set.branches) == 1:
                        del modified_source_logic_tree.branch_sets[branch_set_idx]

                    # if the branch_set has other branches, remove the branch (but leave the rest of the branch_set)
                    if len(branch_set.branches) > 1:
                        del modified_source_logic_tree.branch_sets[branch_set_idx].branches[branch_idx]

    return modified_source_logic_tree

# def sum_weights_in_all_branch_sets(logic_tree):




if __name__ == "__main__":

    # import matplotlib.pyplot as plt

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
    input_file = "/home/arr65/src/gns/toshi-hazard-post/scripts/simple_input.toml"
    args = AggregationArgs(input_file)

    # run model
    #run_aggregation(args)

    # extract logic trees
    slt = args.srm_logic_tree
    glt = args.gmcm_logic_tree

    slt_copy = copy.deepcopy(slt)
    glt_copy = copy.deepcopy(glt)

    slt_copy2 = copy.deepcopy(slt)
    glt_copy2 = copy.deepcopy(glt)


    slt_hw = reduce_to_highest_weighted_branch(slt_copy)
    glt_hw = reduce_to_highest_weighted_branch(glt_copy)

    logic_tree = copy.deepcopy(slt_copy)
    # branch_set_summed_weights = []
    print()
    # for branch_set in logic_tree.branch_sets:
    #
    #     branch_set_running_sum = 0.0
    #
    #     for branch in branch_set.branches:
    #         branch_set_running_sum += branch.weight
    #
    #         print(branch_set)
    #         print(branch)
    #         print(f'{branch.branch_id} {branch.weight} {branch_set_running_sum}')
    #         print()
    #
    #     branch_set_summed_weights.append(branch_set_running_sum)

    branch_set_
    for branch_set_index in range(len(slt.branch_sets)):

        branch_set_running_sum = 0.0

        for branch_index in range(len(slt.branch_sets[branch_set_index].branches)):
            branch_set_running_sum +=




print()
    # bs = slt.branch_sets[2]


    # Confirming that all branch weights within a branch set need to sum to 1.0
    # bw = []
    # for branch_set in glt.branch_sets:
    #     bsw = 0.0
    #     for branch in branch_set.branches:
    #         bsw += branch.weight
    #     bw.append(bsw)

    # slt_reduced_to_highest_weighted = reduce_to_highest_weighted_branch(slt)


# slt_selected_branch_sets = select_source_branch_sets(logic_tree=source_logic_tree, branch_set_short_names_to_select = ['HIK', 'PUY', 'SLAB'])
# slt_selected_branch_sets.to_json('/home/arr65/src/nshm_logic_tree_utilities/custom_logic_trees/slt_HIK_PUY_SLAB.json')

# slt_selected_branch_sets = select_source_branch_sets(logic_tree=source_logic_tree, branch_set_short_names_to_select = ['CRU'])
# slt_selected_branch_sets.to_json('/home/arr65/src/nshm_logic_tree_utilities/custom_logic_trees/slt_CRU.json')


# selected_source_type = 'inversion' # 'distributed' or 'inversion'


# slt_only_selected_source_type = logic_tree_single_source_type(source_logic_tree = source_logic_tree, selected_source_type = selected_source_type)

#slt.to_json('/home/arr65/src/nshm_logic_tree_utilities/custom_logic_trees/test.json')
#print()

# slt_only_selected_source_type.to_json(f'/home/arr65/src/nshm_logic_tree_utilities/custom_logic_trees/slt_full_only_{selected_source_type}.json')





# slt_only_highest_weight_branches = reduce_to_highest_weighted_branch(slt)
# slt_only_highest_weight_branches.to_json('/home/arr65/src/nshm_logic_tree_utilities/custom_logic_trees/slt_only_highest_weighted_branches.json')
#
# glt_only_highest_weight_branches = reduce_to_highest_weighted_branch(glt)
# glt_only_highest_weight_branches.to_json('/home/arr65/src/nshm_logic_tree_utilities/custom_logic_trees/glt_only_highest_weighted_branches.json')


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