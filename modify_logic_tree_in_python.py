from toshi_hazard_post.aggregation_args import AggregationArgs
from toshi_hazard_post.aggregation import run_aggregation
from nzshm_model.logic_tree.correlation import LogicTreeCorrelations
import os
import logging
import numpy as np
import copy
import toml
from nzshm_model.logic_tree import GMCMLogicTree, SourceBranchSet, SourceLogicTree
from typing import Optional
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

@dataclass
class CustomLogicTreeSet:
    """
    A dataclass to hold a set of custom logic trees that are to be used in a single run of
    the logic tree realization.

    slt: SourceLogicTree, optional
        The seismicity rate model (SRM) logic tree to be used in the run.
    glt: GMCMLogicTree, optional
        The ground motion characterization model (GMCM) logic tree to be used in the run
    slt_note: str, optional
        A human-readable note describing changes to the SourceLogicTree.
    glt_note: str, optional
        A human-readable note describing changes to the GMCMLogicTree.
    other_notes: str, optional
        Any other notes that are relevant.
    """

    slt: Optional[SourceLogicTree] = None
    glt: Optional[GMCMLogicTree] = None

    slt_note: Optional[str] = None
    glt_note: Optional[str] = None
    other_notes: Optional[str] = None

    def notes_to_toml(self, path: Path):
        data = {
            'slt_note': self.slt_note,
            'glt_note': self.glt_note,
            'other_notes': self.other_notes
        }
        with path.open('w') as f:
            toml.dump(data, f)

    def notes_to_pandas_df(self):
        data = {
            'slt_note': self.slt_note,
            'glt_note': self.glt_note,
            'other_notes': self.other_notes
        }
        return pd.DataFrame(data, index=[0])

def reduce_to_highest_weighted_branch(logic_tree):

    modified_logic_tree = copy.deepcopy(logic_tree)

    for branch_set_idx, branch_set in enumerate(modified_logic_tree.branch_sets):

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

def check_weight_validity(logic_tree):

    """
    Check that the weights of branches in each branch_set sum to 1.0
    """

    lt = copy.deepcopy(logic_tree)

    branch_set_summed_weights = []
    for branch_set in lt.branch_sets:

        branch_set_running_sum = 0.0

        for branch in branch_set.branches:
            branch_set_running_sum += branch.weight

        branch_set_summed_weights.append(branch_set_running_sum)

    if not all(np.isclose(np.array(branch_set_summed_weights), 1.0, rtol=1e-15)):
        raise ValueError(
            f"The weights of branches in each branch_set do not sum to 1.0.\nThe summed weights for each branch_set are {branch_set_summed_weights}.")

    return True

def transpose_lists(lists):
    # Use zip to combine the lists element-wise and convert to a list of lists
    transposed = list(map(list, zip(*lists)))
    return transposed

def get_branch_parameters(logic_tree):
    values_dict = {}

    for branch_set_index, branch_set in enumerate(logic_tree.branch_sets):

        values_list = []

        for branch_index, branch in enumerate(branch_set.branches):
            values_as_str = [str(value) for value in branch.values]
            values_list.append(values_as_str)

        values_dict[branch_set_index] = values_list

    transpose_dict = copy.deepcopy(values_dict)

    for key, value in values_dict.items():
        transpose_dict[key] = transpose_lists(value)


    unique_values_dict = copy.deepcopy(transpose_dict)

    print()

    for branch_set_index, list_of_branch_values in transpose_dict.items():

        print()

        for value_idx, values in enumerate(list_of_branch_values):

            print(value_idx, values)

            unique_values_dict[branch_set_index][value_idx] = list(set(values))

    return unique_values_dict

def get_params_with_num_options(logic_tree, num_options):

    unique_values_dict = get_branch_parameters(logic_tree)

    dict_n_unique_vals = {}
    for key, item in unique_values_dict.items():
        dict_n_unique_vals[key] = []

    print()

    for key, item in unique_values_dict.items():

        for unique_val_idx, unique_values in enumerate(item):

            if len(unique_values) == num_options:

                dict_n_unique_vals[key].append(unique_values)

    return dict_n_unique_vals

def get_slt_permutations_binary_options(logic_tree):

    slt = copy.deepcopy(logic_tree)


    binary_options_dict = get_params_with_num_options(slt, 2)
    param_val_branch_set_idx_dict = {value: key for key in binary_options_dict for sublist in binary_options_dict[key] for value in sublist}

    modified_slt_list = []

    for param_val in param_val_branch_set_idx_dict.keys():

        modified_slt = copy.deepcopy(slt)

        for branch_set_index, branch_set in enumerate(slt.branch_sets):

            if branch_set_index != param_val_branch_set_idx_dict[param_val]:

                # This parameter value is not in this branch set so just continue to loop,
                # leaving this branch_set unchanged from the copied original.

                continue

            else:

                retained_branches = []
                #discarded_branches = []
                total_weighted_deleted_branches = 0.0

                for branch in branch_set.branches:

                    if str(param_val) in str(branch.values):
                        retained_branches.append(copy.deepcopy(branch))
                    else:
                        total_weighted_deleted_branches += branch.weight

                # equally divide the weights of the deleted branches among the retained branches

                additional_weight_per_branch = total_weighted_deleted_branches/len(retained_branches)

                for branch in retained_branches:
                    branch.weight += additional_weight_per_branch

                modified_slt.branch_sets[branch_set_index] = copy.deepcopy(retained_branches)

        modified_slt_list.append(CustomLogicTreeSet(slt=modified_slt,
                                                    slt_note=f'')

    return












        print()



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



    #unique_param_dict = get_branch_parameters(slt)

    # test = get_params_with_num_options(slt, 2)
    # print()
    # testg = get_params_with_num_options(glt, 2)

    # val = 'dmgeodetic'
    #
    # b = slt.branch_sets[2].branches[0]
    #
    # print()

    test = get_slt_permutations_binary_options(slt)

    print()


    # for branch_set_index in range(len(slt.branch_sets)):
    #
    #     branch_set_running_sum = 0.0
    #
    #     for branch_index in range(len(slt.branch_sets[branch_set_index].branches)):
    #         branch_set_running_sum += 0




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