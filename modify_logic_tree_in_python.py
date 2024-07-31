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
import itertools


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

def reduce_to_nth_highest_weighted_branch(logic_tree, nth_highest):

    """
    Reduce a logic tree to only the nth highest weighted branch in each branch set.
    The highest weighted branch is the 1st highest weighted branch (nth_highest = 1).
    The second highest weighted branch is the 2nd highest weighted branch (nth_highest = 2) etc.

    Parameters
    ----------
    logic_tree : SourceLogicTree or GMCMLogicTree
        The logic tree to be modified.

    Returns
    -------
    modified_logic_tree : SourceLogicTree or GMCMLogicTree
        The logic tree after being reduced only consisting of the nth highest branch in each branch_set.

    Raises
    ------
    ValueError
        If the branches in the modified_logic_tree do not have valid weights.
        That is its branches in each branch_set do not sum to 1.0.

    IndexError
        If the nth_highest is greater than the number of branches in any branch_set
    """

    modified_logic_tree = copy.deepcopy(logic_tree)


    # find the maximum number of branches in any branch set
    max_num_branches = max([len(branch_set.branches) for branch_set in logic_tree.branch_sets])
    if nth_highest > max_num_branches:
        raise ValueError(f"nth_highest ({nth_highest}) is greater than the maximum number of branches in any branch set ({max_num_branches})")

    for branch_set_idx, branch_set in enumerate(modified_logic_tree.branch_sets):

        reverse_sorted_branches = sorted(branch_set.branches, key=lambda x: x.weight, reverse=True)
        if len(reverse_sorted_branches) == 1:
            print(f"Branch set {branch_set.long_name} ({branch_set.short_name}) only has one branch so cannot reduce to nth highest branch."
                  f" Leaving this branch_set unchanged.")
            selected_branch = copy.deepcopy(reverse_sorted_branches[0])
        elif nth_highest > len(reverse_sorted_branches):
            selected_branch = copy.deepcopy(reverse_sorted_branches[len(reverse_sorted_branches) - 1])
            print(f"Branch set {branch_set.long_name} ({branch_set.short_name}) has fewer than {nth_highest} "
                  f"branches so reducing to its lowest weighted branch (branch {len(reverse_sorted_branches)} of {len(reverse_sorted_branches)}).")
        else:
            selected_branch = copy.deepcopy(reverse_sorted_branches[nth_highest-1])
        selected_branch.weight = 1.0
        modified_logic_tree.branch_sets[branch_set_idx].branches = [selected_branch]

    modified_logic_tree.correlations = LogicTreeCorrelations()

    check_weight_validity(modified_logic_tree)

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

def get_slt_permutations_binary_options(logic_tree: SourceLogicTree) -> list[CustomLogicTreeSet]:

    """
    Identifies all parameters with only two options (binary) and creates a new SourceLogicTree for each option.

    Parameters
    ----------
    logic_tree : SourceLogicTree
        The SourceLogicTree to be modified.

    Returns
    -------
    modified_slt_list : list[CustomLogicTreeSet]
        A list of CustomLogicTreeSet instances, each containing a modified SourceLogicTree with only one of the binary
        options.

    Raises
    ------
    ValueError
        If the input is not a SourceLogicTree instance.
        If the weights of branches in each branch_set do not sum to 1.0.
    """

    if not isinstance(logic_tree, SourceLogicTree):
        raise ValueError("This function is only for SourceLogicTree instances.")

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

                modified_slt.branch_sets[branch_set_index].branches = copy.deepcopy(retained_branches)
                check_weight_validity(modified_slt)

        modified_slt_list.append(
            CustomLogicTreeSet(
                slt=modified_slt,
                slt_note=f'Using binary option {param_val} in '
                         f'branch_set {slt.branch_sets[param_val_branch_set_idx_dict[param_val]].long_name} '
                         f'({slt.branch_sets[param_val_branch_set_idx_dict[param_val]].short_name})')
        )

    return modified_slt_list

    def retain_n_branch_sets_combination(logic_tree, n_branch_sets_to_retain):

        modified_logic_tree = copy.deepcopy(logic_tree)

        if n_branch_sets_to_retain > len(modified_logic_tree.branch_sets):
            raise ValueError(f"n_branch_sets_to_retain ({n_branch_sets_to_retain}) is greater than the number of branch_sets in the logic tree ({len(modified_logic_tree.branch_sets)})")

        return modified_logic_tree

def make_logic_tree_combinations_list_branch_sets(full_logic_tree, logic_tree_highest_weighted_branches):
    #from nzshm_model.logic_tree import GMCMLogicTree, SourceBranchSet, SourceLogicTree

    logic_tree_permutation_list = []

    for branch_set_index, branch_set in enumerate(full_logic_tree.branch_sets):

        modified_logic_tree = copy.deepcopy(full_logic_tree)

        modified_logic_tree.branch_sets[branch_set_index] = logic_tree_highest_weighted_branches.branch_sets[branch_set_index]
        modified_logic_tree.correlations = LogicTreeCorrelations()

        if isinstance(full_logic_tree, SourceLogicTree):
            custom_logic_tree_entry = modify_logic_tree_in_python.CustomLogicTreeSet(slt = modified_logic_tree,
                        slt_note = f"branch_set {branch_set.long_name} ({branch_set.short_name}) reduced to its single highest weighted branch. No other changes.")

        elif isinstance(full_logic_tree, GMCMLogicTree):
            custom_logic_tree_entry = modify_logic_tree_in_python.CustomLogicTreeSet(glt = modified_logic_tree,
                         glt_note = f"branch_set {branch_set.long_name} ({branch_set.short_name}) reduced to its single highest weighted branch. No other changes.")

        logic_tree_permutation_list.append(custom_logic_tree_entry)

    return logic_tree_permutation_list

def combine_logic_tree_combinations(slt_permutations, glt_permutations):

    combined_permutations = []

    for custom_slt_entry in slt_permutations:

        for custom_glt_entry in glt_permutations:

            slt_glt_entry = modify_logic_tree_in_python.CustomLogicTreeSet(slt=custom_slt_entry.slt,
                                               slt_note=custom_slt_entry.slt_note,
                                               glt=custom_glt_entry.glt,
                                               glt_note=custom_glt_entry.glt_note)

            combined_permutations.append(slt_glt_entry)

    # check that all required parameters are present
    check_validity_of_combinations(combined_permutations)
    return combined_permutations

def check_validity_of_combinations(logic_tree_permutation_list):

    for custom_logic_tree_entry in logic_tree_permutation_list:
        if custom_logic_tree_entry.slt is None:
            raise ValueError("slt is None")
        if custom_logic_tree_entry.slt_note is None:
            raise ValueError("slt_note is None")
        if custom_logic_tree_entry.glt is None:
            raise ValueError("glt is None")
        if custom_logic_tree_entry.glt_note is None:
            raise ValueError("glt_note is None")

    return True

def combinations_of_n_branch_sets(logic_tree, n_branch_sets_to_retain):

    unchanged_logic_tree = copy.deepcopy(logic_tree)

    modified_logic_tree_list = []

    logic_tree_branch_set_indices = range(len(logic_tree.branch_sets))

    branch_set_index_combinations = itertools.combinations(logic_tree_branch_set_indices, n_branch_sets_to_retain)

    for combination in branch_set_index_combinations:

        modified_logic_tree = copy.deepcopy(logic_tree)

        modified_branch_set = []

        for branch_set_index in logic_tree_branch_set_indices:

            if branch_set_index in combination:
                modified_branch_set.append(copy.deepcopy(unchanged_logic_tree.branch_sets[branch_set_index]))

        modified_logic_tree.branch_sets = modified_branch_set
        branch_set_short_names = [x.short_name for x in modified_branch_set]

        if ("PUY" in branch_set_short_names) & ("HIK" in branch_set_short_names):
            # retain the HIK to PUY correlations
            pass
        else:
            # remove correlations
            modified_logic_tree.correlations = LogicTreeCorrelations()

        if isinstance(unchanged_logic_tree, SourceLogicTree):
            custom_logic_tree_entry = CustomLogicTreeSet(slt = modified_logic_tree,
                        slt_note = f"{str(branch_set_short_names)}")

        elif isinstance(unchanged_logic_tree, GMCMLogicTree):
            custom_logic_tree_entry = CustomLogicTreeSet(glt = modified_logic_tree,
                         glt_note = f"{str(branch_set_short_names)}")
            print()

        modified_logic_tree_list.append(custom_logic_tree_entry)

    return modified_logic_tree_list























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

    #test = get_slt_permutations_binary_options(slt)

    #test = get_slt_permutations_binary_options(slt)

    #test = reduce_to_nth_highest_weighted_branch(logic_tree = slt, nth_highest = 9)

    test = combinations_of_n_branch_sets(slt, 1)

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