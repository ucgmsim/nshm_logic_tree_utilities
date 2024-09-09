"""
Contains functions to modify logic trees.
"""

import copy
from typing import Optional, Union

import numpy as np
from nzshm_model.logic_tree import (
    GMCMLogicTree,
    SourceLogicTree,
)
from nzshm_model.logic_tree.correlation import LogicTreeCorrelations

from run_toshi_hazard_post_helper import CustomLogicTreePair


def reduce_logic_tree_to_nth_highest_weighted_branch(
    logic_tree: Union[SourceLogicTree, GMCMLogicTree], nth_highest: int
) -> Union[SourceLogicTree, GMCMLogicTree]:
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
    max_num_branches = max(
        [len(branch_set.branches) for branch_set in logic_tree.branch_sets]
    )
    if nth_highest > max_num_branches:
        raise ValueError(
            f"nth_highest ({nth_highest}) is greater than the maximum number of branches in any branch set ({max_num_branches})"
        )

    for branch_set_idx, branch_set in enumerate(modified_logic_tree.branch_sets):

        reverse_sorted_branches = sorted(
            branch_set.branches, key=lambda x: x.weight, reverse=True
        )
        if len(reverse_sorted_branches) == 1:
            print(
                f"Branch set {branch_set.long_name} ({branch_set.short_name}) only has one branch so cannot reduce to nth highest branch."
                f" Leaving this branch_set unchanged."
            )
            selected_branch = copy.deepcopy(reverse_sorted_branches[0])
        elif nth_highest > len(reverse_sorted_branches):
            selected_branch = copy.deepcopy(
                reverse_sorted_branches[len(reverse_sorted_branches) - 1]
            )
            print(
                f"Branch set {branch_set.long_name} ({branch_set.short_name}) has fewer than {nth_highest} "
                f"branches so reducing to its lowest weighted branch (branch {len(reverse_sorted_branches)} of {len(reverse_sorted_branches)})."
            )
        else:
            selected_branch = copy.deepcopy(reverse_sorted_branches[nth_highest - 1])
        selected_branch.weight = 1.0
        modified_logic_tree.branch_sets[branch_set_idx].branches = [selected_branch]

    modified_logic_tree.correlations = LogicTreeCorrelations()

    check_weight_validity(modified_logic_tree)

    return modified_logic_tree


def reduce_lt_set_to_nth_highest_branches(
    initial_logic_tree_set: CustomLogicTreePair,
    slt_nth_highest: Optional[int],
    glt_nth_highest: Optional[int],
) -> CustomLogicTreePair:
    """
    Reduce one or both of the logic trees in a logic tree set to only the nth highest weighted
    branch in each branch set.

    The highest weighted branch is the 1st highest weighted branch (nth_highest = 1).
    The second highest weighted branch is the 2nd highest weighted branch (nth_highest = 2) etc.

    Parameters
    ----------
    initial_logic_tree_set : CustomLogicTreePair
        The initial logic tree set to be modified.

    slt_nth_highest : int, optional
        The nth highest branch to reduce the SourceLogicTree to.

    glt_nth_highest : int, optional
        The nth highest branch to reduce the GMCMLogicTree to.

    Returns
    -------
    modified_logic_tree_set : CustomLogicTreePair
        The logic tree set after being reduced only consisting of the nth highest branch in each branch_set.

    Raises
    ------
    ValueError
        If the branches in the modified_logic_tree do not have valid weights.
        That is its branches in each branch_set do not sum to 1.0.

    IndexError
        If the nth_highest is greater than the number of branches in any branch_set
    """

    modified_logic_tree_set = copy.deepcopy(initial_logic_tree_set)

    if (slt_nth_highest is None) and (glt_nth_highest is None):
        return modified_logic_tree_set

    if slt_nth_highest is not None:
        modified_logic_tree_set.slt = reduce_logic_tree_to_nth_highest_weighted_branch(
            initial_logic_tree_set.slt, slt_nth_highest
        )
        modified_logic_tree_set.slt_note += f"{slt_nth_highest} (nth) h.w.b. > "

    if glt_nth_highest is not None:
        modified_logic_tree_set.glt = reduce_logic_tree_to_nth_highest_weighted_branch(
            initial_logic_tree_set.glt, glt_nth_highest
        )
        modified_logic_tree_set.glt_note += f"{glt_nth_highest} (nth) h.w.b. > "

    return modified_logic_tree_set


def check_weight_validity(logic_tree: Union[SourceLogicTree, GMCMLogicTree]) -> bool:
    """
    Check that the weights of branches in each branch_set sum to 1.0.

    Parameters
    ----------
    logic_tree : SourceLogicTree or GMCMLogicTree
        The logic tree to check.

    Returns
    -------
    True : bool
        Returns True if the weights of branches in each branch_set sum to 1.0.

    Raises
    ------
    ValueError
        If the weights of branches in each branch_set do not sum to 1.0.


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
            f"The weights of branches in each branch_set do not sum to 1.0.\nThe summed weights for each branch_set are {branch_set_summed_weights}."
        )

    return True


def transpose_lists(lists):
    """
    Transpose a list of lists.

    Parameters
    ----------
    lists : list of list
        A list containing sublists to be transposed.

    Returns
    -------
    transposed : list of list
        A list of lists where the rows and columns are swapped.

    Examples
    --------
    >>> lists = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    >>> transpose_lists(lists)
    [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
    """

    transposed = list(map(list, zip(*lists)))
    return transposed


def get_source_branch_parameters(logic_tree):
    """
    Extracts and processes the branch parameters from a logic tree.

    This function iterates through the branch sets of a given logic tree,
    converts the branch values to strings, transposes the lists of values,
    and identifies unique values for each parameter.

    Parameters
    ----------
    logic_tree : Union[SourceLogicTree, GMCMLogicTree]
        The logic tree from which to extract branch parameters.

    Returns
    -------
    unique_values_dict : dict
        A dictionary where keys are branch set short names and values are lists of unique parameter values.
    """

    values_dict = {}

    # Iterate through each branch set in the logic tree
    for branch_set_index, branch_set in enumerate(logic_tree.branch_sets):

        values_list = []

        # Convert branch values to strings and collect them in a list
        for branch_index, branch in enumerate(branch_set.branches):
            values_as_str = [str(value) for value in branch.values]
            values_list.append(values_as_str)

        # Store the list of string values in the dictionary with the branch set short name as the key
        values_dict[branch_set.short_name] = values_list

    # Create a deep copy of the values dictionary for transposition
    transpose_dict = copy.deepcopy(values_dict)

    # Transpose the lists of values
    for key, value in values_dict.items():
        transpose_dict[key] = transpose_lists(value)

    # Create a deep copy of the transposed dictionary to store unique values
    unique_values_dict = copy.deepcopy(transpose_dict)

    # Identify unique values for each parameter in the transposed lists
    for branch_set_index, list_of_branch_values in transpose_dict.items():
        for value_idx, values in enumerate(list_of_branch_values):
            unique_values_dict[branch_set_index][value_idx] = list(set(values))

    return unique_values_dict


def get_params_with_num_options(logic_tree, num_options):
    """
    Extracts parameters from a logic tree that have a specific number of unique options.

    This function processes the branch parameters of a given logic tree and identifies
    parameters that have exactly the specified number of unique values.

    Parameters
    ----------
    logic_tree : Union[SourceLogicTree, GMCMLogicTree]
        The logic tree from which to extract branch parameters.
    num_options : int
        The number of unique options to filter the parameters by.

    Returns
    -------
    dict_n_unique_vals : dict
        A dictionary where keys are branch set short names and values are lists of unique parameter values
        that have exactly `num_options` unique values.
    """

    unique_values_dict = get_source_branch_parameters(logic_tree)

    dict_n_unique_vals = {}
    for key, item in unique_values_dict.items():
        dict_n_unique_vals[key] = []

    for key, item in unique_values_dict.items():
        for unique_val_idx, unique_values in enumerate(item):
            if len(unique_values) == num_options:
                dict_n_unique_vals[key].append(unique_values)

    return dict_n_unique_vals


def select_branch_sets_given_tectonic_region_type(
    logic_tree: Union[SourceLogicTree, GMCMLogicTree],
    tectonic_region_types: Union[list[str], str],
    which_interface: str = "HIK_and_PUY",
):
    """
    Modifies a logic tree to only include branch sets that correspond to the selected tectonic region types.

    Parameters
    ----------
    logic_tree : SourceLogicTree or GMCMLogicTree
        The logic tree to modify.

    tectonic_region_types : list[str] or str
        A list of the selected tectonic region types.
        If selecting only a single tectonic region type, can be a string.
        Valid tectonic region types are:
            "Active Shallow Crust",
            "Subduction Interface",
            "Subduction Intraslab".

    which_interface : str, default = "HIK_and_PUY"
        Which subduction interfaces to include.
        Valid options are:
           "HIK_and_PUY" which includes HIK_and_PUY the Hikurangi–Kermadec (only_HIK) and Puysegur (only_PUY) subduction zones
           "only_HIK" which includes only the Hikurangi–Kermadec (only_HIK) subduction zone
           "only_PUY" which includes only the Puysegur (only_PUY) subduction zone.

    Returns
    -------
    modified_logic_tree : SourceLogicTree or GMCMLogicTree
        The modified logic tree that only includes branch sets corresponding
        to the selected tectonic region type.
    """

    if isinstance(tectonic_region_types, str):
        tectonic_region_types = [tectonic_region_types]

    modified_logic_tree = copy.deepcopy(logic_tree)
    new_branch_sets = []
    for branch_set in logic_tree.branch_sets:
        print()
        if isinstance(logic_tree, SourceLogicTree):

            ## even though each branch_set corresponds to one tectonic region type,
            ## branch_set.tectonic_region_types returns a list of one tectonic region
            # type which is accessed with the for loop
            for tectonic_region_type in branch_set.tectonic_region_types:
                print()
                if tectonic_region_type in tectonic_region_types:
                    print()
                    if tectonic_region_type == "Subduction Interface":
                        if which_interface == "HIK_and_PUY":
                            new_branch_sets.append(copy.deepcopy(branch_set))
                        elif which_interface == "only_HIK":
                            if branch_set.short_name == "HIK":
                                new_branch_sets.append(copy.deepcopy(branch_set))
                        elif which_interface == "only_PUY":
                            if branch_set.short_name == "PUY":
                                new_branch_sets.append(copy.deepcopy(branch_set))

                    else:
                        new_branch_sets.append(copy.deepcopy(branch_set))

        if isinstance(logic_tree, GMCMLogicTree):
            if branch_set.tectonic_region_type in tectonic_region_types:
                new_branch_sets.append(copy.deepcopy(branch_set))

    modified_logic_tree.branch_sets = new_branch_sets
    branch_set_short_names = [x.short_name for x in new_branch_sets]

    if ("only_PUY" in branch_set_short_names) & ("only_HIK" in branch_set_short_names):
        # retain the only_HIK to only_PUY correlations
        pass
    else:
        # remove correlations
        modified_logic_tree.correlations = LogicTreeCorrelations()
    print()
    return modified_logic_tree


def logic_tree_set_with_selected_tectonic_region_types(
    initial_logic_tree_set: CustomLogicTreePair,
    tectonic_region_type_set: list,
    which_interface: Optional[str] = None,
):
    """
    Modifies a logic tree set to only include branch sets that correspond to the selected tectonic region types.

    Parameters
    ----------
    logic_tree : SourceLogicTree or GMCMLogicTree
        The logic tree to modify.

    tectonic_region_type_set : list[str] or str
        A list of the selected tectonic region types.
        If selecting only a single tectonic region type, can be a string.
        Valid tectonic region types are:
            "Active Shallow Crust",
            "Subduction Interface",
            "Subduction Intraslab".

    which_interface : str, default = "HIK_and_PUY"
        Which subduction interfaces to include.
        Valid options are:
           "HIK_and_PUY" which includes HIK_and_PUY the Hikurangi–Kermadec (only_HIK) and Puysegur (only_PUY) subduction zones
           "only_HIK" which includes only the Hikurangi–Kermadec (only_HIK) subduction zone
           "only_PUY" which includes only the Puysegur (only_PUY) subduction zone.

    Returns
    -------
    modified_logic_tree : SourceLogicTree or GMCMLogicTree
        The modified logic tree that only includes branch sets corresponding
        to the selected tectonic region type.
    """

    trt_short_lookup_dict = {
        "Active Shallow Crust": "CRU",
        "Subduction Interface": "INTER",
        "Subduction Intraslab": "SLAB",
    }

    modified_logic_tree_sets = []

    slt = copy.deepcopy(initial_logic_tree_set.slt)
    glt = copy.deepcopy(initial_logic_tree_set.glt)

    modified_lt_set = copy.deepcopy(initial_logic_tree_set)

    short_tectonic_region_types_for_glt_note = [
        trt_short_lookup_dict[trt] for trt in tectonic_region_type_set
    ]
    short_tectonic_region_types_for_slt_note = copy.deepcopy(
        short_tectonic_region_types_for_glt_note
    )

    for short_trt_index, short_trt in enumerate(
        short_tectonic_region_types_for_slt_note
    ):
        if short_trt == "INTER":
            short_tectonic_region_types_for_slt_note[short_trt_index] = (
                f"INTER_{which_interface}"
            )

    modified_slt = select_branch_sets_given_tectonic_region_type(
        slt, tectonic_region_type_set, which_interface
    )
    modified_glt = select_branch_sets_given_tectonic_region_type(
        glt, tectonic_region_type_set
    )

    modified_lt_set.slt_note += f"tectonic_region_type_set:[{' '.join(short_tectonic_region_types_for_slt_note)}] > "

    modified_lt_set.glt_note += f"tectonic_region_type_set:[{' '.join(short_tectonic_region_types_for_glt_note)}] > "

    modified_lt_set.slt = copy.deepcopy(modified_slt)
    modified_lt_set.glt = copy.deepcopy(modified_glt)

    modified_logic_tree_sets.append(modified_lt_set)

    return modified_logic_tree_sets


def print_info_about_logic_tree(logic_tree: Union[SourceLogicTree, GMCMLogicTree]):
    """
    Print information about a logic tree.
    """

    print("")  # Add a blank line for readability

    if isinstance(logic_tree, SourceLogicTree):
        print(f"Logic tree is a SourceLogicTree")

    if isinstance(logic_tree, GMCMLogicTree):
        print(f"Logic tree is a GMCMLogicTree")

    print(f"Logic tree has {len(logic_tree.branch_sets)} branch sets")

    for branch_set_index, branch_set in enumerate(logic_tree.branch_sets):

        print(
            f"Branch set index {branch_set_index} has name {branch_set.long_name} ({branch_set.short_name}) and contains {len(branch_set.branches)} branches"
        )


def print_info_about_logic_tree_sets(
    logic_tree_sets: Union[list[CustomLogicTreePair], CustomLogicTreePair]
):
    """
    Print information about a logic tree set or a list of logic tree sets.

    Parameters
    ----------
    logic_tree_sets : list[CustomLogicTreePair] or CustomLogicTreePair
        The logic_tree_set or logic_tree_sets to print information about.
    """

    if not isinstance(logic_tree_sets, list):
        logic_tree_sets = [logic_tree_sets]

    print(f"Printing information about {len(logic_tree_sets)} logic tree sets")

    for logic_tree_set_index, logic_tree_set in enumerate(logic_tree_sets):
        print()
        print(f"Logic tree at index {logic_tree_set_index}:")
        print(f"slt_note: {logic_tree_sets[logic_tree_set_index].slt_note}")
        print(f"glt_note: {logic_tree_sets[logic_tree_set_index].glt_note}")

        if logic_tree_set.slt is not None:
            print_info_about_logic_tree(logic_tree_set.slt)

        if logic_tree_set.glt is not None:
            print_info_about_logic_tree(logic_tree_set.glt)


def get_logic_tree_sets_for_tectonic_selection(
    initial_logic_tree_set: CustomLogicTreePair,
    tectonic_region_type_sets: list[list[str]],
    which_interfaces,
) -> list[CustomLogicTreePair]:
    """
    Produces a list of logic tree sets with the selected tectonic region types.

    Parameters
    ----------
    initial_logic_tree_set : CustomLogicTreePair
        The initial logic tree set to select tectonic region types from.

    tectonic_region_type_sets : list[list[str]]
        A list of lists the selected tectonic region types for this logic tree set.
            Valid tectonic region types are:
            "Active Shallow Crust",
            "Subduction Interface",
            "Subduction Intraslab".
        Examples:
         [["Active Shallow Crust"], ["Subduction Interface"], ["Subduction Intraslab"]]
         [["Active Shallow Crust", "Subduction Interface"], ["Subduction Intraslab"]]
         ["Subduction Interface"]

    which_interface : str, default = "HIK_and_PUY"
        Which subduction interfaces to include.
        Valid options are:
           "HIK_and_PUY" which includes HIK_and_PUY the Hikurangi–Kermadec (only_HIK) and Puysegur (only_PUY) subduction zones
           "only_HIK" which includes only the Hikurangi–Kermadec (only_HIK) subduction zone
           "only_PUY" which includes only the Puysegur (only_PUY) subduction zone.

    Returns
    -------
    logic_tree_set_list : list[CustomLogicTreePair]
        A list of len(tectonic_region_type_sets) CustomLogicTreePair instances, each containing a modified
    """

    logic_tree_set_list = []

    for tectonic_region_type_set in tectonic_region_type_sets:

        # for trt in tectonic_region_type_set:

        if "Subduction Interface" in tectonic_region_type_set:

            for which_interface in which_interfaces:

                lt_entry_for_trts = logic_tree_set_with_selected_tectonic_region_types(
                    initial_logic_tree_set,
                    tectonic_region_type_set=tectonic_region_type_set,
                    which_interface=which_interface,
                )[0]

                logic_tree_set_list.append(lt_entry_for_trts)

        else:

            # trt_list = [tectonic_region_type_set]

            lt_entry_for_trts = logic_tree_set_with_selected_tectonic_region_types(
                initial_logic_tree_set,
                tectonic_region_type_set=tectonic_region_type_set,
                which_interface=None,
            )[0]

            logic_tree_set_list.append(lt_entry_for_trts)

    return logic_tree_set_list


def get_logic_tree_sets_for_individual_ground_motion_models(
    initial_logic_tree_set: CustomLogicTreePair,
    tectonic_region_type_sets: list[list[str]],
    which_interfaces: list[str],
) -> list[CustomLogicTreePair]:
    """
    Creates a list of logic tree sets with all individual ground motion models within the selected tectonic region types.

    Parameters
    ----------
    initial_logic_tree_set : CustomLogicTreePair
        The initial logic tree set to select tectonic region types from.
        Should contain the full SourceLogicTree and the full GMCMLogicTree.

    tectonic_region_type_sets : list[list[str]]
        A list of lists that each containing a single tectonic region type
            Valid tectonic region types are:
            "Active Shallow Crust",
            "Subduction Interface",
            "Subduction Intraslab".
        Examples:
         [["Active Shallow Crust"], ["Subduction Interface"], ["Subduction Intraslab"]]

    which_interface : str, default = "HIK_and_PUY"
        Which subduction interfaces to include.
        Valid options are:
           "HIK_and_PUY" which includes both the Hikurangi–Kermadec and Puysegur subduction zones
           "only_HIK" which includes only the Hikurangi–Kermadec subduction zone
           "only_PUY" which includes only the Puysegur subduction zone.

    Returns
    -------
    logic_tree_set_list : list[CustomLogicTreePair]
        A list of logic tree sets, each containing only an individual ground motion model.

    Raises
    ------
    AssertionError
        If more than one tectonic_region_type is included in any tectonic_region_type_set.
    """

    assert (
        len(tectonic_region_type_sets[0]) == 1
    ), "Only one tectonic_region_type can be included in each tectonic_region_type_set passed to this function."

    initial_logic_tree_set = reduce_lt_set_to_nth_highest_branches(
        initial_logic_tree_set, slt_nth_highest=1, glt_nth_highest=None
    )

    input_logic_tree_set_list = get_logic_tree_sets_for_tectonic_selection(
        initial_logic_tree_set=initial_logic_tree_set,
        tectonic_region_type_sets=tectonic_region_type_sets,
        which_interfaces=which_interfaces,
    )

    modified_logic_tree_set_list = []

    all_glt_gsim_names = []

    for lt_set in input_logic_tree_set_list:

        glt_gsim_names = [
            branch.gsim_name for branch in lt_set.glt.branch_sets[0].branches
        ]
        all_glt_gsim_names.append(glt_gsim_names)

        unique_gsim_names = list(set(glt_gsim_names))

        for gsim_name in unique_gsim_names:

            selected_glt_branches = [
                copy.deepcopy(branch)
                for branch in lt_set.glt.branch_sets[0].branches
                if branch.gsim_name == gsim_name
            ]

            selected_glt_branch_weights = np.array(
                [
                    copy.deepcopy(branch.weight)
                    for branch in lt_set.glt.branch_sets[0].branches
                    if branch.gsim_name == gsim_name
                ]
            )

            needed_scaling_factor = 1.0 / np.sum(selected_glt_branch_weights)

            scaled_weights = selected_glt_branch_weights * needed_scaling_factor

            for logic_tree_set_index, branch in enumerate(selected_glt_branches):
                branch.weight = scaled_weights[logic_tree_set_index]

            modified_lt_set = copy.deepcopy(lt_set)

            modified_lt_set.glt.branch_sets[0].branches = selected_glt_branches
            modified_lt_set.glt_note += f"[{gsim_name}*{needed_scaling_factor:.2f}] > "

            modified_logic_tree_set_list.append(modified_lt_set)

    return modified_logic_tree_set_list


def get_logic_tree_sets_for_individual_source_models(
    initial_logic_tree_set: CustomLogicTreePair,
    tectonic_region_type_sets: list[list[str]],
    which_interfaces: list[str],
) -> list[CustomLogicTreePair]:
    """
    Creates a list of logic tree sets with all individual ground motion models within the selected tectonic region types.

    Parameters
    ----------
    initial_logic_tree_set : CustomLogicTreePair
        The initial logic tree set to select tectonic region types from.
        Should contain the full SourceLogicTree and the full GMCMLogicTree.

    tectonic_region_type_sets : list[list[str]]
        A list of lists that each containing a single tectonic region type
            Valid tectonic region types are:
            "Active Shallow Crust",
            "Subduction Interface",
            "Subduction Intraslab".
        Examples:
         [["Active Shallow Crust"], ["Subduction Interface"], ["Subduction Intraslab"]]

    which_interface : str, default = "HIK_and_PUY"
        Which subduction interfaces to include.
        Valid options are:
           "HIK_and_PUY" which includes both the Hikurangi–Kermadec and Puysegur subduction zones
           "only_HIK" which includes only the Hikurangi–Kermadec subduction zone
           "only_PUY" which includes only the Puysegur subduction zone.

    Returns
    -------
    logic_tree_set_list : list[CustomLogicTreePair]
        A list of logic tree sets, each containing only an individual ground motion model.

    Raises
    ------
    AssertionError
        If more than one tectonic_region_type is included in any tectonic_region_type_set.
    """

    assert (
        len(tectonic_region_type_sets[0]) == 1
    ), "Only one tectonic_region_type can be included in each tectonic_region_type_set passed to this function."

    initial_logic_tree_set = reduce_lt_set_to_nth_highest_branches(
        initial_logic_tree_set, slt_nth_highest=None, glt_nth_highest=1
    )

    ## print a blank line for clarity
    print()

    input_logic_tree_set_list = get_logic_tree_sets_for_tectonic_selection(
        initial_logic_tree_set=initial_logic_tree_set,
        tectonic_region_type_sets=tectonic_region_type_sets,
        which_interfaces=which_interfaces,
    )

    print_info_about_logic_tree_sets(input_logic_tree_set_list)

    ## print a blank line for clarity
    print()

    # print_info_about_logic_tree_sets(input_logic_tree_set_list)

    new_logic_tree_sets = []

    for lt_set in input_logic_tree_set_list:

        ## print a blank line for clarity
        print()

        source_branch_set_short_name_to_index = {
            branch_set.short_name: branch_set_index
            for branch_set_index, branch_set in enumerate(lt_set.slt.branch_sets)
        }
        ## print a blank line for clarity
        print()

        print_info_about_logic_tree_sets(lt_set)

        needed_branches_dict = get_needed_source_branches(lt_set)

        ### HIK and PUY branch_sets will both be in needed_branches_dict if which_interface == "HIK_and_PUY"
        ### so if they are both present, only change the HIK branch_set in the normal way and then
        ### later add the PUY branch_set so that the correlations will work
        if ("HIK" in needed_branches_dict.keys()) and (
            "PUY" in needed_branches_dict.keys()
        ):
            source_branch_set_names_to_change = ["HIK"]

        else:
            source_branch_set_names_to_change = list(needed_branches_dict.keys())

        assert (
            len(source_branch_set_names_to_change) == 1
        ), "Should only have one source branch set to change at this point"

        for source_short_branch_set_name in source_branch_set_names_to_change:

            branches_for_branch_set = needed_branches_dict[source_short_branch_set_name]

            for param_name, needed_branches in branches_for_branch_set.items():

                modified_logic_tree_set = copy.deepcopy(lt_set)

                modified_logic_tree_set.slt.branch_sets[
                    source_branch_set_short_name_to_index[source_short_branch_set_name]
                ].branches = needed_branches
                modified_logic_tree_set.slt_note += f"{param_name} > "

                ### HIK and PUY branch_sets will both be in needed_branches_dict if which_interface == "HIK_and_PUY"
                ### so if they are both present, only change the HIK branch_set in the normal way and then
                ### add the PUY branch_set so that the correlations will work
                if ("HIK" in needed_branches_dict.keys()) and (
                    "PUY" in needed_branches_dict.keys()
                ):
                    modified_logic_tree_set.slt.branch_sets[
                        source_branch_set_short_name_to_index["PUY"]
                    ].branches = needed_branches_dict["PUY"]["moment_rate_scaling"]

                print()  ## print a blank line for clarity
                print("After modification")
                print_info_about_logic_tree_sets(modified_logic_tree_set)
                new_logic_tree_sets.append(copy.deepcopy(modified_logic_tree_set))

    return new_logic_tree_sets


def remove_single_quotes(input_string: str) -> str:
    """
    Remove all occurrences of the single quote character from a string.

    Parameters
    ----------
    input_string : str
        The input string from which single quotes will be removed.

    Returns
    -------
    modified_string : str
        The modified string with all single quotes removed.

    Examples
    --------
    >>> input_string = "It's a test string."
    >>> remove_single_quotes(input_string)
    'Its a test string.'
    """
    modified_string = input_string.replace("'", "")
    return modified_string


def get_needed_source_branches(logic_tree_set):

    branch_param_index_to_description = {
        "CRU": {
            0: "deformation_model",
            1: "time_dependence",
            2: "MFD",
            4: "moment_rate_scaling",
        },
        "HIK": {1: "MFD", 3: "moment_rate_scaling"},
        "PUY": {3: "moment_rate_scaling"},
    }

    results = {}

    slt_branch_params = get_source_branch_parameters(logic_tree_set.slt)

    slt_highest_weighted_branches = reduce_logic_tree_to_nth_highest_weighted_branch(
        logic_tree_set.slt, 1
    )

    slt_branch_set_short_names = [x.short_name for x in logic_tree_set.slt.branch_sets]

    for slt_branch_set_short_name in slt_branch_set_short_names:
        results[slt_branch_set_short_name] = {}

    for branch_set_index, branch_set in enumerate(logic_tree_set.slt.branch_sets):

        if branch_set.short_name == "SLAB":
            results["SLAB"] = {
                "slab_only_branch": [copy.deepcopy(branch_set.branches[0])]
            }

        num_params = len(slt_branch_params[branch_set.short_name])

        for branch_param_idx in range(num_params):

            selected_branches_per_param = []

            ## Allow both values at index branch_param_idx and keep everything else as the highest weighted branch
            possible_values_for_this_param_index = slt_branch_params[
                branch_set.short_name
            ][branch_param_idx]
            print()

            if len(possible_values_for_this_param_index) == 1:
                continue

            # assert len(slt_highest_weighted_branches.branch_sets) == 1, "should only have 1 branch_set at this point"
            print()
            hwb_values = (
                slt_highest_weighted_branches.branch_sets[branch_set_index]
                .branches[0]
                .values
            )

            ## branch_values_to_find list will be modifed for each param value
            branch_values_to_find = copy.deepcopy(hwb_values)

            for possible_param_value in possible_values_for_this_param_index:

                branch_values_to_find[branch_param_idx] = possible_param_value

                ## This is needed as some string conversions return the text of interest surrounded by ' '
                search_str = remove_single_quotes(str(branch_values_to_find))

                for branch in branch_set.branches:

                    if search_str == str(branch.values):
                        selected_branches_per_param.append(copy.deepcopy(branch))

            ## The branches in selected_branches_per_param will be the only branches in this branch_set so
            ## their total weight needs to sum to 1.0

            total_weight_selected_branches = sum(
                [branch.weight for branch in selected_branches_per_param]
            )
            needed_scaling_factor = 1.0 / total_weight_selected_branches

            for branch in selected_branches_per_param:
                branch.weight *= needed_scaling_factor

            assert sum(
                [branch.weight for branch in selected_branches_per_param]
            ), "Scaled weights of branches do not sum to 1.0"

            results[branch_set.short_name][
                branch_param_index_to_description[branch_set.short_name][
                    branch_param_idx
                ]
            ] = selected_branches_per_param

    return results


def make_srm_model_branch_groups(slt, branch_set_idx_to_do):
    branch_param_index_to_description = {
        0: "deformation model",
        1: "time dependence",
        2: "MFD",
        4: "moment rate scaling",
    }

    branch_param_index_list, all_branch_groups = get_needed_source_branches(
        slt, branch_set_idx_to_do
    )

    assert len(branch_param_index_list) == len(all_branch_groups)

    logic_tree_set_list = []

    for branch_group_idx in range(len(all_branch_groups)):

        modified_slt = copy.deepcopy(slt)

        selected_branches = all_branch_groups[branch_group_idx]

        ## Scale the weights of the selected branches so the total weight of the branches in the branch_set is 1.0
        total_weight_selected_branches = sum(
            [branch.weight for branch in selected_branches]
        )
        needed_scaling_factor = 1.0 / total_weight_selected_branches

        for branch in selected_branches:
            branch.weight *= needed_scaling_factor

        print(f"branch_set_idx_to_do {branch_set_idx_to_do}")

        modified_slt.branch_sets[branch_set_idx_to_do].branches = selected_branches

        ## only keeping the needed branch_set
        modified_slt.branch_sets = [
            copy.deepcopy(modified_slt.branch_sets[branch_set_idx_to_do])
        ]

        modified_slt_note = f"{branch_param_index_to_description[branch_param_index_list[branch_group_idx]]} > "

        ### This will only work if subduction interface (INTER)
        ### tectonic region types (TRTs) are not in the logic tree
        modified_slt.correlations = LogicTreeCorrelations()
        custom_logic_tree_entry = CustomLogicTreePair(
            slt=modified_slt, slt_note=modified_slt_note
        )

        logic_tree_set_list.append(custom_logic_tree_entry)

    return logic_tree_set_list


def make_logic_tree_sets_for_srm_models(
    slt, glt_matching_branch_set, branch_set_idx_to_do
):

    modified_glt = copy.deepcopy(glt_matching_branch_set)

    glt_hwb = reduce_logic_tree_to_nth_highest_weighted_branch(
        glt_matching_branch_set, 1
    )

    modified_glt.branch_sets = [copy.deepcopy(glt_hwb[0].glt.branch_sets[0])]

    lt_set_with_slt = make_srm_model_branch_groups(slt, branch_set_idx_to_do)

    logic_tree_set_list = []

    for lt_set in lt_set_with_slt:

        new_lt_set = CustomLogicTreePair(
            slt=copy.deepcopy(lt_set.slt),
            slt_note=copy.deepcopy(lt_set.slt_note),
            glt=copy.deepcopy(modified_glt),
            glt_note=glt_hwb[0].glt_note + " manually selected CRU branch_set >",
        )

        logic_tree_set_list.append(new_lt_set)

    return logic_tree_set_list
