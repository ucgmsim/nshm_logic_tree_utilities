"""
Functions for loading data from the output of run_toshi_hazard_post_script.py.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

import nshm_logic_tree_utilities.lib.plotting_utilities as plotting_utilities
from nshm_logic_tree_utilities.lib import config as cfg


@dataclass
class LoadedResults:
    """
    A class to store loaded results.

    Attributes
    ----------
    data_df : pd.DataFrame
        A DataFrame containing the data.
    collated_notes_df : pd.DataFrame
        A DataFrame containing notes about
        the run that generated these results.
    """

    data_df: pd.DataFrame()
    collated_notes_df: pd.DataFrame()


def load_aggregate_stats_for_one_logic_tree_one_location(
    results_dir_for_one_logic_tree: Union[Path, str], location: str
) -> pd.DataFrame:
    """
    Load aggregate statistics results for a single location.

    Parameters
    ----------
    results_dir_for_one_logic_tree : Union[Path, str]
        The directory containing the results of a run of run_toshi_hazard_post_script.py for a single logic tree. This directory has a
        name like logic_tree_index_0. This directory should contain location subdirectories.

    location : str
        The location code, which must be one of ["AKL", "WLG", "CHC"].

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the results for the specified location.

    Raises
    ------
    ValueError
        If the location is not one of ["AKL", "WLG", "CHC"].
    """

    if isinstance(results_dir_for_one_logic_tree, str):
        results_dir_for_one_logic_tree = Path(results_dir_for_one_logic_tree)

    results_df = pd.DataFrame()

    if location not in ["AKL", "WLG", "CHC"]:
        raise ValueError("location must be AKL, WLG or CHC")

    if location == "CHC":
        nloc_str = "nloc_0=-44.0~173.0"
    if location == "WLG":
        nloc_str = "nloc_0=-41.0~175.0"
    if location == "AKL":
        nloc_str = "nloc_0=-37.0~175.0"

    # noinspection PyUnboundLocalVariable
    results_dir = results_dir_for_one_logic_tree / nloc_str

    for index, file in enumerate(results_dir.glob("*.parquet")):

        results_df = pd.concat([results_df, pd.read_parquet(file)], ignore_index=True)

    results_df = insert_ln_std(results_df)

    return results_df


def load_aggregate_stats_for_one_logic_tree_several_locations(
    results_dir_for_one_logic_tree: Union[Path, str], locations: tuple[str, ...]
) -> pd.DataFrame:
    """
    Load aggregate statistics results for several locations.

    Parameters
    ----------
    results_dir_for_one_logic_tree : Union[Path, str]
        The directory containing the results of a run of run_toshi_hazard_post_script.py for a single logic tree. This directory has a
        name like logic_tree_index_0. This directory should contain location subdirectories.

    locations : tuple[str]
        Location codes to load. Valid location codes are "AKL", "WLG", "CHC".
    Returns
    -------
    lib.loading_functions.LoadedResults
        An instance of LoadedResults containing loaded data and corresponding notes for the specified locations.
    """
    if isinstance(results_dir_for_one_logic_tree, str):
        results_dir_for_one_logic_tree = Path(results_dir_for_one_logic_tree)

    results_df = pd.DataFrame()

    for location in locations:

        results_df = pd.concat(
            [
                results_df,
                load_aggregate_stats_for_one_logic_tree_one_location(
                    results_dir_for_one_logic_tree, location
                ),
            ],
            ignore_index=True,
        )

    return results_df


def load_aggregate_stats_for_all_logic_trees_in_directory(
    results_directory: Union[Path, str],
    locations: tuple[str, ...] = ("AKL", "WLG", "CHC"),
) -> LoadedResults:
    """
    Load aggregate statistics for all logic trees in the results_directory.

    Parameters
    ----------
    results_directory : Union[Path, str]
        The directory containing the results of one or more logic trees.
        Should contain directories named like logic_tree_index_0, logic_tree_index_0 etc.
    locations : tuple[str], optional
        The locations to plot. Default is ("AKL", "WLG", "CHC").

    Returns
    -------
    LoadedResults
        An instance of LoadedResults containing the data and run notes for all logic trees in the directory.
    """

    config = cfg.Config()

    if isinstance(results_directory, str):
        results_directory = Path(results_directory)

    results_df = pd.DataFrame()

    for run_dir in results_directory.iterdir():
        if run_dir.is_dir():
            results_df = pd.concat(
                [
                    results_df,
                    load_aggregate_stats_for_one_logic_tree_several_locations(
                        run_dir, locations
                    ),
                ],
                ignore_index=True,
            )

    return LoadedResults(
        data_df=results_df,
        collated_notes_df=pd.read_csv(
            results_directory / config.get_value("collated_notes_file_name")
        ),
    )


def insert_ln_std(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Insert the natural logarithm of the standard deviation (std_ln) into the DataFrame.

    This function iterates over the rows of the input DataFrame, checks for rows where the 'agg' column is 'cov',
    calculates the std_ln, and appends these new rows to the DataFrame.

    Parameters
    ----------
    data_df : pd.DataFrame
        The input DataFrame containing the data.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the std_ln rows added.
    """

    # Initialize an empty DataFrame to hold the new rows
    new_rows = []

    # Iterate over the DataFrame
    for index, row in data_df.iterrows():
        # Append the current row to the list of new rows
        new_rows.append(row)

        # Check if the 'agg' column is 'cov'
        if row["agg"] == "cov":

            cov_arr = row.loc["values"]
            std_ln_arr = np.sqrt(np.log(cov_arr**2 + 1))

            # Create a new row that's a copy of the current row
            new_row = row.copy()

            # Update the 'agg' and 'values' columns of the new row
            new_row["agg"] = "std_ln"
            new_row["values"] = std_ln_arr

            # Append the new row to the list of new rows
            new_rows.append(new_row)

    # Create a new DataFrame from the list of new rows
    new_df = pd.DataFrame(new_rows)

    return new_df


def lookup_realization_name_from_hash(
    individual_realization_df: pd.DataFrame, registry_directory: Union[Path, str]
) -> list[plotting_utilities.RealizationName]:
    """
    Looks up the model names used in the realization based on the branch hash ids in the output parquet file.

    Parameters
    ----------
    individual_realization_df : pd.DataFrame
        Contains the individual realizations that are produced by the modified version
        of toshi_hazard_post with output_individual_realizations == True.

    registry_directory : Union[Path, str]
        The directory containing the branch registry files that come with the GNS package nshm-model.

    Returns
    -------
    realization_names : list[plotting_utilities.RealizationName]
        List of plotting_utilities.RealizationName objects containing the seismicity rate model (SRM) and
        ground motion characterization model (GMCM) names.
    """

    gmm_registry_df = pd.read_csv(registry_directory / "gmm_branches.csv")
    source_registry_df = pd.read_csv(registry_directory / "source_branches.csv")

    seismicity_rate_model_ids = []
    ground_motion_characterization_models_ids = []

    ### Get all realization ids (consisting of source and gmm ids) for a single location
    for idx, row in individual_realization_df.iterrows():

        contributing_branches_hash_ids = row["contributing_branches_hash_ids"]
        contributing_branches_hash_ids_clean = plotting_utilities.remove_special_characters(
            contributing_branches_hash_ids
        ).split(", ")

        for contributing_branches_hash_id in contributing_branches_hash_ids_clean:
            seismicity_rate_model_id = contributing_branches_hash_id[0:12]
            ground_motion_characterization_models_id = contributing_branches_hash_id[
                12:24
            ]

            gmm_reg_idx = (
                gmm_registry_df["hash_digest"]
                == ground_motion_characterization_models_id
            )
            ground_motion_characterization_models_id = gmm_registry_df[gmm_reg_idx][
                "identity"
            ].values[0]

            source_reg_idx = (
                source_registry_df["hash_digest"] == seismicity_rate_model_id
            )
            seismicity_rate_model_id = source_registry_df[source_reg_idx][
                "extra"
            ].values[0]

            seismicity_rate_model_ids.append(seismicity_rate_model_id)
            ground_motion_characterization_models_ids.append(
                ground_motion_characterization_models_id
            )

    realization_names = [
        plotting_utilities.RealizationName(
            seismicity_rate_model_id, ground_motion_characterization_models_id
        )
        for seismicity_rate_model_id, ground_motion_characterization_models_id in zip(
            seismicity_rate_model_ids, ground_motion_characterization_models_ids
        )
    ]

    return realization_names
