from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path
import pandas as pd
import scipy
import pyarrow.dataset as ds
from typing import Union


from cycler import cycler
import natsort
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt, ticker as mticker
import toshi_hazard_post.calculators as calculators

##########################################
### dataclasses to store data

## Tidied up
@dataclass
class MarginFractions:
    """
    A dataclass to store margin fractions for a plot.

    Attributes:
    left (float): Fraction of the figure width for the left margin.
    right (float): Fraction of the figure width for the right margin.
    bottom (float): Fraction of the figure height for the bottom margin.
    top (float): Fraction of the figure height for the top margin.
    """
    left: float
    right: float
    bottom: float
    top: float

## Tidied up
@dataclass
class LoadedResults():

    """
    A class to store loaded results.

    Attributes
    ----------
    data_df : pd.DataFrame
        A DataFrame containing the data.
    run_notes_df : pd.DataFrame
        A DataFrame containing notes about
        the run that generated these results.
    """

    data_df :  pd.DataFrame()
    run_notes_df : pd.DataFrame()

## tidied up
@dataclass
class RealizationName():
    """
    Class to store the names of the seismicity rate model and ground motion
    characterization model used in a realization.
    """
    seismicity_rate_model_id: str
    ground_motion_characterization_models_id: str

##########################################

### tidied up
def convert_edge_margin_in_pixels_to_fraction(fig: matplotlib.figure.Figure,
                                              left_margin_pixels: int,
                                              right_margin_pixels: int,
                                              bottom_margin_pixels: int,
                                              top_margin_pixels: int) -> MarginFractions:
    """
    Convert edge margins from pixels to fractions of the figure dimensions.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object to get dimensions from.
    left_margin_pixels : int
        The left margin in pixels.
    right_margin_pixels : int
        The right margin in pixels.
    bottom_margin_pixels : int
        The bottom margin in pixels.
    top_margin_pixels : int
        The top margin in pixels.

    Returns
    -------
    MarginFractions
        A dataclass containing the margins as fractions of the figure dimensions.
    """

    # Get the figure height and width in inches and convert it to pixels
    fig_width_inch = fig.get_figwidth()
    fig_height_inch = fig.get_figheight()

    dpi = fig.get_dpi()

    fig_width_px = fig_width_inch * dpi
    fig_height_px = fig_height_inch * dpi

    left_margin_fraction = left_margin_pixels / fig_width_px
    right_margin_fraction = 1.0 - right_margin_pixels / fig_width_px
    bottom_margin_fraction = bottom_margin_pixels / fig_height_px
    top_margin_fraction = 1.0 - top_margin_pixels / fig_height_px

    return MarginFractions(left_margin_fraction, right_margin_fraction, bottom_margin_fraction, top_margin_fraction)


## tidied up function
def remove_special_characters(s: str, chars_to_remove: list[str] = ["'", "[", "]", '"']) -> str:
    """
    Remove specified special characters from a string.

    Parameters
    ----------
    s : str
        The input string from which to remove characters.
    chars_to_remove : list of str, optional
        A list of characters to remove from the input string. Default is ["'", "[", "]", '"'].

    Returns
    -------
    str
        The string with the specified characters removed.
    """
    translation_table = str.maketrans('', '', ''.join(chars_to_remove))
    return s.translate(translation_table)


### tidied up
def load_aggregate_stats_for_one_logic_tree_one_location(results_dir_for_one_logic_tree: Union[Path, str], location: str) -> pd.DataFrame:

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

    if location not in ["AKL","WLG","CHC"]:
        raise ValueError('location must be AKL, WLG or CHC')

    if location == 'CHC':
        nloc_str = "nloc_0=-44.0~173.0"
    if location == 'WLG':
        nloc_str = "nloc_0=-41.0~175.0"
    if location == 'AKL':
        nloc_str = "nloc_0=-37.0~175.0"

    results_dir = results_dir_for_one_logic_tree / nloc_str

    for index, file in enumerate(results_dir.glob('*.parquet')):

        results_df = pd.concat([results_df, pd.read_parquet(file)], ignore_index=True)

    results_df = insert_ln_std(results_df)

    return results_df

### tidied up
def load_aggregate_stats_for_one_logic_tree_several_locations(results_dir_for_one_logic_tree: Union[Path, str], locations: list[str]) -> pd.DataFrame:

    """
    Load aggregate statistics results for several locations.

    Parameters
    ----------
    results_dir_for_one_logic_tree : Union[Path, str]
        The directory containing the results of a run of run_toshi_hazard_post_script.py for a single logic tree. This directory has a
        name like logic_tree_index_0. This directory should contain location subdirectories.

    locations : list[str]
        A list of location codes, each of which must be one of ["AKL", "WLG", "CHC"].

    Returns
    -------
    LoadedResults
        An instance of LoadedResults containing loaded data and corresponding notes for the specified locations.
    """
    if isinstance(results_dir_for_one_logic_tree, str):
        results_dir_for_one_logic_tree = Path(results_dir_for_one_logic_tree)

    results_df = pd.DataFrame()

    for location in locations:

        results_df = (pd.concat
                      ([results_df,
                        load_aggregate_stats_for_one_logic_tree_one_location(results_dir_for_one_logic_tree, location)],
                       ignore_index=True))

    return results_df

### tidied up
def load_aggregate_stats_for_all_logic_trees_in_directory(results_directory: Union[Path, str],
                                                          locations:list[str] = ["AKL","WLG","CHC"]) -> LoadedResults:

    """
    Load aggregate statistics for all logic trees in the results_directory.

    Parameters
    ----------
    results_directory : Union[Path, str]
        The directory containing the results of one or more logic trees.
        Should contain directories named like logic_tree_index_0, logic_tree_index_0 etc.
    locations : list of str, optional
        A list of location codes, each of which must be one of ["AKL", "WLG", "CHC"]. Default is ["AKL", "WLG", "CHC"].

    Returns
    -------
    LoadedResults
        An instance of LoadedResults containing the data and run notes for all logic trees in the directory.
    """

    if isinstance(results_directory, str):
        results_directory = Path(results_directory)

    results_df = pd.DataFrame()

    for run_dir in results_directory.iterdir():
        if run_dir.is_dir():
            results_df = (pd.concat
                          ([results_df,
                            load_aggregate_stats_for_one_logic_tree_several_locations(run_dir,locations)],
                           ignore_index=True))

    return LoadedResults(data_df=results_df, run_notes_df=pd.read_csv(results_directory / "run_notes.csv"))

## Tidied up
def insert_ln_std(data_df:pd.DataFrame) -> pd.DataFrame:

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
        if row['agg'] == 'cov':

            cov_arr = row.loc["values"]
            std_ln_arr = np.sqrt(np.log(cov_arr ** 2 + 1))

            # Create a new row that's a copy of the current row
            new_row = row.copy()

            # Update the 'agg' and 'values' columns of the new row
            new_row['agg'] = 'std_ln'
            new_row['values'] = std_ln_arr

            # Append the new row to the list of new rows
            new_rows.append(new_row)

    # Create a new DataFrame from the list of new rows
    new_df = pd.DataFrame(new_rows)

    return new_df


# def remove_duplicates_in_x(x, y):
#     # Find unique values in x and their indices
#     unique_x, indices = np.unique(x, return_index=True)
#
#     # Use indices to filter both x and y
#     filtered_x = x[indices]
#     filtered_y = y[indices]
#
#     return filtered_x, filtered_y


### tidied up
def lookup_realization_name_from_hash(individual_realization_df: pd.DataFrame,
                                      registry_directory: Union[Path, str]) -> list[RealizationName]:
    """
    Looks up the model names used in the realization based on the branch hash ids in the output parquet file.

    Parameters
    ----------
    individual_realization_df : pd.DataFrame
        Dataframe containing the individual realizations
        (produced by the modified version of toshi_hazard_post with output_individual_realizations == True).
    registry_directory : Union[Path, str]
        The directory containing the branch registry files that come with the GNS package nshm-model.

    Returns
    -------
    realization_names : list[RealizationName]
        List of RealizationName objects containing the seismicity rate model (SRM) and
        ground motion characterization model (GMCM) names.
    """

    gmm_registry_df = pd.read_csv(registry_directory / 'gmm_branches.csv')
    source_registry_df = pd.read_csv(registry_directory / 'source_branches.csv')

    seismicity_rate_model_ids = []
    ground_motion_characterization_models_ids = []

    ### Get all realization ids (consisting of source and gmm ids) for a single location
    for idx, row in individual_realization_df.iterrows():

        contributing_branches_hash_ids = row["contributing_branches_hash_ids"]
        contributing_branches_hash_ids_clean = remove_special_characters(contributing_branches_hash_ids).split(", ")

        for contributing_branches_hash_id in contributing_branches_hash_ids_clean:
            seismicity_rate_model_id = contributing_branches_hash_id[0:12]
            ground_motion_characterization_models_id = contributing_branches_hash_id[12:24]

            gmm_reg_idx = gmm_registry_df["hash_digest"] == ground_motion_characterization_models_id
            ground_motion_characterization_models_id = gmm_registry_df[gmm_reg_idx]["identity"].values[0]

            source_reg_idx = source_registry_df["hash_digest"] == seismicity_rate_model_id
            seismicity_rate_model_id = source_registry_df[source_reg_idx]["extra"].values[0]

            seismicity_rate_model_ids.append(seismicity_rate_model_id)
            ground_motion_characterization_models_ids.append(ground_motion_characterization_models_id)

    realization_names = [RealizationName(seismicity_rate_model_id, ground_motion_characterization_models_id)
                         for seismicity_rate_model_id, ground_motion_characterization_models_id in
                         zip(seismicity_rate_model_ids, ground_motion_characterization_models_ids)]

    return realization_names


### Used function
### tidied up
def sort_logic_tree_index_by_gmcm_model_name(results_directory: Path) -> list[str]:
    """
    Sort the logic_tree_index_[x] names by the ground motion characterization model that
    was isolated by that logic tree.

    Parameters
    ----------
    results_directory : Path
        The directory containing the run notes CSV file.

    Returns
    -------
    list[str]
        A list of sorted logic tree indices based on the ground motion characterization model names.
    """

    run_list_label_tuple_list = []

    # Read the run notes CSV file
    run_notes_df = pd.read_csv(results_directory / "run_notes.csv")

    # Iterate over each logic tree index
    for logic_tree_index in run_notes_df["logic_tree_index"]:

        # Extract the slt_note and glt_note for the current logic tree index
        slt_note = f"{run_notes_df[run_notes_df["logic_tree_index"] == logic_tree_index]["slt_note"].values[0]}"
        glt_note = f"{run_notes_df[run_notes_df["logic_tree_index"]== logic_tree_index]["glt_note"].values[0]}"

        # Isolate the useful parts of the notes
        trts_from_note = slt_note.split(">")[-2].strip().split(":")[-1].strip("[]")
        glt_model_and_weight_str = glt_note.split(">")[-2].strip(" []")
        glt_model = glt_model_and_weight_str.split("*")[0]

        # Handling special cases with the NZNSHM2022_ prefix
        if "NZNSHM2022_" in glt_model:
            glt_model = glt_model.split("NZNSHM2022_")[1]

        # Get tuples of (logic_tree_index, corresponding model name)
        run_list_label_tuple_list.append((logic_tree_index, f"{trts_from_note}_{glt_model}"))

    # Sort the list of tuples based on the model names
    sorted_run_list_label_tuple_list = natsort.natsorted(run_list_label_tuple_list, key=lambda x: x[1])

    # Return the sorted list of logic tree indices
    return [x[0] for x in sorted_run_list_label_tuple_list]

