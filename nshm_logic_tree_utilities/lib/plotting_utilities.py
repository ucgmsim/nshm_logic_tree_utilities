"""
This module contains helpful functions and utilities for plotting logic tree investigation results.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import matplotlib
import natsort
import numpy as np
import pandas as pd
import scipy
import toml
from matplotlib import pyplot as plt

import nshm_logic_tree_utilities.lib.loading_functions as loading_functions

##########################################
### dataclasses to store loaded data


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


@dataclass
class RealizationName:
    """
    Class to store the names of the seismicity rate model and ground motion
    characterization model used in a realization.
    """

    seismicity_rate_model_id: str
    ground_motion_characterization_models_id: str


################################################
### functions


def convert_edge_margin_in_pixels_to_fraction(
    fig: matplotlib.figure.Figure,
    left_margin_pixels: int,
    right_margin_pixels: int,
    bottom_margin_pixels: int,
    top_margin_pixels: int,
) -> MarginFractions:
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

    return MarginFractions(
        left_margin_fraction,
        right_margin_fraction,
        bottom_margin_fraction,
        top_margin_fraction,
    )


def remove_special_characters(
    s: str, chars_to_remove: tuple[str] = ("'", "[", "]", '"')
) -> str:
    """
    Remove specified special characters from a string.

    Parameters
    ----------
    s : str
        The input string from which to remove characters.
    chars_to_remove : tuple[str], optional
        Characters to remove from the input string. Default is ("'", "[", "]", '"').

    Returns
    -------
    str
        The string with the specified characters removed.
    """
    translation_table = str.maketrans("", "", "".join(chars_to_remove))
    return s.translate(translation_table)


def sort_logic_tree_index_by_gmcm_model_name(
    concatenated_notes_df: pd.DataFrame,
) -> list[str]:
    """
    Sort the logic_tree_index_[x] names by the ground motion characterization model that
    was isolated by that logic tree.

    Parameters
    ----------
    concatenated_notes_df : pd.DataFrame
        A DataFrame containing the concatenated notes for all logic trees.

    Returns
    -------
    list[str]
        A list of sorted logic tree indices based on the ground motion characterization model names.
    """

    run_list_label_tuple_list = []

    # Iterate over each logic tree index
    for logic_tree_index in concatenated_notes_df["logic_tree_index"]:

        # Extract the source_logic_tree_note and ground_motion_logic_tree_note for the current logic tree index
        # noinspection DuplicatedCode
        source_logic_tree_note = f"{concatenated_notes_df[concatenated_notes_df["logic_tree_index"] == logic_tree_index]["source_logic_tree_note"].values[0]}"
        ground_motion_logic_tree_note = f"{concatenated_notes_df[concatenated_notes_df["logic_tree_index"]== logic_tree_index]["ground_motion_logic_tree_note"].values[0]}"

        # Isolate the useful parts of the notes
        tectonic_region_type_group_from_note = (
            source_logic_tree_note.split(">")[-2].strip().split(":")[-1].strip("[]")
        )
        ground_motion_logic_tree_model_and_weight_str = (
            ground_motion_logic_tree_note.split(">")[-2].strip(" []")
        )
        ground_motion_logic_tree_model = (
            ground_motion_logic_tree_model_and_weight_str.split("*")[0]
        )

        # Handling special cases with the NZNSHM2022_ prefix
        if "NZNSHM2022_" in ground_motion_logic_tree_model:
            ground_motion_logic_tree_model = ground_motion_logic_tree_model.split(
                "NZNSHM2022_"
            )[1]

        # Get tuples of (logic_tree_index, corresponding model name)
        run_list_label_tuple_list.append(
            (
                logic_tree_index,
                f"{tectonic_region_type_group_from_note}_{ground_motion_logic_tree_model}",
            )
        )

    # Sort the list of tuples based on the model names
    sorted_run_list_label_tuple_list = natsort.natsorted(
        run_list_label_tuple_list, key=lambda x: x[1]
    )

    # Return the sorted list of logic tree indices
    return [x[0] for x in sorted_run_list_label_tuple_list]


def remove_duplicates_in_x(x: np.ndarray, y: np.ndarray) -> tuple:
    """
    Remove duplicate values in x and filter y accordingly.

    Parameters
    ----------
    x : np.ndarray
        The input array from which to remove duplicate values.
    y : np.ndarray
        The input array to filter based on the unique values in x.

    Returns
    -------
    tuple
        A tuple containing two arrays: the filtered x and y arrays.
    """
    # Find unique values in x and their indices
    unique_x, indices = np.unique(x, return_index=True)  # noqa: F841

    # Use indices to filter both x and y
    filtered_x = x[indices]
    filtered_y = y[indices]

    return filtered_x, filtered_y


def get_interpolated_gmms(
    results_directory: Union[Path, str],
    locations: tuple[str, ...],
    filter_strs: tuple[str, ...],
    vs30: int,
    im: str,
    num_interp_mean_points: int,
    min_log10_mean_for_interp: int,
    max_log10_mean_for_interp: int,
    plot_interpolations: bool,
    min_mean_value_for_interp_plots: float,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Get interpolated ground motion models (GMMs) for specified locations and filters.

    Parameters
    ----------
    results_directory : Union[Path, str]
        The directory containing the results data.
    locations : tuple[str]
        The locations to include in the interpolation.
    filter_strs : tuple[str]
        The filter strings needed to select the desired data.
    vs30 : int
        The Vs30 value to use in the interpolation.
    im : str
        The intensity measure to use in the interpolation.
    num_interp_mean_points : int
        The number of interpolation points for the mean.
    min_log10_mean_for_interp : int
        The minimum log10 mean value for interpolation.
    max_log10_mean_for_interp : int
        The maximum log10 mean value for interpolation.
    plot_interpolations : bool
        Whether to plot interpolations.
    min_mean_value_for_interp_plots : float
        The minimum mean value for interpolation plots.

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
        A dictionary containing the dispersion ranges for each location and filter.
    """

    loaded_results = (
        loading_functions.load_aggregate_stats_for_all_logic_trees_in_directory(
            results_directory, locations
        )
    )
    data_df = loaded_results.data_df
    collated_notes_df = loaded_results.collated_notes_df

    dispersion_range_dict = {}

    for location in locations:
        dispersion_range_dict[location] = {}

    for filter_str in filter_strs:

        filtered_run_notes_df = collated_notes_df[
            collated_notes_df["source_logic_tree_note"].str.contains(filter_str)
        ]

        logic_tree_names = [
            f"logic_tree_index_{x}"
            for x in filtered_run_notes_df["logic_tree_index"].values
        ]

        filtered_data_df = data_df[data_df["hazard_model_id"].isin(logic_tree_names)]

        for location in locations:

            _, interp_disp_array = interpolate_ground_motion_models(
                filtered_data_df,
                location,
                vs30,
                im,
                num_interp_mean_points,
                min_log10_mean_for_interp,
                max_log10_mean_for_interp,
                plot_interpolations,
                min_mean_value_for_interp_plots,
            )

            dispersion_range_dict[location][filter_str] = np.nanmax(
                interp_disp_array, axis=0
            ) - np.nanmin(interp_disp_array, axis=0)

    return dispersion_range_dict


def interpolate_ground_motion_models(
    data_df: pd.DataFrame,
    location: str,
    vs30: int,
    im: str,
    num_interp_mean_points: int,
    min_log10_mean_for_interp: int,
    max_log10_mean_for_interp: int,
    plot_interpolations: bool,
    min_mean_value_for_interp_plots: float,
):
    """
    Interpolate ground motion models (GMMs) for a specified location.

    Parameters
    ----------
    data_df : pd.DataFrame
        The DataFrame containing the data to be interpolated.
    location : str
        The location code for which to perform the interpolation.
    vs30 : int
        The Vs30 value to use in the interpolation.
    im : str
        The intensity measure to use in the interpolation.
    num_interp_mean_points : int
        The number of interpolation points for the mean.
    min_log10_mean_for_interp : int
        The minimum log10 mean value for interpolation.
    max_log10_mean_for_interp : int
        The maximum log10 mean value for interpolation.
    plot_interpolations : bool
        Whether to plot interpolations.
    min_mean_value_for_interp_plots : float
        The minimum mean value for interpolation plots.

    Returns
    -------
    tuple
        A tuple containing the interpolation points (mean_interpolation_points) and the interpolated dispersion array (interp_disp_array).
    """

    mean_list = []
    std_ln_list = []
    non_zero_name_list = []

    locations_nloc_dict = toml.load(
        Path(__file__).parent.parent / "resources/location_code_to_nloc_str.toml"
    )

    ### Gather the data for the interpolation
    for logic_tree_name in natsort.natsorted(data_df["hazard_model_id"].unique()):

        # noinspection DuplicatedCode
        nloc_001_str = locations_nloc_dict[location]

        mean = data_df[
            (data_df["agg"] == "mean")
            & (data_df["vs30"] == vs30)
            & (data_df["imt"] == im)
            & (data_df["hazard_model_id"] == logic_tree_name)
            & (data_df["nloc_001"] == nloc_001_str)
        ]["values"].values[0]

        # noinspection DuplicatedCode
        mean_max = np.max(mean)
        print(f"logic_tree_name {logic_tree_name} max mean: {mean_max}")

        std_ln = data_df[
            (data_df["agg"] == "std_ln")
            & (data_df["vs30"] == vs30)
            & (data_df["imt"] == im)
            & (data_df["hazard_model_id"] == logic_tree_name)
            & (data_df["nloc_001"] == nloc_001_str)
        ]["values"].values[0]

        mean_list.append(mean)
        std_ln_list.append(std_ln)
        non_zero_name_list.append(logic_tree_name)

    mean_array = np.array(mean_list)
    std_ln_array = np.array(std_ln_list)

    mean_interpolation_points = np.logspace(
        min_log10_mean_for_interp, max_log10_mean_for_interp, num_interp_mean_points
    )

    interp_disp_array = np.zeros((len(mean_array), num_interp_mean_points))

    ### Interpolate the data
    for model_idx in range(len(mean_array)):

        std_ln_vect = std_ln_array[model_idx]
        mean_vect = mean_array[model_idx]

        std_ln_vect2 = std_ln_vect[mean_vect > min_mean_value_for_interp_plots]
        mean_vect2 = mean_vect[mean_vect > min_mean_value_for_interp_plots]

        mean_vect3, std_ln_vect3 = remove_duplicates_in_x(mean_vect2, std_ln_vect2)

        if len(mean_vect3) == 0:
            interp_disp_array[model_idx, :] = np.nan

        else:

            mean_to_dispersion_func = scipy.interpolate.interp1d(
                mean_vect3,
                std_ln_vect3,
                kind="linear",
                bounds_error=False,
                fill_value=np.nan,
            )
            interp_disp = mean_to_dispersion_func(mean_interpolation_points)
            interp_disp_array[model_idx, :] = interp_disp

            if plot_interpolations:
                plt.figure()

                print(
                    f"model_idx: {model_idx}, logic_tree_name: {non_zero_name_list[model_idx]}"
                )

                plt.semilogx(
                    mean_vect, std_ln_vect, ".", label="data points before lower cutoff"
                )

                plt.semilogx(
                    mean_vect3, std_ln_vect3, "r.", label="available data points"
                )
                plt.semilogx(
                    mean_interpolation_points,
                    interp_disp,
                    "r--",
                    label="interpolation (range that is defined for all models)",
                )
                plt.title(f"model_index_{model_idx} location {location}")
                plt.xlabel(
                    rf"Mean annual hazard probability, $\mu_{{P({im.upper()}={im.lower()})}}$"
                )
                plt.ylabel(
                    rf"Range in dispersion in hazard probability, $\sigma_{{\ln P({im.upper()}={im.lower()})}}$"
                )
                plt.legend()
                plt.show()

    return mean_interpolation_points, interp_disp_array
