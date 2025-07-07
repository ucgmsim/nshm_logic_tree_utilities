"""
This file contains functions to generate figures of the logic tree investigation results.
"""

import re
from pathlib import Path
from typing import Union

import natsort
import numpy as np
import pyarrow.dataset as ds
import toml
import toshi_hazard_post.calculators as calculators
from matplotlib import pyplot as plt

import nshm_logic_tree_utilities.lib.constants as constants
import nshm_logic_tree_utilities.lib.loading_functions as loading_functions
import nshm_logic_tree_utilities.lib.plotting_utilities as plotting_utilities


def make_figure_of_coefficient_of_variation(
    results_directory: Union[Path, str],
    plot_output_directory: Union[Path, str],
    plot_dpi: int = 500,
    plot_fontsize: float = 12.0,
    plot_lineweight: float = 5.0,
    location: constants.LocationCode = constants.LocationCode.WLG,
    im: constants.IntensityMeasure = constants.IntensityMeasure.PGA,
    vs30: int = 400,
    xlims: tuple[float, float] = (1e-2, 5),
    ylims: tuple[float, float] = (0.05, 0.8),
    figsize: tuple[float, float] = (7.3, 4.62),
):
    """
    Generates a figure showing the coefficient of variation of model predictions
    for a given location, intensity measure, and Vs30 value.

    Parameters
    ----------
    results_directory : Union[Path, str]
        The directory containing the data. This directory should contain subdirectories
        named as logic_tree_index_[x] where [x] is the index the logic_tree_set had in the input list.
    plot_output_directory : Union[Path, str]
        The directory where the plot will be saved.
    plot_dpi : int, optional
        The resolution of the plot in dots per inch (default is 500).
    plot_fontsize : float, optional
        The font size used in the plot (default is 12.0).
    plot_lineweight : int, optional
        The line weight used in the plot (default is 5).
    location : constants.LocationCode, optional
        The location code (default is constants.LocationCode.WLG).
    im : constants.IntensityMeasure, optional
        The intensity measure (default is constants.IntensityMeasure.PGA).
    vs30 : int, optional
        The Vs30 value (default is 400).
    xlims : tuple, optional
        The x-axis limits of the plot (default is (1e-2, 5)).
    ylims : tuple, optional
        The y-axis limits of the plot (default is (0.05, 0.8)).
    figsize : tuple, optional
        The size of the figure in inches (default is (7.3, 4.62)).
        The default size was used for the poster and size of
        (5.12,4.62) was used for the poster showcase slide.

    """

    nshm_im_levels = np.loadtxt(Path(__file__).parent.parent / "resources/nshm_im_levels.txt")

    if isinstance(results_directory, str):
        results_directory = Path(results_directory)
    if isinstance(plot_output_directory, str):
        plot_output_directory = Path(plot_output_directory)
    plot_output_directory.mkdir(parents=True, exist_ok=True)

    locations_nloc_dict = toml.load(Path(__file__).parent.parent / "resources/location_code_to_nloc_str.toml")

    nloc_001_str = locations_nloc_dict[location]

    mean_list = []
    cov_list = []

    resulting_hazard_curves = (
        loading_functions.load_aggregate_stats_for_all_logic_trees_in_directory(
            results_directory
        )
    )

    data_df = resulting_hazard_curves.data_df
    collated_notes_df = resulting_hazard_curves.collated_notes_df

    ### Identify the outputs that are needed
    source_logic_tree_note_condition_idx = (
        collated_notes_df["source_logic_tree_note"] == "full > "
    ) | (collated_notes_df["source_logic_tree_note"] == "full > 1 (nth) h.w.b. > ")

    ground_motion_logic_tree_note_condition_idx = (
        collated_notes_df["ground_motion_logic_tree_note"] == "full > "
    ) | (
        collated_notes_df["ground_motion_logic_tree_note"] == "full > 1 (nth) h.w.b. > "
    )

    collated_notes_df = collated_notes_df[
        source_logic_tree_note_condition_idx
        & ground_motion_logic_tree_note_condition_idx
    ]

    logic_tree_index_list = [
        f"logic_tree_index_{x}" for x in collated_notes_df["logic_tree_index"].values
    ]

    for logic_tree_idx, logic_tree_index_str in enumerate(logic_tree_index_list):
        mean = data_df[
            (data_df["agg"] == "mean")
            & (data_df["vs30"] == vs30)
            & (data_df["imt"] == im)
            & (data_df["hazard_model_id"] == logic_tree_index_str)
            & (data_df["nloc_001"] == nloc_001_str)
        ]["values"].values[0]

        cov = data_df[
            (data_df["agg"] == "cov")
            & (data_df["vs30"] == vs30)
            & (data_df["imt"] == im)
            & (data_df["hazard_model_id"] == logic_tree_index_str)
            & (data_df["nloc_001"] == nloc_001_str)
        ]["values"].values[0]

        mean_list.append(mean)
        cov_list.append(cov)

    original_fontsize = plt.rcParams["font.size"]
    plt.rcParams.update({"font.size": plot_fontsize})

    plt.figure(figsize=figsize)

    plt.semilogx(
        nshm_im_levels,
        cov_list[0],
        linestyle="--",
        linewidth=plot_lineweight,
        label="source model",
    )
    plt.semilogx(
        nshm_im_levels,
        cov_list[1],
        linestyle="-.",
        linewidth=plot_lineweight,
        label="ground motion model",
    )
    plt.semilogx(
        nshm_im_levels,
        cov_list[2],
        linestyle="-",
        linewidth=plot_lineweight,
        label="both",
    )
    plt.legend(handlelength=4)
    plt.ylabel("Modelling uncertainty\n(coefficient of variation of model predictions)")
    plt.xlabel("Peak ground acceleration (g)")
    plt.xlim(xlims)
    plt.ylim(ylims)

    plt.grid(which="major", linestyle="--", linewidth="0.5", color="black", alpha=0.6)

    plt.subplots_adjust(left=0.11, right=0.99, bottom=0.12, top=0.97)
    plt.savefig(plot_output_directory / "coefficient_of_variation.png", dpi=plot_dpi)

    ## reset the rcParams font size back to original
    plt.rcParams.update({"font.size": original_fontsize})


# noinspection PyUnboundLocalVariable
def make_figure_of_srm_and_gmcm_model_dispersions(
    locations: tuple[constants.LocationCode, ...],
    results_directory: Union[Path, str],
    plot_output_directory: Union[Path, str],
    vs30: int = 400,
    im: constants.IntensityMeasure = constants.IntensityMeasure.PGA,
    plot_title_font_size: float = 12,
    plot_dpi: int = 500,
):
    """
    Make a figure containing subplots of mean prediction on the vertical axis and the dispersion in
    predictions on the horizontal axis, following Bradley (2009).

    This figure can only be for one vs30 value and one intensity measure (im). Each column of subplots is for a
    different location so the figure will have a number of columns equal to the length of the locations list.
    The figure will always have 3 rows of subplots. The top row shows the crustal ground motion
    characterization models (GMCMs), the middle row shows the interface and intraslab GMCMs,
    and the bottom row shows the seismicity rate model (SRM) components.

    Note that the data that this function uses is loaded from the output of run_toshi_hazard_post_script.py so that needs to be run first.

    Parameters
    ----------
    locations : tuple[constants.LocationCode]
        The locations to plot.
    results_directory : Union[Path, str]
        The directory containing the data. This directory should contain subdirectories
        named as logic_tree_index_[x] where [x] is the index the logic_tree_set had in the input list.
    plot_output_directory : Union[Path, str]
        The directory where the plot will be saved.
    vs30 : int, optional
        The Vs30 value (default is 400).
    im : constants.IntensityMeasure, optional
        The intensity measure (default is constants.IntensityMeasure.PGA).
    plot_title_font_size : float, optional
        The font size of the plot titles (default is 12).
    plot_dpi : int, optional
        The resolution of the plot in dots per inch (default is 500).

    """

    if isinstance(results_directory, str):
        results_directory = Path(results_directory)
    if isinstance(plot_output_directory, str):
        plot_output_directory = Path(plot_output_directory)
    plot_output_directory.mkdir(parents=True, exist_ok=True)

    num_plot_cols = len(locations)

    locations_nloc_dict = toml.load(Path(__file__).parent.parent / "resources/location_code_to_nloc_str.toml")
    location_to_full_location = toml.load(Path(__file__).parent.parent / "resources/location_code_to_full_name.toml")
    model_to_plot_label = toml.load(Path(__file__).parent.parent / "resources/model_name_lookup_for_plot.toml")
    ground_motion_logic_tree_model_color = toml.load(Path(__file__).parent.parent / "resources/model_plot_colors.toml")

    loaded_results = (
        loading_functions.load_aggregate_stats_for_all_logic_trees_in_directory(
            results_directory
        )
    )
    data_df = loaded_results.data_df
    collated_notes_df = loaded_results.collated_notes_df

    #### Isolate the results for the ground motion models
    ## Filter to only include isolated ground motion models
    gmcm_filtered_collated_notes_df = collated_notes_df[
        collated_notes_df["ground_motion_logic_tree_note"].str.contains(
            "*", regex=False
        )
    ]
    ## In the remaining SRM component notes, filter out runs with only one of HIK or PUY subduction zone
    # (identified by "only" in the source_logic_tree_note reading "INTER_only_HIK" or "INTER_only_PUY")
    gmcm_filtered_collated_notes_df = gmcm_filtered_collated_notes_df[
        ~gmcm_filtered_collated_notes_df["source_logic_tree_note"].str.contains("only")
    ]
    sorted_gmm_logic_tree_indices = (
        plotting_utilities.sort_logic_tree_index_by_gmcm_model_name(
            gmcm_filtered_collated_notes_df
        )
    )

    #### Isolate the results for the seismicity rate model components
    ## filter to only include the SRM components
    srm_filtered_collated_notes_df = collated_notes_df[
        collated_notes_df["ground_motion_logic_tree_note"].str.contains("h.w.b. > t")
    ]

    ## In remaining SRM component models, filter out runs with only one of HIK or PUY subduction zone
    # (identified by "only" in the source_logic_tree_note reading "INTER_only_HIK" or "INTER_only_PUY")
    srm_filtered_collated_notes_df = srm_filtered_collated_notes_df[
        ~srm_filtered_collated_notes_df["source_logic_tree_note"].str.contains("only")
    ]

    ### For each row of the subplot, identify the logic_tree_indices (output directories) that will
    ### be plotted by looking at the collated_notes dataframe, and identifying the rows that contain key strings
    ### and then getting the logic tree indices (several indexes) that correspond to those rows.

    plot_row_to_logic_tree_index = {}

    for row_index in range(3):
        if row_index == 0:
            plot_row_to_logic_tree_index[row_index] = gmcm_filtered_collated_notes_df[
                gmcm_filtered_collated_notes_df["source_logic_tree_note"].str.contains(
                    "CRU"
                )
            ]["logic_tree_index"]
        if row_index == 1:
            plot_row_to_logic_tree_index[row_index] = gmcm_filtered_collated_notes_df[
                gmcm_filtered_collated_notes_df["source_logic_tree_note"].str.contains(
                    "INTER_HIK_and_PUY|SLAB"
                )
            ]["logic_tree_index"]
        if row_index == 2:
            plot_row_to_logic_tree_index[row_index] = srm_filtered_collated_notes_df[
                srm_filtered_collated_notes_df["source_logic_tree_note"].str.contains(
                    "CRU|INTER_HIK_and_PUY"
                )
            ]["logic_tree_index"]

    ####################################################

    fig, axes = plt.subplots(3, num_plot_cols, figsize=(3 * num_plot_cols, 9))

    for row_index in range(3):

        for column_index in range(num_plot_cols):

            axes[row_index, column_index].set_prop_cycle(None)

            plot_location = locations[column_index]
            if row_index == 0:
                axes[row_index, column_index].set_title(
                    location_to_full_location[plot_location],
                    fontsize=plot_title_font_size,
                )

            if row_index != 2:
                sorted_logic_tree_indices = sorted_gmm_logic_tree_indices
            if row_index == 2:
                sorted_logic_tree_indices = srm_filtered_collated_notes_df[
                    "logic_tree_index"
                ].values

            for sorted_logic_tree_index in sorted_logic_tree_indices:

                if sorted_logic_tree_index in plot_row_to_logic_tree_index[row_index]:

                    logic_tree_name_str = f"logic_tree_index_{sorted_logic_tree_index}"

                    if row_index in [0, 1]:
                        logic_tree_note = gmcm_filtered_collated_notes_df[
                            gmcm_filtered_collated_notes_df["logic_tree_index"]
                            == sorted_logic_tree_index
                        ]["ground_motion_logic_tree_note"].values[0]
                        short_note = (
                            logic_tree_note.split(">")[-2].split("*")[-2].strip(" [")
                        )
                        plot_label = model_to_plot_label[short_note]

                    if row_index == 2:
                        logic_tree_note = srm_filtered_collated_notes_df[
                            srm_filtered_collated_notes_df["logic_tree_index"]
                            == sorted_logic_tree_index
                        ]["source_logic_tree_note"].values[0]
                        short_note = (
                            logic_tree_note.split(">")[1].split(":")[-1].strip(" []")
                            + "_"
                            + logic_tree_note.split(">")[2].strip()
                        )

                        plot_label = model_to_plot_label[short_note]

                    mean = data_df[
                        (data_df["agg"] == "mean")
                        & (data_df["vs30"] == vs30)
                        & (data_df["imt"] == im)
                        & (data_df["hazard_model_id"] == logic_tree_name_str)
                        & (data_df["nloc_001"] == locations_nloc_dict[plot_location])
                    ]["values"].values[0]

                    std_ln = data_df[
                        (data_df["agg"] == "std_ln")
                        & (data_df["vs30"] == vs30)
                        & (data_df["imt"] == im)
                        & (data_df["hazard_model_id"] == logic_tree_name_str)
                        & (data_df["nloc_001"] == locations_nloc_dict[plot_location])
                    ]["values"].values[0]

                    if row_index != 2:
                        if "CRU" in logic_tree_note:
                            plot_linestyle = "--"
                        if "INTER" in logic_tree_note:
                            plot_linestyle = "--"
                        if "SLAB" in logic_tree_note:
                            plot_linestyle = "-."

                    elif row_index == 2:
                        if "CRU" in logic_tree_note:
                            plot_linestyle = "--"
                        if "INTER" in logic_tree_note:
                            plot_linestyle = "-."

                    axes[row_index, column_index].semilogy(
                        std_ln,
                        mean,
                        label=plot_label,
                        color=ground_motion_logic_tree_model_color[short_note],
                        linestyle=plot_linestyle,
                    )

                    axes[row_index, column_index].grid(
                        which="major",
                        linestyle="--",
                        linewidth="0.5",
                        color="black",
                        alpha=0.6,
                    )

                    axes[row_index, column_index].set_ylim(1e-5, 0.7)
                    axes[row_index, column_index].set_xlim(-0.01, 0.7)

                    axes[row_index, column_index].legend(
                        loc="lower left",
                        prop={"size": 6},
                        framealpha=0.4,
                        handlelength=2.2,
                        handletextpad=0.2,
                    )

                    if row_index in [0, 1]:
                        axes[row_index, column_index].set_xticklabels([])
                    if column_index > 0:
                        axes[row_index, column_index].set_yticklabels([])

                    if (row_index == 1) & (column_index == 0):
                        axes[row_index, column_index].set_ylabel(
                            r"Mean annual hazard probability, $\mu_{P(PGA=pga)}$"
                        )

    ### The text on the left and bottom of the figure require a constant margin in pixels
    ### but the figure margins need to be provided in fractions of the figure dimensions.
    ### As the figure dimensions change depending on how many columns are in the figure,
    ### the figure margins need to be calculated in pixels and then converted to fractions
    fig_margins = plotting_utilities.convert_edge_margin_in_pixels_to_fraction(
        fig, 100, 5, 45, 30
    )

    ### Adjust the figure margins before using the figure axes positions to place text
    plt.subplots_adjust(
        left=fig_margins.left,
        right=fig_margins.right,
        bottom=fig_margins.bottom,
        top=fig_margins.top,
        wspace=0.0,
        hspace=0.0,
    )

    ### Center the x-axis label differently depending on whether there are an odd or even number of columns
    if num_plot_cols % 2 != 0:  # odd number of columns
        middle_col_index = int(np.floor(num_plot_cols / 2))
        axes[2, middle_col_index].set_xlabel(
            r"Dispersion in hazard probability, $\sigma_{\ln P(PGA=pga)}$"
        )
    if num_plot_cols % 2 == 0:  # even number of columns
        # anchoring the middle of the x-axis label text to the right edge of the column calculated here
        anchor_col_index = int(num_plot_cols - num_plot_cols / 2 - 1)
        # fig.text(axes[2, anchor_col_index].get_position().x1, axes[2, anchor_col_index].get_position().y0 - 0.05, r'Dispersion in hazard probability, $\sigma_{\ln P(PGA=pga)}$', ha='center', va='center')
        fig.text(
            axes[2, anchor_col_index].get_position().x1,
            0.01,
            r"Dispersion in hazard probability, $\sigma_{\ln P(PGA=pga)}$",
            ha="center",
            va="center",
        )

    row_titles_x0 = 0.02
    fig.text(
        row_titles_x0,
        axes[0, 0].get_position().y0,
        "Ground Motion Characterization Models",
        ha="center",
        va="center",
        rotation=90,
        fontsize=plot_title_font_size,
    )

    fig.text(
        row_titles_x0,
        (axes[2, 0].get_position().y0 + axes[2, 0].get_position().y1) / 2.0,
        "Seismicity Rate Models",
        ha="center",
        va="center",
        rotation=90,
        fontsize=plot_title_font_size,
    )

    plt.savefig(
        plot_output_directory / f"{"_".join(locations)}_dispersion_poster_plot.png",
        dpi=plot_dpi,
    )


def make_figure_of_srm_model_components(
    results_directory: Union[Path, str],
    plot_output_directory: Union[Path, str],
    locations: tuple[constants.LocationCode, ...] = (
        constants.LocationCode.AKL,
        constants.LocationCode.WLG,
        constants.LocationCode.CHC,
    ),
    im: constants.IntensityMeasure = constants.IntensityMeasure.PGA,
    vs30: int = 400,
    plot_dpi: int = 500,
):
    """
    Make a figure of the dispersion in the seismicity rate model components.

    Parameters
    ----------
    results_directory : Union[Path, str]
        The directory containing the data. This directory should contain subdirectories
        named as logic_tree_index_[x] where [x] is the index the logic_tree_set had in the input list.
    plot_output_directory : Union[Path, str]
        The directory where the plot will be saved.
    locations : tuple[LocationCode], optional
        The locations to plot. Default is (LocationCode.AKL, LocationCode.WLG, LocationCode.CHC).
    im : constants.IntensityMeasure, optional
        The intensity measure (default is constants.IntensityMeasure.PGA).
    vs30 : int, optional
        The Vs30 value (default is 400).
    plot_dpi : int, optional
        The resolution of the plot in dots per inch (default is 500).
    """

    ### tectonic region type and model name lookup dictionaries
    trt_short_to_long = {"CRU": "crust", "INTER": "subduction interface\n"}

    ### model name lookup dictionary
    model_name_short_to_long = {
        "deformation_model": "deformation model\n(geologic or geodetic)",
        "time_dependence": "time dependence\n(time-dependent or time-independent)",
        "MFD": "magnitude frequency distribution",
        "moment_rate_scaling": "moment rate scaling",
    }

    locations_nloc_dict = toml.load(Path(__file__).parent.parent / "resources/location_code_to_nloc_str.toml")
    location_to_full_location = toml.load(Path(__file__).parent.parent / "resources/location_code_to_full_name.toml")

    if isinstance(results_directory, str):
        results_directory = Path(results_directory)
    if isinstance(plot_output_directory, str):
        plot_output_directory = Path(plot_output_directory)

    plot_output_directory.mkdir(parents=True, exist_ok=True)

    loaded_results = (
        loading_functions.load_aggregate_stats_for_all_logic_trees_in_directory(
            results_directory
        )
    )

    logic_tree_name_notes_df = loaded_results.collated_notes_df
    data_df = loaded_results.data_df

    ## filter to only include the SRM components
    filtered_logic_tree_name_notes_df = logic_tree_name_notes_df[
        logic_tree_name_notes_df["ground_motion_logic_tree_note"].str.contains(
            "h.w.b. > t"
        )
    ]

    ## In remaining SRM component models, filter out runs with only one of HIK or PUY subduction zone
    # (identified by "only" in the source_logic_tree_note reading "INTER_only_HIK" or "INTER_only_PUY")
    filtered_logic_tree_name_notes_df = filtered_logic_tree_name_notes_df[
        ~filtered_logic_tree_name_notes_df["source_logic_tree_note"].str.contains(
            "only"
        )
    ]

    logic_tree_name_strs = [
        f"logic_tree_index_{x}"
        for x in filtered_logic_tree_name_notes_df["logic_tree_index"].values
    ]

    _, axes = plt.subplots(1, len(locations), figsize=(2.7 * len(locations), 4))

    linestyle_lookup_dict = {"CRU": "-", "INTER": "--"}

    for location_idx, location in enumerate(locations):

        for logic_tree_name in logic_tree_name_strs:

            # noinspection DuplicatedCode
            nloc_001_str = locations_nloc_dict[location]

            mean = data_df[
                (data_df["agg"] == "mean")
                & (data_df["vs30"] == vs30)
                & (data_df["imt"] == im)
                & (data_df["hazard_model_id"] == logic_tree_name)
                & (data_df["nloc_001"] == nloc_001_str)
            ]["values"].values[0]

            std_ln = data_df[
                (data_df["agg"] == "std_ln")
                & (data_df["vs30"] == vs30)
                & (data_df["imt"] == im)
                & (data_df["hazard_model_id"] == logic_tree_name)
                & (data_df["nloc_001"] == nloc_001_str)
            ]["values"].values[0]

            needed_idx = mean > 1e-8
            mean = mean[needed_idx]
            std_ln = std_ln[needed_idx]

            source_logic_tree_note = f"{logic_tree_name_notes_df[logic_tree_name_notes_df["logic_tree_index"] == int(logic_tree_name.split("_")[-1])]["source_logic_tree_note"].values[0]}"

            tectonic_region_type = (
                source_logic_tree_note.split(">")[1].strip(" ]'").split("[")[-1]
            )

            if "INTER" in tectonic_region_type:
                tectonic_region_type = "INTER"

            model_name = source_logic_tree_note.split(">")[-2].strip(" ")

            note = f"{trt_short_to_long[tectonic_region_type]} {model_name_short_to_long[model_name]}"

            axes[location_idx].semilogy(
                std_ln,
                mean,
                label=note,
                linestyle=linestyle_lookup_dict[tectonic_region_type],
            )

            axes[location_idx].set_ylim(1e-6, 0.6)
            axes[location_idx].set_xlim(-0.01, 0.37)
            axes[location_idx].set_title(location_to_full_location[location])

            if location_idx == 0:
                axes[location_idx].set_ylabel(
                    r"Mean annual hazard probability, $\mu_{P(IM=im)}$"
                )

            if location_idx == 1:
                axes[location_idx].set_xlabel(
                    r"Dispersion in hazard probability, $\sigma_{\ln P(IM=im)}$"
                )
                axes[location_idx].set_yticklabels([])
            if location_idx == 2:
                axes[location_idx].set_yticklabels([])

        axes[location_idx].grid(
            which="major", linestyle="--", linewidth="0.5", color="black"
        )

        axes[location_idx].legend(
            loc="lower left",
            prop={"size": 6},
            framealpha=0.6,
            handlelength=2.2,
            handletextpad=0.2,
        )

    plt.subplots_adjust(wspace=0.0, left=0.1, right=0.99, top=0.94)
    plt.savefig(
        plot_output_directory / f"srm_dispersions_{"_".join(locations)}.png",
        dpi=plot_dpi,
    )


def make_figure_of_gmcms(
    results_directory: Union[Path, str],
    plot_output_directory: Union[Path, str],
    locations: tuple[constants.LocationCode, ...] = (
        constants.LocationCode.AKL,
        constants.LocationCode.WLG,
        constants.LocationCode.CHC,
    ),
    vs30: int = 400,
    im: constants.IntensityMeasure = constants.IntensityMeasure.PGA,
    plot_dpi: int = 500,
):
    """
    Generate a figure of ground motion characterization models (GMCMs) for specified locations.

    Parameters
    ----------
    results_directory : Union[Path, str]
        The directory containing the results to plot.
    plot_output_directory : Union[Path, str]
        The directory where the plot will be saved.
    locations : tuple[constants.LocationCode], optional
        The locations to plot. Default is (constants.LocationCode.AKL,
                                           constants.LocationCode.WLG,
                                           constants.LocationCode.CHC).
    vs30 : int, optional
        The Vs30 value to use in the plot. Default is 400.
    im : constants.IntensityMeasure, optional
        The intensity measure to use in the plot. Default is constants.IntensityMeasure.PGA.
    plot_dpi : int, optional
        The resolution of the plot in dots per inch. Default is 500.

    Returns
    -------
    None
    """

    locations_nloc_dict = toml.load(Path(__file__).parent.parent / "resources/location_code_to_nloc_str.toml")
    tectonic_type_to_linestyle = toml.load(
        Path(__file__).parent.parent / "resources/tectonic_region_type_to_linestyle.toml"
    )
    location_to_full_location = toml.load(Path(__file__).parent.parent / "resources/location_code_to_full_name.toml")
    model_to_plot_label = toml.load(Path(__file__).parent.parent / "resources/model_name_lookup_for_plot.toml")
    ground_motion_logic_tree_model_color = toml.load(Path(__file__).parent.parent / "resources/model_plot_colors.toml")

    if isinstance(results_directory, str):
        results_directory = Path(results_directory)

    if isinstance(plot_output_directory, str):
        plot_output_directory = Path(plot_output_directory)

    plot_output_directory.mkdir(parents=True, exist_ok=True)

    loaded_results = (
        loading_functions.load_aggregate_stats_for_all_logic_trees_in_directory(
            results_directory, locations
        )
    )
    # noinspection DuplicatedCode
    data_df = loaded_results.data_df
    collated_notes_df = loaded_results.collated_notes_df

    ## Filter to only include isolated ground motion models
    filtered_collated_notes_df = collated_notes_df[
        collated_notes_df["ground_motion_logic_tree_note"].str.contains(
            "*", regex=False
        )
    ]

    ## In the remaining SRM component notes, filter out runs with only one of HIK or PUY subduction zone
    # (identified by "only" in the source_logic_tree_note reading "INTER_only_HIK" or "INTER_only_PUY")
    filtered_collated_notes_df = filtered_collated_notes_df[
        ~filtered_collated_notes_df["source_logic_tree_note"].str.contains("only")
    ]

    sorted_logic_tree_indices = (
        plotting_utilities.sort_logic_tree_index_by_gmcm_model_name(
            filtered_collated_notes_df
        )
    )

    _, axes = plt.subplots(3, 3, figsize=(6, 9))

    for location_row_idx, location in enumerate(locations):

        nloc_001_str = locations_nloc_dict[location]

        mean_list = []
        std_ln_list = []
        non_zero_run_list = []

        for logic_tree_index in sorted_logic_tree_indices:

            logic_tree_name_str = f"logic_tree_index_{logic_tree_index}"

            # noinspection DuplicatedCode
            mean = data_df[
                (data_df["agg"] == "mean")
                & (data_df["vs30"] == vs30)
                & (data_df["imt"] == im)
                & (data_df["hazard_model_id"] == logic_tree_name_str)
                & (data_df["nloc_001"] == nloc_001_str)
            ]["values"].values[0]

            mean_max = np.max(mean)
            print(f"logic_tree_name_str {logic_tree_name_str} max mean: {mean_max}")

            std_ln = data_df[
                (data_df["agg"] == "std_ln")
                & (data_df["vs30"] == vs30)
                & (data_df["imt"] == im)
                & (data_df["hazard_model_id"] == logic_tree_name_str)
                & (data_df["nloc_001"] == nloc_001_str)
            ]["values"].values[0]

            mean_list.append(mean)
            std_ln_list.append(std_ln)
            non_zero_run_list.append(logic_tree_name_str)
            # noinspection DuplicatedCode
            source_logic_tree_note = f"{collated_notes_df[collated_notes_df["logic_tree_index"] == logic_tree_index]["source_logic_tree_note"].values[0]}"
            ground_motion_logic_tree_note = f"{collated_notes_df[collated_notes_df["logic_tree_index"]==logic_tree_index]["ground_motion_logic_tree_note"].values[0]}"

            tectonic_region_type_group_from_note = (
                source_logic_tree_note.split(">")[-2].strip().split(":")[-1].strip("[]")
            )
            ground_motion_logic_tree_model_and_weight_str = (
                ground_motion_logic_tree_note.split(">")[-2].strip(" []")
            )
            ground_motion_logic_tree_model = (
                ground_motion_logic_tree_model_and_weight_str.split("*")[0]
            )

            linestyle = tectonic_type_to_linestyle[tectonic_region_type_group_from_note]

            if tectonic_region_type_group_from_note == "INTER_only_HIK":
                continue
            if tectonic_region_type_group_from_note == "INTER_only_PUY":
                continue

            if "CRU" in tectonic_region_type_group_from_note:
                subplot_idx = 0
            if "INTER" in tectonic_region_type_group_from_note:
                subplot_idx = 1
            if "SLAB" in tectonic_region_type_group_from_note:
                subplot_idx = 2

            # noinspection PyUnboundLocalVariable
            axes[location_row_idx, subplot_idx].semilogy(
                std_ln,
                mean,
                label=model_to_plot_label[ground_motion_logic_tree_model],
                linestyle=linestyle,
                color=ground_motion_logic_tree_model_color[
                    ground_motion_logic_tree_model
                ],
            )

            axes[location_row_idx, subplot_idx].text(
                x=0.68,
                y=0.2,
                s=location_to_full_location[location],
                horizontalalignment="right",
                bbox=dict(facecolor="white", alpha=0.4, edgecolor="none", pad=0),
            )

            axes[location_row_idx, subplot_idx].set_ylim(1e-5, 0.6)
            axes[location_row_idx, subplot_idx].set_xlim(-0.01, 0.7)
            axes[location_row_idx, subplot_idx].grid(
                which="major", linestyle="--", linewidth="0.5", color="black", alpha=0.5
            )

            if subplot_idx == 0:
                axes[0, 0].set_title("Active shallow crust", fontsize=11)
                axes[0, 1].set_title("Subduction interface", fontsize=11)
                axes[0, 2].set_title("Subduction intraslab", fontsize=11)

                if location_row_idx == 1:
                    axes[location_row_idx, subplot_idx].set_ylabel(
                        r"Mean annual hazard probability, $\mu_{P(IM=im)}$"
                    )

            if subplot_idx == 1:
                if location_row_idx == 2:
                    axes[location_row_idx, subplot_idx].set_xlabel(
                        r"Dispersion in hazard probability, $\sigma_{\ln P(IM=im)}$"
                    )
                axes[location_row_idx, subplot_idx].set_yticklabels([])

            if subplot_idx == 2:
                axes[location_row_idx, subplot_idx].set_yticklabels([])

            if (location_row_idx == 0) or (location_row_idx == 1):
                axes[location_row_idx, subplot_idx].set_xticklabels([])

            axes[location_row_idx, subplot_idx].legend(
                loc="lower left",
                prop={"size": 6},
                framealpha=0.4,
                handlelength=2.2,
                handletextpad=0.2,
            )

    plt.subplots_adjust(
        wspace=0.0, hspace=0.0, left=0.11, right=0.99, bottom=0.05, top=0.97
    )

    plt.savefig(
        plot_output_directory / f"gmcms_{im}_{"_".join(locations)}.png", dpi=plot_dpi
    )


def make_figure_showing_bradley2009_method(
    results_directory: Union[Path, str],
    plot_output_directory: Union[Path, str],
    registry_directory: Union[Path, str],
    location_short_name: constants.LocationCode = constants.LocationCode.WLG,
    vs30: int = 400,
    im: constants.IntensityMeasure = constants.IntensityMeasure.PGA,
    plot_dpi: int = 500,
):
    """
    For a given model, make a figure showing the predictions of the individual realizations and the
    epistemic dispersion as in Bradley (2009).

    Parameters
    ----------
    results_directory : Union[Path, str]
        The directory containing the data. This directory should contain subdirectories
        named as logic_tree_index_[x] where [x] is the index the logic_tree_set had in the input list.
    plot_output_directory : Union[Path, str]
        The directory where the plot will be saved.
    registry_directory : Union[Path, str]
        The directory containing the branch registry files that come with the GNS package nshm-model.
    location_short_name : constants.LocationCode, optional
        The location code (default is constants.LocationCode.WLG).
    vs30 : int, optional
        The Vs30 value (default is 400).
    im : constants.IntensityMeasure, optional
        The intensity measure (default is constants.IntensityMeasure.PGA).
    plot_dpi : int, optional
        The resolution of the plot in dots per inch (default is 500).

    Returns
    -------
    None
    """

    if isinstance(results_directory, str):
        results_directory = Path(results_directory)

    if isinstance(plot_output_directory, str):
        plot_output_directory = Path(plot_output_directory)

    if isinstance(registry_directory, str):
        registry_directory = Path(registry_directory)

    locations_nloc_dict = toml.load(Path(__file__).parent.parent / "resources/location_code_to_nloc_str.toml")
    nshm_im_levels = np.loadtxt(Path(__file__).parent.parent / "resources/nshm_im_levels.txt")

    plot_colors = ["tab:purple", "tab:orange", "tab:green"]
    plot_linestyles = [":", "-", "--"]

    plot_label_lookup = {
        "Stafford2022(mu_branch=Upper)": "Upper branch of Stafford 2022",
        "Stafford2022(mu_branch=Central)": "Central branch of Stafford 2022",
        "Stafford2022(mu_branch=Lower)": "Lower branch of Stafford 2022",
    }

    gmcm_name_formatting_lookup = {"Stafford2022": "Stafford (2022)"}

    individual_realization_df = (
        ds.dataset(
            source=results_directory / "individual_realizations", format="parquet"
        )
        .to_table()
        .to_pandas()
    )

    individual_realizations_needed_indices = (
        individual_realization_df["hazard_model_id"] == results_directory.name
    ) & (
        individual_realization_df["nloc_001"]
        == locations_nloc_dict[location_short_name]
    )

    filtered_individual_realization_df = individual_realization_df[
        individual_realizations_needed_indices
    ]

    realization_names = loading_functions.lookup_realization_name_from_hash(
        filtered_individual_realization_df, registry_directory
    )

    hazard_rate_array = np.zeros((len(filtered_individual_realization_df), 44))

    for realization_index in range(len(filtered_individual_realization_df)):
        hazard_rate_array[realization_index, :] = (
            filtered_individual_realization_df.iloc[realization_index][
                "branches_hazard_rates"
            ]
        )

    ### Convert the rate to annual probability of exceedance
    hazard_prob_of_exceedance = calculators.rate_to_prob(hazard_rate_array, 1.0)

    resulting_hazard_curves = (
        loading_functions.load_aggregate_stats_for_all_logic_trees_in_directory(
            results_directory.parent
        )
    )

    agg_stats_df = resulting_hazard_curves.data_df

    ## Get the needed mean and standard deviation values
    mean = agg_stats_df[
        (agg_stats_df["agg"] == "mean")
        & (agg_stats_df["vs30"] == vs30)
        & (agg_stats_df["imt"] == im)
        & (agg_stats_df["hazard_model_id"] == results_directory.name)
        & (agg_stats_df["nloc_001"] == locations_nloc_dict[location_short_name])
    ]["values"].values[0]

    std_ln = agg_stats_df[
        (agg_stats_df["agg"] == "std_ln")
        & (agg_stats_df["vs30"] == vs30)
        & (agg_stats_df["imt"] == im)
        & (agg_stats_df["hazard_model_id"] == results_directory.name)
        & (agg_stats_df["nloc_001"] == locations_nloc_dict[location_short_name])
    ]["values"].values[0]

    plot_ylims = (1e-5, 1)
    _, axes = plt.subplots(1, 2, figsize=(8, 5))

    for realization_index in range(len(hazard_prob_of_exceedance)):
        gmcm_name_with_branch = realization_names[
            realization_index
        ].ground_motion_characterization_models_id
        gmcm_name = gmcm_name_with_branch.split("(")[0]

        if gmcm_name_with_branch in plot_label_lookup.keys():
            plot_label = plot_label_lookup[gmcm_name_with_branch]
        else:
            plot_label = gmcm_name_with_branch

        axes[0].loglog(
            nshm_im_levels,
            hazard_prob_of_exceedance[realization_index],
            label=plot_label,
            marker="o",
            color=plot_colors[realization_index],
            linestyle=plot_linestyles[realization_index],
        )

    # Customize the second subplot
    # noinspection PyUnboundLocalVariable
    if gmcm_name in gmcm_name_formatting_lookup.keys():
        gmcm_name_label = gmcm_name_formatting_lookup[gmcm_name]
    else:
        gmcm_name_label = gmcm_name

    axes[1].semilogy(
        std_ln, mean, marker="o", linestyle="-", color="tab:blue", label=gmcm_name_label
    )

    axes[0].legend(loc="lower left")
    axes[1].legend(loc="lower left")

    for ax_index in range(len(axes)):
        axes[ax_index].grid(
            which="major", linestyle="--", linewidth="0.5", color="black", alpha=0.5
        )

        axes[ax_index].set_ylim(plot_ylims)
    axes[0].set_xlim(9e-5, 5)
    axes[1].set_xlim(-0.01, 0.68)

    ### The annotations for explanation
    annotation_ims = [1e-2, 1e-1, 1e0]

    ### latex formatted strings corresponding to the annotation_ims.
    ### These could be generated automatically from the annotation_ims
    ### but they are just hard coded for simplicity.
    manually_matched_latex_strings = ["$10^{-2}$", "$10^{-1}$", "$10^{0}$"]
    annotation_labels = ["A", "B", "C"]

    for annotation_im in annotation_ims:

        im_index = np.where(nshm_im_levels == annotation_im)[0][0]

        print(
            f"im index = {im_index}, im value = {annotation_im}, mean = {mean[im_index]:.1e}"
        )

        ### Draw vertical lines at IM values
        axes[0].hlines(
            y=mean[im_index],
            xmin=nshm_im_levels[0] / 10,
            xmax=annotation_im,
            color="black",
            linestyle="--",
        )
        axes[0].vlines(
            x=annotation_im,
            ymin=plot_ylims[0],
            ymax=mean[im_index],
            color="black",
            linestyle="--",
        )
        axes[0].vlines(
            x=annotation_im,
            ymin=mean[im_index],
            ymax=plot_ylims[1],
            color="tab:blue",
            linestyle="--",
        )

        ### Draw the arrows
        axes[0].plot(
            annotation_im,
            plot_ylims[1] - 0.45,
            marker=r"$\uparrow$",
            markersize=20,
            color="tab:blue",
        )

        ### Write the standard deviation value
        axes[0].text(
            annotation_im,
            plot_ylims[1] + 0.5,
            f"{std_ln[im_index]:.2f}",
            ha="center",
            va="center",
            color="black",
        )

        ### Plot labels (A), (B), (C)
        axes[0].text(
            annotation_im,
            plot_ylims[1] + 2.0,
            f"{manually_matched_latex_strings[annotation_ims.index(annotation_im)]}",
            ha="center",
            va="center",
            color="black",
        )

        ### Draw the horizontal lines at the mean values
        axes[1].hlines(
            y=mean[im_index],
            xmin=-0.01,
            xmax=std_ln[im_index],
            color="black",
            linestyle="--",
        )

        ### Draw the vertical lines at the standard deviation values
        axes[1].vlines(
            x=std_ln[im_index],
            ymin=plot_ylims[0],
            ymax=plot_ylims[1],
            color="black",
            linestyle="--",
        )

        ### Plot labels (A), (B), (C)
        axes[1].text(
            std_ln[im_index],
            plot_ylims[1] + 0.5,
            f"({annotation_labels[annotation_ims.index(annotation_im)]})",
            ha="center",
            va="center",
            color="black",
        )

    axes[0].set_xlabel(r"Peak ground acceleration (g)")
    axes[0].set_ylabel(r"Annual hazard probability, $\mu_{P(PGA=pga)}$")

    axes[1].set_ylabel(r"Mean annual hazard probability, $\mu_{P(PGA=pga)}$")
    axes[1].set_xlabel(r"Dispersion in hazard probability, $\sigma_{\ln P(PGA=pga)}$")

    axes[0].text(
        1.25e-4,
        12,
        "Dispersion in hazard probability",
        ha="left",
        va="center",
        color="black",
    )

    text_row_2_height = plot_ylims[1] + 5.0

    axes[0].text(
        6e-3,
        text_row_2_height,
        "Reference point:",
        ha="right",
        va="center",
        color="black",
    )

    axes[0].text(
        annotation_ims[0],
        text_row_2_height,
        "(A)",
        ha="center",
        va="center",
        color="black",
    )

    axes[0].text(
        annotation_ims[1],
        text_row_2_height,
        "(B)",
        ha="center",
        va="center",
        color="black",
    )

    axes[0].text(
        annotation_ims[2],
        text_row_2_height,
        "(C)",
        ha="center",
        va="center",
        color="black",
    )

    axes[0].text(
        8e-3, plot_ylims[1] + 2.0, f"{im} =  ", ha="right", va="center", color="black"
    )

    axes[0].text(
        5.8e-3,
        plot_ylims[1] + 0.5,
        rf"$\sigma_{{\ln P({im.upper()}={im.lower()})}} = $",
        ha="right",
        va="center",
        color="black",
    )

    plt.subplots_adjust(bottom=0.1, top=0.81, left=0.085, right=0.99, wspace=0.23)

    plt.savefig(
        plot_output_directory
        / f"{gmcm_name}_{location_short_name}_predictions_and_aggregate_stats.png",
        dpi=plot_dpi,
    )


def make_figure_of_gmm_dispersion_ranges(
    results_directory: Union[Path, str],
    plot_output_directory: Union[Path, str],
    locations: tuple[constants.LocationCode, ...] = (
        constants.LocationCode.AKL,
        constants.LocationCode.WLG,
        constants.LocationCode.CHC,
    ),
    filter_strs: tuple[str, ...] = ("CRU", "HIK_and_PUY", "SLAB"),
    vs30: int = 400,
    im: constants.IntensityMeasure = constants.IntensityMeasure.PGA,
    plot_dpi: int = 500,
    num_interp_mean_points: int = 1000,
    min_log10_mean_for_interp: int = -6,
    max_log10_mean_for_interp: int = -2,
    plot_interpolations: bool = False,
    min_mean_value_for_interp_plots: float = 1e-9,
):
    """
    Generate a figure showing the dispersion ranges of ground motion models (GMMs) for specified locations.

    Parameters
    ----------
    results_directory : Union[Path, str]
        The directory containing the results data.
    plot_output_directory : Union[Path, str]
        The directory where the plot will be saved.
    locations : tuple[constants.LocationCode], optional
        The locations to plot. Default is (constants.LocationCode.AKL,
                                           constants.LocationCode.WLG,
                                           constants.LocationCode.CHC).
    filter_strs : tuple[str], optional
        The filter strings needed to select the desired data. Default is ("CRU", "HIK_and_PUY", "SLAB").
    vs30 : int, optional
        The Vs30 value to use in the plot. Default is 400.
    im : constants.IntensityMeasure, optional
        The intensity measure to use in the plot. Default is constants.IntensityMeasure.PGA.
    plot_dpi : int, optional
        The resolution of the plot in dots per inch. Default is 500.
    num_interp_mean_points : int, optional
        The number of interpolation points for the mean. Default is 1000.
    min_log10_mean_for_interp : int, optional
        The minimum log10 mean value for interpolation. Default is -6.
    max_log10_mean_for_interp : int, optional
        The maximum log10 mean value for interpolation. Default is -2.
    plot_interpolations : bool, optional
        Whether to plot interpolations. Default is False.
    min_mean_value_for_interp_plots : float, optional
        The minimum mean value for interpolation plots. Default is 1e-9.

    Returns
    -------
    None
    """

    if isinstance(results_directory, str):
        results_directory = Path(results_directory)
    if isinstance(plot_output_directory, str):
        plot_output_directory = Path(plot_output_directory)
    plot_output_directory.mkdir(parents=True, exist_ok=True)

    dispersion_range_dict = plotting_utilities.get_interpolated_gmms(
        results_directory=results_directory,
        locations=locations,
        filter_strs=filter_strs,
        vs30=vs30,
        im=im,
        num_interp_mean_points=num_interp_mean_points,
        min_log10_mean_for_interp=min_log10_mean_for_interp,
        max_log10_mean_for_interp=max_log10_mean_for_interp,
        plot_interpolations=plot_interpolations,
        min_mean_value_for_interp_plots=min_mean_value_for_interp_plots,
    )

    linestyle_lookup_dict = {"CRU": "--", "HIK_and_PUY": "-.", "SLAB": ":"}

    color_lookup_dict = {"AKL": "blue", "WLG": "orange", "CHC": "red"}

    plt.figure()

    mean_interpolation_points = np.logspace(
        min_log10_mean_for_interp, max_log10_mean_for_interp, num_interp_mean_points
    )

    for location in locations:

        for filter_str in filter_strs:

            if filter_str == "HIK_and_PUY":

                plt.semilogy(
                    dispersion_range_dict[location][filter_str],
                    mean_interpolation_points,
                    label=f"{location} INTER",
                    linestyle=linestyle_lookup_dict[filter_str],
                    color=color_lookup_dict[location],
                )

            else:

                plt.semilogy(
                    dispersion_range_dict[location][filter_str],
                    mean_interpolation_points,
                    label=f"{location} {filter_str}",
                    linestyle=linestyle_lookup_dict[filter_str],
                    color=color_lookup_dict[location],
                )

    plt.legend()
    plt.ylabel(
        rf"Mean annual hazard probability, $\mu_{{P({im.upper()}={im.lower()})}}$"
    )
    plt.xlabel(
        rf"Range in dispersion in hazard probability, $\sigma_{{\ln P({im.upper()}={im.lower()})}}$"
    )
    plt.grid(linestyle="--")
    plt.subplots_adjust(right=0.99, top=0.98)
    plt.savefig(
        plot_output_directory / "gmcm_models_dispersion_ranges.png",
        dpi=plot_dpi,
    )


def make_figures_of_individual_realizations_for_a_single_logic_tree(
    logic_tree_index_dir: Union[Path, str],
    plot_output_directory: Union[Path, str],
    locations: tuple[constants.LocationCode, ...] = (
        constants.LocationCode.AKL,
        constants.LocationCode.WLG,
        constants.LocationCode.CHC,
    ),
    im: constants.IntensityMeasure = constants.IntensityMeasure.PGA,
    vs30: int = 400,
    im_xlims: tuple = (9e-5, 5),
    poe_min_plot: float = 1e-5,
    ybuffer_absmax_over_val: float = 10.0,
    plot_dpi: int = 500,
):
    """
    Generate figures of individual realizations for a single logic tree.

    This function processes the individual realizations of a given logic tree and generates
    plots for specified locations, intensity measure, and Vs30 value.

    Parameters
    ----------
    logic_tree_index_dir : Union[Path, str]
        Directory containing the logic tree index.
    plot_output_directory : Union[Path, str]
        Directory where the generated plots will be saved.
    locations : tuple[constants.LocationCode], optional
        Locations to generate plots for. Default is (constants.LocationCode.AKL,
                                                     constants.LocationCode.WLG,
                                                     constants.LocationCode.CHC).
    im : constants.IntensityMeasure, optional
        Intensity measure to be used in the plots. Default is constants.IntensityMeasure.PGA.
    vs30 : int, optional
        Vs30 value to be used in the plots. Default is 400.
    im_xlims : tuple, optional
        X-axis limits for the intensity measure. Default is (9e-5, 5).
    poe_min_plot : float, optional
        Minimum probability of exceedance to be plotted. Default is 1e-5.
    ybuffer_absmax_over_val : float, optional
        Defines how much to extend the y-axis limits above the maximum value in the plot (for a buffer).
        The amount of extension is defined as the maximum plotted value divided by this value. Default is 10.0.
    plot_dpi : int, optional
        DPI for the generated plots. Default is 500.

    Returns
    -------
    None
    """

    locations_nloc_dict = toml.load(Path(__file__).parent.parent / "resources/location_code_to_nloc_str.toml")
    model_name_to_plot_format = toml.load(Path(__file__).parent.parent / "resources/model_name_lookup_for_plot.toml")
    nshm_im_levels = np.loadtxt(Path(__file__).parent.parent / "resources/nshm_im_levels.txt")

    needed_im_level_indices = np.where(
        (nshm_im_levels >= im_xlims[0]) & (nshm_im_levels <= im_xlims[1])
    )[0]

    if isinstance(logic_tree_index_dir, str):
        logic_tree_index_dir = Path(logic_tree_index_dir)
    if isinstance(plot_output_directory, str):
        plot_output_directory = Path(plot_output_directory)

    if not plot_output_directory.exists():
        plot_output_directory.mkdir(parents=True)

    individual_realization_df = (
        ds.dataset(
            source=logic_tree_index_dir / "individual_realizations", format="parquet"
        )
        .to_table()
        .to_pandas()
    )

    output_notes = toml.load(logic_tree_index_dir / "notes.toml")

    if (
        output_notes["source_logic_tree_note"] == "full > "
        and output_notes["ground_motion_logic_tree_note"] == "full > "
    ):
        model_name_short = "full"
        model_name_long = model_name_short

    if (
        output_notes["source_logic_tree_note"] == "full > 1 (nth) h.w.b. > "
        and output_notes["source_logic_tree_note"] == "full > "
    ):
        model_name_short = "all_srm"
        model_name_long = model_name_short

    if (
        output_notes["source_logic_tree_note"] == "full > "
        and output_notes["source_logic_tree_note"] == "full > 1 (nth) h.w.b. > "
    ):
        model_name_short = "all_gmcm"
        model_name_long = model_name_short

    ### The GMCM logic tree was reduced to the single highest weighted branch so we can use the SRM model components
    if "1 (nth) h.w.b." in output_notes["ground_motion_logic_tree_note"]:

        ### Extract the short source model name from its position between the last two ">" characters
        ### in a source_logic_tree_note such as 'full > tectonic_region_type_group:[CRU] > deformation_model > '
        # Using re.VERBOSE to allow comments in the pattern string.
        pattern = r"""
                  >[ ]               # Search for the ">" character followed by a space ([ ])
                  (?P<group_id>\w+)  # Followed by one or more alphanumeric characters (\w+) and capture them 
                                     # in a group called group_id (?P<group_id>...)                  
                  [ ]>[ ]$           # Followed by a space ([ ]) and then a ">" and then a space ([ ]) and then the end of the string ($)
                  """
        last_part = re.search(
            pattern, output_notes["source_logic_tree_note"], re.VERBOSE
        ).group("group_id")

        ### Extract the short tectonic region type from its position between the ":[" and "]" characters in a
        # source_logic_tree_note such as 'full > tectonic_region_type_group:[CRU] > deformation_model > '
        pattern = r"""                    
                    :\[                 # Search for the ":" character followed by a "[" character (:\[)
                     (?P<group_id>\w+)  # Followed by one or more alphanumeric characters (\w+) and capture them 
                                        # in a group called group_id (?P<group_id>...)      
                    \]                  # Followed by a "]" character (\]) 
                    """
        tectonic_region_type_part = re.search(
            pattern, output_notes["source_logic_tree_note"], re.VERBOSE
        ).group("group_id")

        model_name_short = f"{tectonic_region_type_part}_{last_part}"
        model_name_long = model_name_to_plot_format[model_name_short]

    ### The source logic tree was reduced to the single highest weighted branch so we can use the gmcm models
    if "1 (nth) h.w.b." in output_notes["source_logic_tree_note"]:

        ### Extract the ground model name from its position between the second "[" character and the "*" character.
        ### in a ground_motion_logic_tree_note such as  'full > tectonic_region_type_group:[CRU] > [Bradley2013*15.15] > '
        pattern = r"""
                  \[                    # Search for the "[" character
                     (?P<group_id>\w+)  # Followed by one or more alphanumeric characters (\w+) and capture them 
                                        # in a group called group_id (?P<group_id>...)      
                  \*                    # Followed by a "*" character
                  """
        model_name_short = re.search(
            pattern, output_notes["ground_motion_logic_tree_note"], re.VERBOSE
        ).group("group_id")
        model_name_long = model_name_to_plot_format[model_name_short]

    _, axes = plt.subplots(2, len(locations), figsize=(3 * len(locations), 6))

    poe_maxs = []
    ln_resid_mins = []
    ln_resid_maxs = []

    for location_idx, location in enumerate(locations):

        nloc_001_str = locations_nloc_dict[location]

        individual_realizations_needed_indices = (
            (individual_realization_df["hazard_model_id"] == logic_tree_index_dir.name)
            & (individual_realization_df["nloc_001"] == nloc_001_str)
            & (individual_realization_df["vs30"] == vs30)
            & (individual_realization_df["imt"] == im)
        )

        filtered_individual_realization_df = individual_realization_df[
            individual_realizations_needed_indices
        ]

        hazard_rate_array = np.zeros((len(filtered_individual_realization_df), 44))
        for realization_index in range(len(filtered_individual_realization_df)):
            hazard_rate_array[realization_index, :] = (
                filtered_individual_realization_df.iloc[realization_index][
                    "branches_hazard_rates"
                ]
            )

        ### Convert the rate to annual probability of exceedance
        hazard_prob_of_exceedance = calculators.rate_to_prob(hazard_rate_array, 1.0)

        ln_resid_poe = np.log(hazard_prob_of_exceedance) - np.log(
            hazard_prob_of_exceedance[0]
        )

        ### Plot the individual realizations in a loop
        for realization_index in range(len(filtered_individual_realization_df)):

            poe_maxs.append(
                np.nanmax(
                    hazard_prob_of_exceedance[realization_index][
                        needed_im_level_indices
                    ]
                )
            )
            # noinspection PyUnboundLocalVariable
            axes[0, location_idx].loglog(
                nshm_im_levels[needed_im_level_indices],
                hazard_prob_of_exceedance[realization_index][needed_im_level_indices],
                label=model_name_short,
            )

            ln_resid_mins.append(
                np.nanmin(ln_resid_poe[realization_index][needed_im_level_indices])
            )
            ln_resid_maxs.append(
                np.nanmax(ln_resid_poe[realization_index][needed_im_level_indices])
            )
            axes[1, location_idx].semilogx(
                nshm_im_levels[needed_im_level_indices],
                ln_resid_poe[realization_index][needed_im_level_indices],
                label=model_name_short,
            )

        axes[0, location_idx].set_xlim(im_xlims)
        axes[0, location_idx].grid(
            which="major", linestyle="--", linewidth="0.5", color="black", alpha=0.5
        )

        axes[0, location_idx].set_title(location)
        axes[0, location_idx].legend(loc="lower left", prop={"size": 5})

        axes[1, location_idx].set_xlim(im_xlims)

        axes[1, location_idx].grid(
            which="major", linestyle="--", linewidth="0.5", color="black", alpha=0.5
        )

        axes[1, location_idx].legend(loc="lower left", prop={"size": 5})

        if location_idx > 0:
            axes[0, location_idx].set_yticklabels([])
            axes[1, location_idx].set_yticklabels([])
        axes[0, location_idx].set_xticklabels([])

    ### Set all the y-axis limits to the max values found in the last loop over locations
    ln_resid_mins = np.array(ln_resid_mins)
    ln_resid_maxs = np.array(ln_resid_maxs)
    poe_maxs = np.array(poe_maxs)

    finite_ln_resid_mins = ln_resid_mins[np.isfinite(ln_resid_mins)]
    finite_ln_resid_maxs = ln_resid_maxs[np.isfinite(ln_resid_maxs)]

    abs_max = np.nanmax(np.hstack((np.abs(finite_ln_resid_mins), finite_ln_resid_maxs)))

    for location_idx in range(len(locations)):
        axes[0, location_idx].set_ylim(
            poe_min_plot,
            np.nanmax(poe_maxs) + np.nanmax(poe_maxs) / ybuffer_absmax_over_val,
        )
        axes[1, location_idx].set_ylim(
            np.min(finite_ln_resid_mins) - abs_max / ybuffer_absmax_over_val,
            np.max(finite_ln_resid_maxs) + abs_max / ybuffer_absmax_over_val,
        )

    axes[0, 0].set_ylabel("Annual probability of exceedance")
    axes[1, 0].set_ylabel(r"$\ln$(APoE$_n$)-$\ln$(APoE$_0$)")

    axes[1, 1].set_xlabel(f"{im} level")

    # noinspection PyUnboundLocalVariable
    plt.suptitle(model_name_long)

    plt.subplots_adjust(left=0.10, right=0.99, bottom=0.1, wspace=0.0, hspace=0.0)

    plt.savefig(
        plot_output_directory / f"{model_name_short}_individual_realizations.png",
        dpi=plot_dpi,
    )


def make_figures_of_several_individual_realizations(
    results_directory: Union[Path, str],
    plot_output_directory: Union[Path, str],
    locations: tuple[constants.LocationCode, ...] = (
        constants.LocationCode.AKL,
        constants.LocationCode.WLG,
        constants.LocationCode.CHC,
    ),
    im: constants.IntensityMeasure = constants.IntensityMeasure.PGA,
    vs30: int = 400,
    im_xlims: tuple = (9e-5, 5),
    poe_min_plot: float = 1e-5,
    ybuffer_absmax_over_val: float = 10.0,
    selected_subduction_interface: constants.InterfaceName = constants.InterfaceName.HIK_and_PUY,
    plot_dpi: int = 500,
    notes_to_exclude: Union[tuple[tuple[str, str]], tuple] = (),
):
    """
    Generate figures of individual realizations for several logic trees.

    This function iterates through logic tree directories, filters out specified logic trees,
    and generates figures for the remaining logic trees.

    Parameters
    ----------
    results_directory : Union[Path, str]
        Directory containing the results of the logic tree realizations.
    plot_output_directory : Union[Path, str]
        Directory where the generated plots will be saved.
    locations : tuple[constants.LocationCode], optional
        The locations to plot. Default is (constants.LocationCode.AKL,
                                           constants.LocationCode.WLG,
                                           constants.LocationCode.CHC).
    im : constants.IntensityMeasure, optional
        Intensity measure to be used in the plots. Default is constants.IntensityMeasure.PGA.
    vs30 : int, optional
        Vs30 value to be used in the plots. Default is 400.
    im_xlims : tuple, optional
        X-axis limits for the intensity measure. Default is (9e-5, 5).
    poe_min_plot : float, optional
        Minimum probability of exceedance to be plotted. Default is 1e-5.
    ybuffer_absmax_over_val : float, optional
        Defines how much to extend the y-axis limits above the maximum value in the plot (for a buffer).
        The amount of extension is defined as the maximum plotted value divided by this value. Default is 10.0.
    selected_subduction_interface : InterfaceName, optional
        Subduction interface to be selected. Default is constants.InterfaceName.HIK_and_PUY.
    plot_dpi : int, optional
        DPI for the generated plots. Default is 500.
    notes_to_exclude : tuple, optional
        Notes for results to exclude from the plots in the form of
        (source_logic_tree_note, ground_motion_logic_tree_note). Default is an empty tuple.

    Returns
    -------
    None
    """

    full_subduction_interface_str = f"INTER_{selected_subduction_interface}"

    if isinstance(results_directory, str):
        results_directory = Path(results_directory)
    if isinstance(plot_output_directory, str):
        plot_output_directory = Path(plot_output_directory)

    if not plot_output_directory.exists():
        plot_output_directory.mkdir(parents=True)

    logic_tree_indices_to_skip = []

    aggregate_stats_results = (
        loading_functions.load_aggregate_stats_for_all_logic_trees_in_directory(
            results_directory
        )
    )
    collated_notes_df = aggregate_stats_results.collated_notes_df

    ## exclude the specified logic tree indices
    for source_logic_tree_note, ground_motion_logic_tree_note in notes_to_exclude:

        print(source_logic_tree_note, ground_motion_logic_tree_note)

        exclude_bool_idx = (
            collated_notes_df["source_logic_tree_note"] == source_logic_tree_note
        ) & (
            collated_notes_df["ground_motion_logic_tree_note"]
            == ground_motion_logic_tree_note
        )
        logic_tree_indices_to_skip.append(
            collated_notes_df[exclude_bool_idx]["logic_tree_index"].values[0]
        )

    ### Skip the interface branches that are not the selected one
    interface_logic_tree_indices = collated_notes_df[
        "source_logic_tree_note"
    ].str.contains("INTER")
    interface_indices_to_skip = ~collated_notes_df[interface_logic_tree_indices][
        "source_logic_tree_note"
    ].str.contains(full_subduction_interface_str)
    logic_tree_indices_to_skip.extend(interface_indices_to_skip.index)

    for logic_tree_index_dir in natsort.natsorted(results_directory.iterdir()):

        if logic_tree_index_dir.is_dir():

            print(f"Plotting hazard curves from {logic_tree_index_dir.name}")

            logic_tree_idx = int(logic_tree_index_dir.name.split("_")[-1])

            if logic_tree_idx in logic_tree_indices_to_skip:
                print(f"Skipping logic tree index {logic_tree_idx}")
                continue

            make_figures_of_individual_realizations_for_a_single_logic_tree(
                logic_tree_index_dir=logic_tree_index_dir,
                plot_output_directory=plot_output_directory,
                locations=locations,
                im=im,
                vs30=vs30,
                im_xlims=im_xlims,
                poe_min_plot=poe_min_plot,
                ybuffer_absmax_over_val=ybuffer_absmax_over_val,
                plot_dpi=plot_dpi,
            )
