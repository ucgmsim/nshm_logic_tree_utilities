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


import toml
import plotting_helpers


### Used function
### tidied up
def make_figure_of_gmm_dispersion_ranges(results_directory: Union[Path, str],
                         plot_output_directory: Union[Path, str],
                         locations : list[str] = ["AKL", "WLG", "CHC"],
                         filter_strs: list[str] = ["CRU", "HIK_and_PUY", "SLAB"],
                         vs30: int = 400,
                         im:str = "PGA",
                         plot_dpi=500,
                         num_interp_mean_points = 1000,
                         min_log10_mean_for_interp = -6,
                         max_log10_mean_for_interp = -2,
                         plot_interpolations=False,
                         min_mean_value_for_interp_plots = 1e-9):

    """
    Generate a figure showing the dispersion ranges of ground motion models (GMMs) for specified locations.

    Parameters
    ----------
    results_directory : Union[Path, str]
        The directory containing the results data.
    plot_output_directory : Union[Path, str]
        The directory where the plot will be saved.
    locations : list of str, optional
        A list of location codes to include in the plot. Default is ["AKL", "WLG", "CHC"].
    filter_strs : list of str, optional
        A list of filter strings to apply to the data. Default is ["CRU", "HIK_and_PUY", "SLAB"].
    vs30 : int, optional
        The Vs30 value to use in the plot. Default is 400.
    im : str, optional
        The intensity measure to use in the plot. Default is "PGA".
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


    dispersion_range_dict = plotting_helpers.get_interpolated_gmms(
        results_directory=results_directory,
        locations=              locations,
        filter_strs =                filter_strs,
        vs30 =                 vs30,
        im =                   im,
        num_interp_mean_points=num_interp_mean_points,
    min_log10_mean_for_interp= min_log10_mean_for_interp,
    max_log10_mean_for_interp= max_log10_mean_for_interp,
        plot_interpolations =                  plot_interpolations,
        min_mean_value_for_interp_plots =    min_mean_value_for_interp_plots)


    linestyle_lookup_dict = {"CRU":"--",
                             "HIK_and_PUY":"-.",
                             "SLAB":":"}

    color_lookup_dict = {"AKL":"blue",
                         "WLG":"orange",
                         "CHC":"red"}

    plt.figure()


    mm = np.logspace(min_log10_mean_for_interp, max_log10_mean_for_interp, num_interp_mean_points)

    for location in locations:

            for filter_str in filter_strs:

                print(filter_str)

                if filter_str == "HIK_and_PUY":

                    plt.semilogy(dispersion_range_dict[location][filter_str],
                    mm,
                    label=f"{location} INTER",
                    linestyle=linestyle_lookup_dict[filter_str],
                    color=color_lookup_dict[location])

                else:

                    plt.semilogy(dispersion_range_dict[location][filter_str],
                    mm,
                    label=f"{location} {filter_str}",
                    linestyle=linestyle_lookup_dict[filter_str],
                    color=color_lookup_dict[location])

    plt.legend()
    plt.ylabel(rf'Mean annual hazard probability, $\mu_{{P({im.upper()}={im.lower()})}}$')
    plt.xlabel(rf'Range in dispersion in hazard probability, $\sigma_{{\ln P({im.upper()}={im.lower()})}}$')
    plt.grid(linestyle='--')
    plt.savefig(plot_output_directory / f"dispersion_ranges_from_dir_{results_directory.name}.png", dpi=plot_dpi)

    #plt.show()
    print()


## A good plotting function. Use autorun21 for these plots
## Plots the ground motion models with subplots for different tectonic region types
## for a given location
def make_figure_of_gmcms(results_directory: Union[Path, str],
                         plot_output_directory: Union[Path, str],
                         locations : list[str] = ["AKL", "WLG", "CHC"],
                         vs30: int = 400,
                         im:str = "PGA",
                         plot_dpi=500):

    locations_nloc_dict = toml.load('resources/location_code_to_nloc_str.toml')
    tectonic_type_to_linestyle = toml.load('resources/tectonic_region_type_to_linestyle.toml')
    location_to_full_location = toml.load('resources/location_code_to_full_name.toml')
    model_to_plot_label = toml.load('resources/model_name_lookup_for_plot.toml')
    glt_model_color = toml.load('resources/model_plot_colors.toml')

    if isinstance(results_directory, str):
        results_directory = Path(results_directory)

    if isinstance(plot_output_directory, str):
        plot_output_directory = Path(plot_output_directory)

    loaded_results = plotting_helpers.load_aggregate_stats_for_all_logic_trees_in_directory(results_directory, locations)
    data_df = loaded_results.data_df
    run_notes_df = loaded_results.run_notes_df

    plt.close("all")
    fig, axes = plt.subplots(3, 3,figsize=(6,9))

    sorted_logic_tree_indices = plotting_helpers.sort_logic_tree_index_by_gmcm_model_name(results_directory)

    for location_row_idx, location in enumerate(locations):

        nloc_001_str = locations_nloc_dict[location]

        mean_list = []
        std_ln_list = []
        non_zero_run_list = []

        for logic_tree_index in sorted_logic_tree_indices:

            logic_tree_name_str = f"logic_tree_index_{logic_tree_index}"

            print()

            mean = data_df[(data_df["agg"] == "mean") &
                      (data_df["vs30"] == vs30) &
                      (data_df["imt"] == im) &
                      (data_df["hazard_model_id"] == logic_tree_name_str) &
                      (data_df["nloc_001"] == nloc_001_str)]["values"].values[0]

            mean_max = np.max(mean)
            print(f'logic_tree_name_str {logic_tree_name_str} max mean: {mean_max}')

            std_ln = data_df[(data_df["agg"] == "std_ln") &
                      (data_df["vs30"] == vs30) &
                      (data_df["imt"] == im) &
                      (data_df["hazard_model_id"] == logic_tree_name_str) &
                      (data_df["nloc_001"] == nloc_001_str)]["values"].values[0]

            mean_list.append(mean)
            std_ln_list.append(std_ln)
            non_zero_run_list.append(logic_tree_name_str)
            slt_note = f"{run_notes_df[run_notes_df["logic_tree_index"] == logic_tree_index]["slt_note"].values[0]}"
            glt_note = f"{run_notes_df[run_notes_df["logic_tree_index"]==logic_tree_index]["glt_note"].values[0]}"

            trts_from_note = slt_note.split(">")[-2].strip().split(":")[-1].strip("[]")
            glt_model_and_weight_str = glt_note.split(">")[-2].strip(" []")
            glt_model = glt_model_and_weight_str.split("*")[0]

            linestyle = tectonic_type_to_linestyle[trts_from_note]

            if trts_from_note == "INTER_only_HIK":
                continue
            if trts_from_note == "INTER_only_PUY":
                continue

            if "CRU" in trts_from_note:
                subplot_idx = 0
            if "INTER" in trts_from_note:
                subplot_idx = 1
            if "SLAB" in trts_from_note:
                subplot_idx = 2

            axes[location_row_idx, subplot_idx].semilogy(std_ln, mean, label=model_to_plot_label[glt_model],
                                        linestyle=linestyle, color=glt_model_color[glt_model])

            axes[location_row_idx, subplot_idx].text(
                x=0.68,
                y=0.2,
                s=location_to_full_location[location],
                horizontalalignment="right",
                bbox=dict(facecolor='white', alpha=0.4, edgecolor='none',pad=0)
            )

            axes[location_row_idx, subplot_idx].set_ylim(1e-5,0.6)
            axes[location_row_idx, subplot_idx].set_xlim(-0.01, 0.7)
            axes[location_row_idx, subplot_idx].grid(which='major',
                                                     linestyle='--',
                                                     linewidth='0.5',
                                                     color='black',
                                                     alpha=0.5)

            if subplot_idx == 0:
                axes[0,0].set_title("Active shallow crust", fontsize=11)
                axes[0, 1].set_title("Subduction interface", fontsize=11)
                axes[0,2].set_title("Subduction intraslab", fontsize=11)

                if location_row_idx == 1:
                    axes[location_row_idx, subplot_idx].set_ylabel(r'Mean annual hazard probability, $\mu_{P(IM=im)}$')

            if subplot_idx == 1:
                if location_row_idx == 2:
                    axes[location_row_idx, subplot_idx].set_xlabel(r'Dispersion in hazard probability, $\sigma_{\ln P(IM=im)}$')
                axes[location_row_idx, subplot_idx].set_yticklabels([])

            if subplot_idx == 2:
                axes[location_row_idx, subplot_idx].set_yticklabels([])

            if (location_row_idx == 0) or (location_row_idx == 1):
                axes[location_row_idx, subplot_idx].set_xticklabels([])

            axes[location_row_idx, subplot_idx].legend(
                             loc="lower left",
                             prop={'size': 6},
                             framealpha=0.4,
                             handlelength=2.2,
                             handletextpad=0.2)

    plt.subplots_adjust(wspace=0.0, hspace=0.0, left=0.11, right=0.99, bottom=0.05, top=0.97)

    plt.savefig(plot_output_directory / f"gmcms_in_dir_{results_directory.name}_{im}_all_locations.png",dpi=plot_dpi)


## tidied up function
def make_figure_of_srm_and_gmcm_model_dispersions(locations: list[str],
                                               srm_models_data_directory: Union[Path, str],
                                               gmcm_models_data_directory: Union[Path, str],
                                               plot_output_directory: Union[Path, str],
                                               vs30:int=400,
                                               im:str="PGA",
                                               plot_title_font_size: float = 12,
                                               plot_dpi=500):

    """
    Make a figure containing subplots of mean prediction on the vertical axis and the dispersion in
    predictions on the horizontal axis, following Bradley (2009).

    This figure can only be for one vs30 value and one intensity measure (im). Each column of subplots is for a
    different location so the figure will have a number of columns equal to the length of the locations list.
    The figure will always have 3 rows of subplots. The top row shows the crustal ground motion
    characterization models (GMCMs), the middle row shows the interface and intraslab GMCMs,
    and the bottom row shows the seismicity rate model (SRM) components.

    Note that the data that this function uses is loaded from the output of run_toshi_hazard_post_script.py so that needs to be run first.
    """

    if isinstance(srm_models_data_directory, str):
        srm_models_data_directory = Path(srm_models_data_directory)
    if isinstance(gmcm_models_data_directory, str):
        gmcm_models_data_directory = Path(gmcm_models_data_directory)
    if isinstance(plot_output_directory, str):
        plot_output_directory = Path(plot_output_directory)

    num_plot_cols = len(locations)

    locations_nloc_dict = toml.load('resources/location_code_to_nloc_str.toml')
    location_to_full_location = toml.load('resources/location_code_to_full_name.toml')
    model_to_plot_label = toml.load('resources/model_name_lookup_for_plot.toml')
    glt_model_color = toml.load('resources/model_plot_colors.toml')

    ## relate plot row to data
    plot_row_to_data_lookup = {0:plotting_helpers.load_aggregate_stats_for_all_logic_trees_in_directory(gmcm_models_data_directory),
                               1:plotting_helpers.load_aggregate_stats_for_all_logic_trees_in_directory(gmcm_models_data_directory),
                               2:plotting_helpers.load_aggregate_stats_for_all_logic_trees_in_directory(srm_models_data_directory)}

    ## Sort the logic_tree_index_[x] names by the ground motion characterization model that
    ## was isolated by that logic tree.
    sorted_gmm_run_nums = plotting_helpers.sort_logic_tree_index_by_gmcm_model_name(gmcm_models_data_directory)

    ### For each row of the subplot, identify the logic_tree_indices (output directories) that will
    ### be plotted by looking at the run_notes dataframe, and identifying the rows that contain key strings
    ### and then getting the logic tree indices (several indexes) that correspond to those rows.

    plot_row_to_logic_tree_index = {}

    for row_index in range(3):
        if row_index == 0:
            plot_row_to_logic_tree_index[row_index] = plot_row_to_data_lookup[row_index].run_notes_df[
                plot_row_to_data_lookup[row_index].run_notes_df["slt_note"].str.contains("CRU")]["logic_tree_index"]
        if row_index == 1:
            plot_row_to_logic_tree_index[row_index] = plot_row_to_data_lookup[row_index].run_notes_df[
                plot_row_to_data_lookup[row_index].run_notes_df["slt_note"].str.contains("INTER_HIK_and_PUY|SLAB")]["logic_tree_index"]
        if row_index == 2:
            plot_row_to_logic_tree_index[row_index] = plot_row_to_data_lookup[row_index].run_notes_df[
                plot_row_to_data_lookup[row_index].run_notes_df["slt_note"].str.contains("CRU|INTER_HIK_and_PUY")]["logic_tree_index"]

    ####################################################

    plt.close("all")
    fig, axes = plt.subplots(3, num_plot_cols, figsize=(3*num_plot_cols, 9))

    for row_index in range(3):

        for column_index in range(num_plot_cols):

            axes[row_index, column_index].set_prop_cycle(None)

            plot_location = locations[column_index]
            if row_index == 0:
                axes[row_index, column_index].set_title(location_to_full_location[plot_location],
                                                        fontsize=plot_title_font_size)

            if row_index != 2:
                sorted_logic_tree_indices = sorted_gmm_run_nums
            if row_index == 2:
                sorted_logic_tree_indices = plot_row_to_logic_tree_index[row_index]

            for sorted_logic_tree_index in sorted_logic_tree_indices:

                if sorted_logic_tree_index in plot_row_to_logic_tree_index[row_index]:

                    logic_tree_name_str = f"logic_tree_index_{sorted_logic_tree_index}"

                    if row_index in [0,1]:
                        run_note = plot_row_to_data_lookup[row_index].run_notes_df[plot_row_to_data_lookup[row_index].run_notes_df["logic_tree_index"] == sorted_logic_tree_index]["glt_note"].values[0]
                        short_note = run_note.split(">")[-2].split("*")[-2].strip(" [")
                        plot_label = model_to_plot_label[short_note]

                    if row_index == 2:
                        run_note = plot_row_to_data_lookup[row_index].run_notes_df[plot_row_to_data_lookup[row_index].run_notes_df["logic_tree_index"] == sorted_logic_tree_index]["slt_note"].values[0]
                        short_note = run_note.split(">")[1].split(":")[-1].strip(" []") + "_" +\
                                     run_note.split(">")[2].strip()

                        plot_label = model_to_plot_label[short_note]

                    mean = plot_row_to_data_lookup[row_index].data_df[(plot_row_to_data_lookup[row_index].data_df["agg"] == "mean") &
                                                 (plot_row_to_data_lookup[row_index].data_df["vs30"] == vs30) &
                                                 (plot_row_to_data_lookup[row_index].data_df["imt"] == im) &
                                                 (plot_row_to_data_lookup[row_index].data_df["hazard_model_id"] == logic_tree_name_str) &
                                                 (plot_row_to_data_lookup[row_index].data_df["nloc_001"] == locations_nloc_dict[plot_location])]["values"].values[0]

                    std_ln = plot_row_to_data_lookup[row_index].data_df[(plot_row_to_data_lookup[row_index].data_df["agg"] == "std_ln") &
                                                 (plot_row_to_data_lookup[row_index].data_df["vs30"] == vs30) &
                                                 (plot_row_to_data_lookup[row_index].data_df["imt"] == im) &
                                                 (plot_row_to_data_lookup[row_index].data_df["hazard_model_id"] == logic_tree_name_str) &
                                                 (plot_row_to_data_lookup[row_index].data_df["nloc_001"] == locations_nloc_dict[plot_location])]["values"].values[0]

                    if row_index != 2:
                        if "CRU" in run_note:
                            plot_linestyle = '--'
                        if "INTER" in run_note:
                            plot_linestyle = '--'
                        if "SLAB" in run_note:
                            plot_linestyle = '-.'

                    elif row_index == 2:
                        if "CRU" in run_note:
                            plot_linestyle = '--'
                        if "INTER" in run_note:
                            plot_linestyle = '-.'

                    axes[row_index, column_index].semilogy(std_ln, mean, label=plot_label,
                                                       color=glt_model_color[short_note],
                                                           linestyle=plot_linestyle)

                    axes[row_index, column_index].grid(which='major',
                                                             linestyle='--',
                                                             linewidth='0.5',
                                                             color='black',
                                                             alpha=0.6)

                    axes[row_index, column_index].set_ylim(1e-5, 0.7)
                    axes[row_index, column_index].set_xlim(-0.01, 0.7)

                    axes[row_index, column_index].legend(
                             loc="lower left",
                             prop={'size': 6},
                             framealpha=0.4,
                             handlelength=2.2,
                             handletextpad=0.2)

                    if row_index in [0,1]:
                        axes[row_index, column_index].set_xticklabels([])
                    if column_index > 0:
                        axes[row_index, column_index].set_yticklabels([])

                    if (row_index == 1) & (column_index == 0):
                        axes[row_index, column_index].set_ylabel(r'Mean annual hazard probability, $\mu_{P(PGA=pga)}$')


    ### The text on the left and bottom of the figure require a constant margin in pixels
    ### but the figure margins need to be provided in fractions of the figure dimensions.
    ### As the figure dimensions change depending on how many columns are in the figure,
    ### the figure margins need to be calculated in pixels and then converted to fractions
    fig_margins = plotting_helpers.convert_edge_margin_in_pixels_to_fraction(fig,
                                                            100,
                                                            5,
                                                            45,
                                                            30)

    ### Adjust the figure margins before using the figure axes positions to place text
    plt.subplots_adjust(left=fig_margins.left,
                        right = fig_margins.right,
                        bottom = fig_margins.bottom,
                        top = fig_margins.top,
                        wspace=0.0,
                        hspace=0.0)

    ### Center the x-axis label differently depending on whether there are an odd or even number of columns
    if num_plot_cols % 2 != 0: # odd number of columns
        middle_col_index = int(np.floor(num_plot_cols/2))
        axes[2, middle_col_index].set_xlabel(r'Dispersion in hazard probability, $\sigma_{\ln P(PGA=pga)}$')
    if num_plot_cols % 2 == 0: # even number of columns
        # anchoring the middle of the x-axis label text to the right edge of the column calculated here
        anchor_col_index = int(num_plot_cols - num_plot_cols/2 - 1)
        #fig.text(axes[2, anchor_col_index].get_position().x1, axes[2, anchor_col_index].get_position().y0 - 0.05, r'Dispersion in hazard probability, $\sigma_{\ln P(PGA=pga)}$', ha='center', va='center')
        fig.text(axes[2, anchor_col_index].get_position().x1, 0.01,
             r'Dispersion in hazard probability, $\sigma_{\ln P(PGA=pga)}$', ha='center', va='center')

    row_titles_x0 = 0.02
    fig.text(row_titles_x0, axes[0, 0].get_position().y0,'Ground Motion Characterization Models', ha='center',
             va='center',rotation=90, fontsize=plot_title_font_size)

    fig.text(row_titles_x0, (axes[2, 0].get_position().y0 + axes[2, 0].get_position().y1)/2.0,
             'Seismicity Rate Models',
             ha='center', va='center', rotation=90, fontsize=plot_title_font_size)


    #plt.show()
    plt.savefig(plot_output_directory / f"{"_".join(locations)}_dispersion_poster_plot.png", dpi=plot_dpi)
    print()

## tidied up function
def make_figure_of_coefficient_of_variation(results_directory: Union[Path,str], plot_output_directory: Union[Path,str], plot_dpi:int=500,
                                         plot_fontsize:float=12.0, plot_lineweight=5,
                                         location:str="WLG", im:str="PGA", vs30:int=400):

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
    location : str, optional
        The location code (default is "WLG").
    im : str, optional
        The intensity measure (default is "PGA").
    vs30 : int, optional
        The Vs30 value (default is 400).

    Returns
    -------
    None
    """

    nshm_im_levels = np.loadtxt('resources/nshm_im_levels.txt')

    if isinstance(results_directory, str):
        results_directory = Path(results_directory)
    if isinstance(plot_output_directory, str):
        plot_output_directory = Path(plot_output_directory)

    locations_nloc_dict = toml.load('resources/location_code_to_nloc_str.toml')

    nloc_001_str = locations_nloc_dict[location]

    mean_list = []
    cov_list = []

    resulting_hazard_curves = plotting_helpers.load_aggregate_stats_for_all_logic_trees_in_directory(results_directory)

    data_df = resulting_hazard_curves.data_df
    run_notes_df = resulting_hazard_curves.run_notes_df

    logic_tree_index_list = [f"logic_tree_index_{x}" for x in run_notes_df["logic_tree_index"].values]
    for run_idx, logic_tree_index_str in enumerate(logic_tree_index_list):

        mean = data_df[(data_df["agg"] == "mean") &
                  (data_df["vs30"] == vs30) &
                  (data_df["imt"] == im) &
                  (data_df["hazard_model_id"] == logic_tree_index_str) &
                  (data_df["nloc_001"] == nloc_001_str)]["values"].values[0]

        cov = data_df[(data_df["agg"] == "cov") &
                  (data_df["vs30"] == vs30) &
                  (data_df["imt"] == im) &
                  (data_df["hazard_model_id"] == logic_tree_index_str) &
                  (data_df["nloc_001"] == nloc_001_str)]["values"].values[0]

        mean_list.append(mean)
        cov_list.append(cov)

    plt.rcParams.update({'font.size': plot_fontsize})

    ### This figsize was used for the 1 minute Poster Showcase slide
    #plt.figure(figsize=(5.12,4.62))

    ### This figsize was use for the poster
    plt.figure(figsize=(7.3, 4.62))

    plt.semilogx(nshm_im_levels, cov_list[0], linestyle='--', linewidth=plot_lineweight, label='source model')
    plt.semilogx(nshm_im_levels, cov_list[1], linestyle='-.', linewidth=plot_lineweight, label='ground motion model')
    plt.semilogx(nshm_im_levels, cov_list[2], linestyle='-', linewidth=plot_lineweight, label='both')
    plt.legend(handlelength=4)
    plt.ylabel("Modelling uncertainty\n(coefficient of variation of model predictions)")
    plt.xlabel('Peak ground acceleration (g)')
    plt.xlim(1e-2,5)
    plt.ylim(0.05,0.8)

    plt.grid(which='major',
           linestyle='--',
           linewidth='0.5',
           color='black',
           alpha=0.6)

    plt.subplots_adjust(left=0.11, right=0.99, bottom=0.12, top=0.97 )
    plt.savefig(plot_output_directory / "coefficient_of_variation.png",dpi=plot_dpi)
    plt.close()

## tidied up
def make_figure_of_srm_model_components(results_directory: Union[Path,str],
                                           plot_output_directory:Union[Path, str],
                                           locations: list[str] = ["AKL", "WLG", "CHC"],
                                           im: str="PGA",
                                           vs30:int=400,
                                           plot_dpi:int=500):

    """
    Make a figure of the dispersion in seismicity rate model components.

    Parameters
    ----------
    results_directory : Union[Path, str]
        The directory containing the data. This directory should contain subdirectories
        named as logic_tree_index_[x] where [x] is the index the logic_tree_set had in the input list.
    plot_output_directory : Union[Path, str]
        The directory where the plot will be saved.
    locations : list[str], optional
        The locations to plot (default is ["AKL", "WLG", "CHC"]).
    im : str, optional
        The intensity measure (default is "PGA").
    vs30 : int, optional
        The Vs30 value (default is 400).
    plot_dpi : int, optional
        The resolution of the plot in dots per inch (default is 500).
    """

    ### tectonic region type and model name lookup dictionaries
    trt_short_to_long = {"CRU":"crust",
                         "INTER":"subduction interface\n"}

    ### model name lookup dictionary
    model_name_short_to_long = {"deformation_model":"deformation model\n(geologic or geodetic)",
                                "time_dependence":"time dependence\n(time-dependent or time-independent)",
                                "MFD":"magnitude frequency distribution",
                                "moment_rate_scaling":"moment rate scaling"}

    locations_nloc_dict = toml.load('resources/location_code_to_nloc_str.toml')
    location_to_full_location = toml.load('resources/location_code_to_full_name.toml')

    if isinstance(results_directory, str):
        results_directory = Path(results_directory)
    if isinstance(plot_output_directory, str):
        plot_output_directory = Path(plot_output_directory)

    loaded_results = plotting_helpers.load_aggregate_stats_for_all_logic_trees_in_directory(results_directory)

    logic_tree_name_notes_df = loaded_results.run_notes_df
    data_df = loaded_results.data_df

    ## filter out all notes except those that are needed
    filtered_logic_tree_name_notes_df = logic_tree_name_notes_df[~logic_tree_name_notes_df["slt_note"].str.contains("only")]

    logic_tree_name_strs = [f"logic_tree_index_{x}" for x in filtered_logic_tree_name_notes_df["logic_tree_index"].values]

    fig, axes = plt.subplots(1, 3, figsize=(8,4))

    linestyle_lookup_dict = {"CRU":"-", "INTER":"--"}

    for location_idx, location in enumerate(locations):

        for logic_tree_name in logic_tree_name_strs:

            nloc_001_str = locations_nloc_dict[location]

            mean = data_df[(data_df["agg"] == "mean") &
                      (data_df["vs30"] == vs30) &
                      (data_df["imt"] == im) &
                      (data_df["hazard_model_id"] == logic_tree_name) &
                      (data_df["nloc_001"] == nloc_001_str)]["values"].values[0]

            std_ln = data_df[(data_df["agg"] == "std_ln") &
                      (data_df["vs30"] == vs30) &
                      (data_df["imt"] == im) &
                      (data_df["hazard_model_id"] == logic_tree_name) &
                      (data_df["nloc_001"] == nloc_001_str)]["values"].values[0]

            needed_idx = mean > 1e-8
            mean = mean[needed_idx]
            std_ln = std_ln[needed_idx]

            slt_note = f"{logic_tree_name_notes_df[logic_tree_name_notes_df["logic_tree_index"] == int(logic_tree_name.split("_")[-1])]["slt_note"].values[0]}"

            tectonic_region_type = slt_note.split(">")[1].strip(" ]'").split("[")[-1]

            if "INTER" in tectonic_region_type:
                tectonic_region_type = "INTER"

            model_name = slt_note.split(">")[-2].strip(" ")

            note = f"{trt_short_to_long[tectonic_region_type]} {model_name_short_to_long[model_name]}"

            axes[location_idx].semilogy(std_ln, mean, label=note, linestyle = linestyle_lookup_dict[tectonic_region_type])

            axes[location_idx].set_ylim(1e-6,0.6)
            axes[location_idx].set_xlim(-0.01, 0.37)
            axes[location_idx].set_title(location_to_full_location[location])

            if location_idx == 0:
                axes[location_idx].set_ylabel(r'Mean annual hazard probability, $\mu_{P(IM=im)}$')

            if location_idx == 1:
                axes[location_idx].set_xlabel(r'Dispersion in hazard probability, $\sigma_{\ln P(IM=im)}$')
                axes[location_idx].set_yticklabels([])
            if location_idx == 2:
                axes[location_idx].set_yticklabels([])

        axes[location_idx].grid(which='major', linestyle='--', linewidth='0.5', color='black')

        axes[location_idx].legend(loc="lower left",
                                  prop={'size': 6},
                                  framealpha=0.6,
                                  handlelength=2.2,
                                  handletextpad=0.2)

    plt.subplots_adjust(wspace=0.0,left=0.1,right=0.99,top=0.94)
    plt.savefig(plot_output_directory / f"srm_dispersions_from_dir_{results_directory.name}.png",dpi=plot_dpi)


### tidied up
def make_figure_showing_Bradley2009_method(results_directory: Union[Path,str],
                                     plot_output_directory: Union[Path,str],
                                     registry_directory: Union[Path,str],
                                     location_short_name:str="WLG",
                                     vs30:int=400,
                                     im:str="PGA",
                                     plot_dpi:int=500):

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
    location_short_name : str, optional
        The location code (default is "WLG").
    vs30 : int, optional
        The Vs30 value (default is 400).
    im : str, optional
        The intensity measure (default is "PGA").
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


    locations_nloc_dict = toml.load('resources/location_code_to_nloc_str.toml')
    nshm_im_levels = np.loadtxt("resources/nshm_im_levels.txt")

    plot_colors = ["tab:purple", "tab:orange", "tab:green"]
    plot_linestyles = [":", "-", "--"]

    plot_label_lookup = {"Stafford2022(mu_branch=Upper)":"Upper branch of Stafford 2022",
                        "Stafford2022(mu_branch=Central)":"Central branch of Stafford 2022",
                        "Stafford2022(mu_branch=Lower)":"Lower branch of Stafford 2022"}

    gmcm_name_formatting_lookup = {"Stafford2022":"Stafford (2022)"}

    individual_realization_df = ds.dataset(source=results_directory/"individual_realizations",
                                           format="parquet").to_table().to_pandas()

    individual_realizations_needed_indices = (individual_realization_df["hazard_model_id"] == results_directory.name) & \
                     (individual_realization_df["nloc_001"] == locations_nloc_dict[location_short_name])

    filtered_individual_realization_df = individual_realization_df[individual_realizations_needed_indices]

    realization_names = plotting_helpers.lookup_realization_name_from_hash(filtered_individual_realization_df, registry_directory)

    hazard_rate_array = np.zeros((len(filtered_individual_realization_df),44))

    for realization_index in range(len(filtered_individual_realization_df)):
        hazard_rate_array[realization_index,:] = filtered_individual_realization_df.iloc[realization_index]["branches_hazard_rates"]

    ### Convert the rate to annual probability of exceedance
    hazard_prob_of_exceedance = calculators.rate_to_prob(hazard_rate_array, 1.0)

    resulting_hazard_curves = plotting_helpers.load_aggregate_stats_for_all_logic_trees_in_directory(results_directory.parent)

    agg_stats_df = resulting_hazard_curves.data_df

    ## Get the needed mean and standard deviation values
    mean = agg_stats_df[(agg_stats_df["agg"] == "mean") &
                   (agg_stats_df["vs30"] == vs30) &
                   (agg_stats_df["imt"] == im) &
                   (agg_stats_df["hazard_model_id"] == results_directory.name) &
                   (agg_stats_df["nloc_001"] == locations_nloc_dict[location_short_name])]["values"].values[0]

    std_ln = agg_stats_df[(agg_stats_df["agg"] == "std_ln") &
                     (agg_stats_df["vs30"] == vs30) &
                     (agg_stats_df["imt"] == im) &
                     (agg_stats_df["hazard_model_id"] == results_directory.name) &
                     (agg_stats_df["nloc_001"] == locations_nloc_dict[location_short_name])]["values"].values[0]

    plot_ylims = (1e-5,1)
    fig, axes = plt.subplots(1, 2, figsize=(8, 5))

    for realization_index in range(len(hazard_prob_of_exceedance)):
        gmcm_name_with_branch = realization_names[realization_index].ground_motion_characterization_models_id
        gmcm_name = gmcm_name_with_branch.split("(")[0]

        if gmcm_name_with_branch in plot_label_lookup.keys():
            plot_label = plot_label_lookup[gmcm_name_with_branch]
        else:
            plot_label = gmcm_name_with_branch

        axes[0].loglog(nshm_im_levels, hazard_prob_of_exceedance[realization_index],
                       label=plot_label,
                       marker="o",
                       color=plot_colors[realization_index],
                       linestyle=plot_linestyles[realization_index])

    # Customize the second subplot
    if gmcm_name in gmcm_name_formatting_lookup.keys():
        gmcm_name_label = gmcm_name_formatting_lookup[gmcm_name]
    else:
        gmcm_name_label = gmcm_name

    axes[1].semilogy(std_ln, mean,marker="o", linestyle="-", color="tab:blue", label=gmcm_name_label)

    axes[0].legend(loc="lower left")
    axes[1].legend(loc="lower left")

    for ax_index in range(len(axes)):
        axes[ax_index].grid(which='major',
                     linestyle='--',
                     linewidth='0.5',
                     color='black',
                     alpha=0.5)

        axes[ax_index].set_ylim(plot_ylims)
    axes[0].set_xlim(9e-5,5)
    axes[1].set_xlim(-0.01, 0.68)

    ### The annotations for explanation
    annotation_ims = [1e-2, 1e-1, 1e0]

    ### latex formatted strings corresponding to the annotation_ims.
    ### These could be generated automatically from the annotation_ims
    ### but they are just hard coded for simplicity.
    manually_matched_latex_strings = [r"10$^{-2}$", r"10$^{-1}$", r"10$^{0}$"]
    annotation_labels = ["A", "B", "C"]

    for annotation_im in annotation_ims:

        im_index = np.where(nshm_im_levels == annotation_im)[0][0]

        print(f"im index = {im_index}, im value = {annotation_im}, mean = {mean[im_index]:.1e}")

        ### Draw vertical lines at IM values
        axes[0].hlines(y=mean[im_index], xmin=nshm_im_levels[0]/10, xmax=annotation_im, color="black", linestyle="--")
        axes[0].vlines(x=annotation_im, ymin=plot_ylims[0], ymax=mean[im_index], color="black", linestyle="--")
        axes[0].vlines(x=annotation_im, ymin=mean[im_index], ymax=plot_ylims[1], color="tab:blue", linestyle="--")

        ### Draw the arrows
        axes[0].plot(annotation_im, plot_ylims[1]-0.45,
                     marker=r'$\uparrow$',
                     markersize=20,
                     color="tab:blue")

        ### Write the standard deviation value
        axes[0].text(annotation_im,
                     plot_ylims[1]+0.5,
                     f"{std_ln[im_index]:.2f}",
                     ha='center',
                     va='center',
                     color="black")

        ### Plot labels (A), (B), (C)
        axes[0].text(annotation_im,
                     plot_ylims[1]+2.0,
                     f"{manually_matched_latex_strings[annotation_ims.index(annotation_im)]}",
                     ha='center',
                     va='center',
                     color="black")

        ### Draw the horizontal lines at the mean values
        axes[1].hlines(y=mean[im_index],
                       xmin=-0.01,
                       xmax=std_ln[im_index],
                       color="black", linestyle="--")

        ### Draw the vertical lines at the standard deviation values
        axes[1].vlines(x=std_ln[im_index],
                       ymin=plot_ylims[0],
                       ymax=plot_ylims[1],
                       color="black", linestyle="--")

        ### Plot labels (A), (B), (C)
        axes[1].text(std_ln[im_index],
                     plot_ylims[1]+0.5,
                     f"({annotation_labels[annotation_ims.index(annotation_im)]})",
                     ha='center',
                     va='center',
                     color="black")

    axes[0].set_xlabel(r'Peak ground acceleration (g)')
    axes[0].set_ylabel(r'Annual hazard probability, $\mu_{P(PGA=pga)}$')

    axes[1].set_ylabel(r'Mean annual hazard probability, $\mu_{P(PGA=pga)}$')
    axes[1].set_xlabel(r'Dispersion in hazard probability, $\sigma_{\ln P(PGA=pga)}$')

    axes[0].text(1.25e-4,
                 12,
                 "Dispersion in hazard probability",
                 ha='left',
                 va='center',
                 color="black")

    text_row_2_height = plot_ylims[1]+5.0

    axes[0].text(6e-3,
                 text_row_2_height,
                 "Reference point:",
                 ha='right',
                 va='center',
                 color="black")

    axes[0].text(annotation_ims[0],
                 text_row_2_height,
                 "(A)",
                 ha='center',
                 va='center',
                 color="black")

    axes[0].text(annotation_ims[1],
                 text_row_2_height,
                 "(B)",
                 ha='center',
                 va='center',
                 color="black")

    axes[0].text(annotation_ims[2],
                 text_row_2_height,
                 "(C)",
                 ha='center',
                 va='center',
                 color="black")

    axes[0].text(8e-3,
                 plot_ylims[1]+2.0,
                 f"{im} =  ",
                 ha='right',
                 va='center',
                 color="black")

    axes[0].text(5.8e-3,
                 plot_ylims[1]+0.5,
                 rf"$\sigma_{{\ln P({im.upper()}={im.lower()})}} = $",
                 ha='right',
                 va='center',
                 color="black")

    plt.subplots_adjust(bottom=0.1, top=0.81,left=0.085, right=0.99,wspace=0.23)

    plt.savefig(plot_output_directory / f"{gmcm_name}_{location_short_name}_predictions_and_aggregate_stats.png", dpi=plot_dpi)


def make_figures_of_individual_realizations_for_a_single_logic_tree(srm_or_gmcm:str,
                                                logic_tree_index_dir:Union[Path, str],
                                                plot_output_directory:Union[Path, str],
                                                registry_directory:Union[Path, str],
                                                locations: list[str] = ["AKL", "WLG", "CHC"],
                                                im: str = "PGA",
                                                vs30: int = 400,
                                                im_xlims = (9e-5, 5),
                                                poe_min_plot = 1e-5,
                                                plot_dpi: int = 500):

    locations_nloc_dict = toml.load('resources/location_code_to_nloc_str.toml')
    model_name_to_plot_format = toml.load('resources/model_name_lookup_for_plot.toml')
    srm_name_component_index_to_name = toml.load('resources/srm_name_component_index_to_name.toml')

    nshm_im_levels = np.loadtxt("resources/nshm_im_levels.txt")

    needed_im_level_indices = np.where((nshm_im_levels >= im_xlims[0]) & (nshm_im_levels <= im_xlims[1]))[0]

    if isinstance(logic_tree_index_dir, str):
        logic_tree_index_dir = Path(logic_tree_index_dir)
    if isinstance(plot_output_directory, str):
        plot_output_directory = Path(plot_output_directory)
    if isinstance(registry_directory, str):
        registry_directory = Path(registry_directory)

    if not plot_output_directory.exists():
        plot_output_directory.mkdir(parents=True)

    aggregate_stats_results = plotting_helpers.load_aggregate_stats_for_all_logic_trees_in_directory(logic_tree_index_dir.parent)

    run_notes_df = aggregate_stats_results.run_notes_df

    ## Get the tectonic region type from run_notes_df
    logic_tree_idx = int(logic_tree_index_dir.name.split("_")[-1])

    slt_note = run_notes_df[run_notes_df["logic_tree_index"] == logic_tree_idx]["slt_note"].values[0]
    tectonic_region_type = slt_note[slt_note.find("[")+1:slt_note.find("]")].strip()

    print(f"Processing {logic_tree_index_dir.name}")

    if logic_tree_index_dir.is_dir():

        individual_realization_df = ds.dataset(source=logic_tree_index_dir / "individual_realizations",
                                               format="parquet").to_table().to_pandas()

        plt.close("all")

        fig, axes = plt.subplots(2, len(locations),figsize=(3*len(locations), 6))

        poe_maxs = []
        ln_resid_mins = []
        ln_resid_maxs = []

        for location_idx, location in enumerate(locations):

            nloc_001_str = locations_nloc_dict[location]

            individual_realizations_needed_indices = (individual_realization_df["hazard_model_id"] == logic_tree_index_dir.name) & \
                 (individual_realization_df["nloc_001"] == nloc_001_str) & \
                 (individual_realization_df["vs30"] == vs30) & \
                 (individual_realization_df["imt"] == im)

            filtered_individual_realization_df = individual_realization_df[individual_realizations_needed_indices]

            realization_names = plotting_helpers.lookup_realization_name_from_hash(filtered_individual_realization_df, registry_directory)

            hazard_rate_array = np.zeros((len(filtered_individual_realization_df),44))

            for realization_index in range(len(filtered_individual_realization_df)):
                hazard_rate_array[realization_index,:] = filtered_individual_realization_df.iloc[realization_index]["branches_hazard_rates"]

            ### Convert the rate to annual probability of exceedance
            hazard_prob_of_exceedance = calculators.rate_to_prob(hazard_rate_array, 1.0)

            ln_resid_poe = np.log(hazard_prob_of_exceedance) - np.log(hazard_prob_of_exceedance[0])

            if len(hazard_prob_of_exceedance) == 0:
                print()

            model_names = []
            for realization_index in range(len(hazard_prob_of_exceedance)):
                if srm_or_gmcm == "gmcm":
                    gmcm_name_with_branch = realization_names[realization_index].ground_motion_characterization_models_id
                    gmcm_name = gmcm_name_with_branch.split("(")[0]
                    plot_label = gmcm_name_with_branch
                    title = f"{model_name_to_plot_format[gmcm_name]} ({logic_tree_index_dir.name})"

                if srm_or_gmcm == "srm":
                    srm_name = realization_names[realization_index].seismicity_rate_model_id
                    plot_label = srm_name
                    model_names.append(srm_name)

                poe_maxs.append(np.nanmax(hazard_prob_of_exceedance[realization_index][needed_im_level_indices]))
                axes[0,location_idx].loglog(nshm_im_levels[needed_im_level_indices], hazard_prob_of_exceedance[realization_index][needed_im_level_indices],
                               label=plot_label)

                axes[0, location_idx].set_xlim(im_xlims)
                axes[0, location_idx].grid(which='major',
                            linestyle='--',
                            linewidth='0.5',
                            color='black',
                            alpha=0.5)

                axes[0, location_idx].set_title(location)
                axes[0, location_idx].legend(loc="lower left",prop={'size': 3})

                ln_resid_mins.append(np.nanmin(ln_resid_poe[realization_index][needed_im_level_indices]))
                ln_resid_maxs.append(np.nanmax(ln_resid_poe[realization_index][needed_im_level_indices]))
                axes[1,location_idx].semilogx(nshm_im_levels[needed_im_level_indices], ln_resid_poe[realization_index][needed_im_level_indices],
                               label=plot_label)
                axes[1, location_idx].set_xlim(im_xlims)

                axes[1, location_idx].grid(which='major',
                            linestyle='--',
                            linewidth='0.5',
                            color='black',
                            alpha=0.5)

                axes[1, location_idx].legend(loc="lower left",prop={'size': 4})

                if location_idx > 0:
                    axes[0, location_idx].set_yticklabels([])
                    axes[1, location_idx].set_yticklabels([])
                axes[0, location_idx].set_xticklabels([])

            if srm_or_gmcm == "srm":

                srm_name_components_0 = model_names[0].split(",")

                for srm_name_component_index in range(len(srm_name_components_0)):
                    if model_names[1].split(",")[srm_name_component_index] != srm_name_components_0[srm_name_component_index]:
                        different_srm_name_component_index = srm_name_component_index

                        break
                print()
                ### The dictionary is stored as a toml file which can only have strings as dictionary keys
                srm_model_component_name = srm_name_component_index_to_name[str(different_srm_name_component_index)]
                model_name_for_lookup = f"{tectonic_region_type}_{srm_model_component_name}"
                title = f"{model_name_to_plot_format[model_name_for_lookup]} (index {logic_tree_index_dir.name.split("_")[-1]})"

        ### Set all the y-axis limits to the max values found in the last loop over locations
        for location_idx in range(len(locations)):
            axes[0, location_idx].set_ylim(poe_min_plot, np.max(poe_maxs)*1.1)
            axes[1, location_idx].set_ylim(np.min(ln_resid_mins), np.max(ln_resid_maxs))

        axes[0, 0].set_ylabel('Annual probability of exceedance')
        axes[1, 0].set_ylabel(r"$\ln$(APoE$_1$)-$\ln$(APoE$_2$)")

        axes[1, 1].set_xlabel(f'{im} level')

        plt.suptitle(title)

        plt.subplots_adjust(left=0.08, right=0.99, bottom=0.1, wspace=0.0, hspace=0.0)

        plt.savefig(
            plot_output_directory / f"{title}_individual_realizations.png",
            dpi=plot_dpi)

    print()








    #
    #
    #
    #
    # individual_realization_df = ds.dataset(source=results_directory/"individual_realizations",
    #                                        format="parquet").to_table().to_pandas()
    #
    # individual_realizations_needed_indices = (individual_realization_df["hazard_model_id"] == results_directory.name) & \
    #                  (individual_realization_df["nloc_001"] == locations_nloc_dict[location_short_name])
    #
    # filtered_individual_realization_df = individual_realization_df[individual_realizations_needed_indices]
    #
    # realization_names = plotting_helpers.lookup_realization_name_from_hash(filtered_individual_realization_df, registry_directory)
    #
    # hazard_rate_array = np.zeros((len(filtered_individual_realization_df),44))
    #
    # for realization_index in range(len(filtered_individual_realization_df)):
    #     hazard_rate_array[realization_index,:] = filtered_individual_realization_df.iloc[realization_index]["branches_hazard_rates"]
    #
    # ### Convert the rate to annual probability of exceedance
    # hazard_prob_of_exceedance = calculators.rate_to_prob(hazard_rate_array, 1.0)
    #
    # resulting_hazard_curves = plotting_helpers.load_aggregate_stats_for_all_logic_trees_in_directory(results_directory.parent)
    #
    # agg_stats_df = resulting_hazard_curves.data_df
    #
    #
    #









def make_figures_of_all_individual_realizations(srm_or_gmcm:str,
                                                results_directory:Union[Path, str],
                                                plot_output_directory:Union[Path, str],
                                                registry_directory:Union[Path, str],
                                                locations: list[str] = ["AKL", "WLG", "CHC"],
                                                im: str = "PGA",
                                                vs30: int = 400,
                                                im_xlims: tuple = (9e-5, 5),
                                                poe_min_plot: float = 1e-5,
                                                plot_dpi: int = 500):

    if isinstance(results_directory, str):
        results_directory = Path(results_directory)
    if isinstance(plot_output_directory, str):
        plot_output_directory = Path(plot_output_directory)
    if isinstance(registry_directory, str):
        registry_directory = Path(registry_directory)

    if not plot_output_directory.exists():
        plot_output_directory.mkdir(parents=True)

    aggregate_stats_results = plotting_helpers.load_aggregate_stats_for_all_logic_trees_in_directory(results_directory)

    run_notes_df = aggregate_stats_results.run_notes_df

    ### Slab only has one branch so needs to be treated differently
    slab_index = run_notes_df[run_notes_df["slt_note"].str.contains("SLAB")]["logic_tree_index"].values[0]

    for logic_tree_index_dir in results_directory.iterdir():

        print(f"Processing {logic_tree_index_dir.name}")

        if logic_tree_index_dir.is_dir():

            logic_tree_idx = int(logic_tree_index_dir.name.split("_")[-1])

            if logic_tree_idx == slab_index:
                print("Skipping SLAB as it only has one branch")
                continue

            make_figures_of_individual_realizations_for_a_single_logic_tree(srm_or_gmcm=srm_or_gmcm,
                logic_tree_index_dir = logic_tree_index_dir,
                plot_output_directory = plot_output_directory,
                locations = locations,
                registry_directory = registry_directory,
                im = im,
                vs30 = vs30,
                im_xlims = im_xlims,
                poe_min_plot = poe_min_plot,
                plot_dpi = plot_dpi)







    #
    #
    #
    #
    # individual_realization_df = ds.dataset(source=results_directory/"individual_realizations",
    #                                        format="parquet").to_table().to_pandas()
    #
    # individual_realizations_needed_indices = (individual_realization_df["hazard_model_id"] == results_directory.name) & \
    #                  (individual_realization_df["nloc_001"] == locations_nloc_dict[location_short_name])
    #
    # filtered_individual_realization_df = individual_realization_df[individual_realizations_needed_indices]
    #
    # realization_names = plotting_helpers.lookup_realization_name_from_hash(filtered_individual_realization_df, registry_directory)
    #
    # hazard_rate_array = np.zeros((len(filtered_individual_realization_df),44))
    #
    # for realization_index in range(len(filtered_individual_realization_df)):
    #     hazard_rate_array[realization_index,:] = filtered_individual_realization_df.iloc[realization_index]["branches_hazard_rates"]
    #
    # ### Convert the rate to annual probability of exceedance
    # hazard_prob_of_exceedance = calculators.rate_to_prob(hazard_rate_array, 1.0)
    #
    # resulting_hazard_curves = plotting_helpers.load_aggregate_stats_for_all_logic_trees_in_directory(results_directory.parent)
    #
    # agg_stats_df = resulting_hazard_curves.data_df
    #
    #
    #






