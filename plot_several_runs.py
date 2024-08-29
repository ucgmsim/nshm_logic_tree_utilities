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


### tidied up
def load_aggregate_stats_for_one_logic_tree_one_location(results_dir_for_one_logic_tree: Union[Path, str], location: str) -> pd.DataFrame:

    """
    Load aggregate statistics results for a single location.

    Parameters
    ----------
    results_dir_for_one_logic_tree : Union[Path, str]
        The directory containing the results of a run of main.py for a single logic tree. This directory has a
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
        The directory containing the results of a run of main.py for a single logic tree. This directory has a
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


def remove_duplicates_in_x(x, y):
    # Find unique values in x and their indices
    unique_x, indices = np.unique(x, return_index=True)

    # Use indices to filter both x and y
    filtered_x = x[indices]
    filtered_y = y[indices]

    return filtered_x, filtered_y


##################################################################################################
##################################################################################################
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



### Used function
def interpolate_ground_motion_models(data_df, location, im):

    mean_list = []
    std_ln_list = []
    non_zero_run_list = []

    for run in natsort.natsorted(data_df["hazard_model_id"].unique()):

        nloc_001_str = locations_nloc_dict[location]

        run_counter = int(run.split("_")[-1])

        mean = data_df[(data_df["agg"] == "mean") &
                  (data_df["vs30"] == vs30) &
                  (data_df["imt"] == im) &
                  (data_df["hazard_model_id"] == run) &
                  (data_df["nloc_001"] == nloc_001_str)]["values"].values[0]

        mean_max = np.max(mean)
        print(f'run {run} max mean: {mean_max}')

        std_ln = data_df[(data_df["agg"] == "std_ln") &
                  (data_df["vs30"] == vs30) &
                  (data_df["imt"] == im) &
                  (data_df["hazard_model_id"] == run) &
                  (data_df["nloc_001"] == nloc_001_str)]["values"].values[0]

        mean_list.append(mean)
        std_ln_list.append(std_ln)
        non_zero_run_list.append(run)

    mean_array = np.array(mean_list)
    std_ln_array = np.array(std_ln_list)

    num_mean_points = 1000
    mm = np.logspace(-6, -2, num_mean_points)

    num_runs = len(mean_array)

    interp_disp_array = np.zeros((num_runs,num_mean_points))

    ## Interpolation part
    for run_idx in range(num_runs):

        std_ln_vect = std_ln_array[run_idx]
        mean_vect = mean_array[run_idx]

        min_mean_value = 1e-9

        std_ln_vect2 = std_ln_vect[mean_vect > min_mean_value]
        mean_vect2 = mean_vect[mean_vect > min_mean_value]

        mean_vect3, std_ln_vect3 = remove_duplicates_in_x(mean_vect2, std_ln_vect2)

        plot_interpolations = False

        if len(mean_vect3) == 0:
            interp_disp_array[run_idx, :] = np.nan

        else:

            mean_to_dispersion_func = scipy.interpolate.interp1d(mean_vect3, std_ln_vect3, kind='linear',bounds_error=False, fill_value=np.nan)
            #mean_to_dispersion_func = scipy.interpolate.interp1d(mean_vect3, std_ln_vect3, kind='linear')
            interp_disp = mean_to_dispersion_func(mm)
            interp_disp_array[run_idx, :] = interp_disp

            if plot_interpolations:
                plt.figure()

                print(f"run_idx: {run_idx}, run: {logic_tree_index_list[run_idx]}")

                plt.semilogx(mean_vect, std_ln_vect, '.')

                plt.semilogx(mean_vect3, std_ln_vect3, 'r.')
                plt.semilogx(mean_array[run_idx], std_ln_array[run_idx], '.')
                plt.semilogx(mm, interp_disp, 'r--')
                plt.title(f"run_{run_idx} location {location}")
                plt.show()

    return mm, interp_disp_array

### Used function
def get_interpolated_gmms():

    locations = ["AKL", "WLG", "CHC"]
    filter_strs = ["CRU", "HIK_and_PUY", "SLAB"]
    #filter_strs = ["SLAB"]

    dispersion_range_dict = {}

    for location in locations:
        dispersion_range_dict[location] = {}

    for filter_str in filter_strs:

            filtered_run_notes_df = run_notes_df[run_notes_df["slt_note"].str.contains(filter_str)]

            logic_tree_index_list = [f"run_{x}" for x in filtered_run_notes_df["run_counter"].values]

            filtered_data_df = data_df[data_df["hazard_model_id"].isin(logic_tree_index_list)]

            for location in locations:

                mm, interp_disp_array = interpolate_ground_motion_models(filtered_data_df, location, "PGA")

                dispersion_range_dict[location][filter_str] = np.nanmax(interp_disp_array, axis=0) - np.nanmin(interp_disp_array, axis=0)

    return dispersion_range_dict

### Used function
def plot_gmm_dispersion_ranges():

    dispersion_range_dict = get_interpolated_gmms()
    filter_strs = ["CRU", "HIK_and_PUY", "SLAB"]

    linestyle_lookup_dict = {"CRU":"--",
                             "HIK_and_PUY":"-.",
                             "SLAB":":"}

    color_lookup_dict = {"AKL":"blue",
                         "WLG":"orange",
                         "CHC":"red"}

    plt.figure()

    num_mean_points = 1000
    mm = np.logspace(-6, -2, num_mean_points)

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
    plt.ylabel(r'Mean annual hazard probability, $\mu_{P(IM=im)}$')
    plt.xlabel(r'Range in dispersion in hazard probability, $\sigma_{\ln P(IM=im)}$')
    plt.grid(linestyle='--')
    plt.savefig("/home/arr65/data/nshm/output_plots/dispersion_range_plot.png", dpi=500)

    #plt.show()
    print()









    # #filtered_df = run_notes_df[~run_notes_df["slt_note"].str.contains("only")]
    # filtered_run_notes_df = run_notes_df[run_notes_df["slt_note"].str.contains("CRU")]
    #
    # logic_tree_index_list = [f"run_{x}" for x in filtered_run_notes_df["run_counter"].values]
    #
    # filtered_data_df = data_df[data_df["hazard_model_id"].isin(logic_tree_index_list)]
    #
    # print()
    #
    # mm, interp_disp_array = interpolate_ground_motion_models(filtered_data_df, "WLG", "PGA")
    #
    # plt.semilogx(mm, np.nanmean(interp_disp_array, axis=0), 'r--')
    # plt.show()
    #
    # print()


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

    loaded_run_results = load_aggregate_stats_for_all_logic_trees_in_directory(results_directory, locations)
    data_df = loaded_run_results.data_df
    run_notes_df = loaded_run_results.run_notes_df

    plt.close("all")
    fig, axes = plt.subplots(3, 3,figsize=(6,9))

    sorted_logic_tree_indices = sort_logic_tree_index_by_gmcm_model_name(results_directory)

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

    Note that the data that this function uses is loaded from the output of main.py so that needs to be run first.
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
    plot_row_to_data_lookup = {0:load_aggregate_stats_for_all_logic_trees_in_directory(gmcm_models_data_directory),
                               1:load_aggregate_stats_for_all_logic_trees_in_directory(gmcm_models_data_directory),
                               2:load_aggregate_stats_for_all_logic_trees_in_directory(srm_models_data_directory)}

    ## Sort the logic_tree_index_[x] names by the ground motion characterization model that
    ## was isolated by that logic tree.
    sorted_gmm_run_nums = sort_logic_tree_index_by_gmcm_model_name(gmcm_models_data_directory)

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
    fig_margins = convert_edge_margin_in_pixels_to_fraction(fig,
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

    resulting_hazard_curves = load_aggregate_stats_for_all_logic_trees_in_directory(results_directory)

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

## A good plotting function
def do_srm_model_plots_with_seperate_location_subplots(im):

    trt_short_to_long = {"CRU":"crust",
                         "INTER":"subduction interface\n"}

    model_name_short_to_long = {"deformation_model":"deformation model\n(geologic or geodetic)",
                                "time_dependence":"time dependence\n(time-dependent or time-independent)",
                                "MFD":"magnitude frequency distribution",
                                "moment_rate_scaling":"moment rate scaling"}


    #filtered_df = run_notes_df[~run_notes_df["slt_note"].str.contains("INTER_only")]

    ## Filter out the slab as well with just "only"
    filtered_df = run_notes_df[~run_notes_df["slt_note"].str.contains("only")]

    logic_tree_index_list = [f"run_{x}" for x in filtered_df["run_counter"].values]

    fig, axes = plt.subplots(1, 3, figsize=(8,4))

    linestyle_lookup_dict = {"CRU":"-", "INTER":"--"}

    for location_idx, location in enumerate(locations):

        for run in logic_tree_index_list:

            nloc_001_str = locations_nloc_dict[location]

            run_counter = int(run.split("_")[-1])

            mean = data_df[(data_df["agg"] == "mean") &
                      (data_df["vs30"] == vs30) &
                      (data_df["imt"] == im) &
                      (data_df["hazard_model_id"] == run) &
                      (data_df["nloc_001"] == nloc_001_str)]["values"].values[0]

            std_ln = data_df[(data_df["agg"] == "std_ln") &
                      (data_df["vs30"] == vs30) &
                      (data_df["imt"] == im) &
                      (data_df["hazard_model_id"] == run) &
                      (data_df["nloc_001"] == nloc_001_str)]["values"].values[0]

            needed_idx = mean > 1e-8
            mean = mean[needed_idx]
            std_ln = std_ln[needed_idx]

            slt_note = f"{run_notes_df[run_notes_df["run_counter"] == run_counter]["slt_note"].values[0]}"
            #glt_note = f"{run_notes_df[run_notes_df["run_counter"]==run_counter]["glt_note"].values[0]}"

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

    #fig.suptitle(f'Fixed: IM={im}, Vs30 = 400 m/s')

    plt.subplots_adjust(wspace=0.0,left=0.1,right=0.99,top=0.94)
    plt.savefig(plot_output_directory / f"{auto_dir.name}_source_mean_vs_dispersion.png",dpi=1000)

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


## tidied up
@dataclass
class RealizationName():
    """
    Class to store the names of the seismicity rate model and ground motion
    characterization model used in a realization.
    """
    seismicity_rate_model_id: str
    ground_motion_characterization_models_id: str
    

### tidied up
def lookup_realization_name_from_hash(individual_realization_df: pd.DataFrame,
                                      registry_directory:Union[Path,str]) -> list[RealizationName]:

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

    realization_names = lookup_realization_name_from_hash(filtered_individual_realization_df, registry_directory)

    hazard_rate_array = np.zeros((len(filtered_individual_realization_df),44))

    for realization_index in range(len(filtered_individual_realization_df)):
        hazard_rate_array[realization_index,:] = filtered_individual_realization_df.iloc[realization_index]["branches_hazard_rates"]

    ### Convert the rate to annual probability of exceedance
    hazard_prob_of_exceedance = calculators.rate_to_prob(hazard_rate_array, 1.0)

    resulting_hazard_curves = load_aggregate_stats_for_all_logic_trees_in_directory(results_directory.parent)

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



if __name__ == "__main__":

    # get_interpolated_gmms()

    #plot_gmm_dispersion_ranges()

    #do_srm_model_plots_with_seperate_location_subplots("PGA")

    make_figure_of_gmcms(results_directory = "/home/arr65/data/nshm/output/gmcm_models",
                         plot_output_directory = "/home/arr65/data/nshm/plots",
                         locations = ["AKL", "WLG", "CHC"])


    print()

    # make_figure_showing_Bradley2009_method(results_directory = "/home/arr65/data/nshm/output/gmcm_models/logic_tree_index_3",
    #                                 plot_output_directory = "/home/arr65/data/nshm/plots",
    #                                 registry_directory = "/home/arr65/src/gns/modified_gns/nzshm-model/resources",
    #                                 location_short_name = "WLG",
    #                                 vs30 = 400,
    #                                 im = "PGA")

    #print()




    ### use autorun15 for these plots

    make_figure_of_coefficient_of_variation(results_directory="/home/arr65/data/nshm/output/full_component_logic_trees",
                                         plot_output_directory="/home/arr65/data/nshm/plots")


    #print()

    ### use autorun21 for these plots
    # run_list_sorted = get_alphabetical_run_list()
    # for location in ["AKL", "WLG", "CHC"]:
    #     do_gmcm_plots_with_seperate_tectonic_region_type(run_list_sorted, location, "PGA")

    #make_figure_of_gmcm(21)

    #do_plot_for_poster()

    #make_srm_and_gmcm_model_dispersions_figure(["WLG", "CHC"])
    #make_srm_and_gmcm_model_dispersions_figure(["WLG","CHC","AKL"])
    make_figure_of_srm_and_gmcm_model_dispersions(
        locations=["WLG","CHC"],#,"AKL"],
        srm_models_data_directory="/home/arr65/data/nshm/output/srm_models",
        gmcm_models_data_directory="/home/arr65/data/nshm/output/gmcm_models",
        plot_output_directory="/home/arr65/data/nshm/plots")

    #make_explanation_plot_for_poster(21,3)

    print()

    #range_dispersions = np.nanmax(interp_disp_array, axis=0) - np.nanmin(interp_disp_array, axis=0)

    # plt.figure()
    # plt.semilogx(mm, range_dispersions, 'r--')
    # plt.show()

    #do_plots_with_seperate_tectonic_region_type_per_location("CHC", "PGA")

    #print()
    #do_plots(over_plot_all=True)
    #do_plots(over_plot_all=False)

    #do_plots_with_seperate_location_subplots(over_plot_all=True)
    #do_plots_with_seperate_location_subplots(over_plot_all=False)
