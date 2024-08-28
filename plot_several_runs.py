from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path
import pandas as pd
import scipy
import pyarrow.dataset as ds

from cycler import cycler
import natsort
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt, ticker as mticker
import toshi_hazard_post.calculators as calculators


import toml

nshm_im_levels = np.loadtxt("resources/nshm_im_levels.txt")

five_colors = ['b', 'g', 'r', 'c', 'm']

colors = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#7f7f7f',  # Gray
    '#bcbd22',  # Olive
    '#17becf',  # Cyan
    '#1f78b4',  # Dark Blue
    '#33a02c',  # Dark Green
    '#e31a1c',  # Dark Red
    '#ff7f00',  # Dark Orange
    '#6a3d9a',  # Dark Purple
    '#b15928',  # Dark Brown
    '#a6cee3',  # Light Blue
    '#b2df8a',  # Light Green
    '#fb9a99',  # Light Red
    '#fdbf6f'   # Light Orange
]

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



custom_cycler = (cycler(color=colors) *
                  cycler(linestyle=['-', '--', ':', '-.']))

# custom_cycler = (cycler(color=colors) *
#                   cycler(linestyle=['-', '--', '-.']))

# default_cycler = (cycler(color=['r', 'g', 'b', 'y']) +
#                   cycler(linestyle=['-', '--', ':', '-.']))



custom_cycler_slt_nth_branch = (cycler(linestyle=['--', ':', '-.'])*
                                  cycler(color=five_colors))


custom_cycler_location_subplot = (cycler(linestyle=['--', ':', '-.'])*
                                  cycler(color=colors))
def load_locations_from_run(output_dir: Path, locations: list[str]) -> pd.DataFrame:
    results_df = pd.DataFrame()

    for location in locations:

        results_df = (pd.concat
                      ([results_df,
                        load_location_from_run(output_dir, location)],
                       ignore_index=True))

    return results_df

@dataclass
class LoadedRunResults():

    data_df :  pd.DataFrame()
    run_notes_df : pd.DataFrame()



def load_toshi_hazard_post_agg_stats_in_rungroup(output_dir: Path,
                                                 locations:list[str]=["AKL","WLG","CHC"]) -> LoadedRunResults:

    results_df = pd.DataFrame()

    for run_dir in output_dir.iterdir():
        if run_dir.is_dir():
            results_df = (pd.concat
                          ([results_df,
                            load_locations_from_run(run_dir,locations)],
                           ignore_index=True))

    return LoadedRunResults(data_df=results_df, run_notes_df=pd.read_csv(output_dir / "run_notes.csv"))




def load_location_from_run(output_dir: Path, location: str) -> pd.DataFrame:
    results_df = pd.DataFrame()

    if location not in ["AKL","WLG","CHC"]:
        raise ValueError('location must be AKL, WLG or CHC')

    if location == 'CHC':
        nloc_str = "nloc_0=-44.0~173.0"
    if location == 'WLG':
        nloc_str = "nloc_0=-41.0~175.0"
    if location == 'AKL':
        nloc_str = "nloc_0=-37.0~175.0"

    results_dir = output_dir / nloc_str

    for index, file in enumerate(results_dir.glob('*.parquet')):

        results_df = pd.concat([results_df, pd.read_parquet(file)], ignore_index=True)

    results_df = insert_ln_std(results_df)

    return results_df

def insert_ln_std(data_df):

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



#auto_dir = Path("/home/arr65/data/nshm/auto_output/auto1")
#auto_dir = Path("/home/arr65/data/nshm/auto_output/auto2")
#auto_dir = Path("/home/arr65/data/nshm/auto_output/auto3")
#auto_dir = Path("/home/arr65/data/nshm/auto_output/auto4")
#auto_dir = Path("/home/arr65/data/nshm/auto_output/auto5")
#auto_dir = Path("/home/arr65/data/nshm/auto_output/auto6")
#auto_dir = Path("/home/arr65/data/nshm/auto_output/auto7")
#auto_dir = Path("/home/arr65/data/nshm/auto_output/auto8")
#auto_dir = Path("/home/arr65/data/nshm/auto_output/auto9")
#auto_dir = Path("/home/arr65/data/nshm/auto_output/auto11")
#auto_dir = Path("/home/arr65/data/nshm/auto_output/auto12")
#auto_dir = Path("/home/arr65/data/nshm/auto_output/auto13")
#auto_dir = Path("/home/arr65/data/nshm/auto_output/auto14")
#auto_dir = Path("/home/arr65/data/nshm/auto_output/auto15")

#auto_dir = Path("/home/arr65/data/nshm/auto_output/auto17")

#auto_dir = Path("/home/arr65/data/nshm/auto_output/auto18")

#auto_dir = Path("/home/arr65/data/nshm/auto_output/auto20")   ## For making the main SRM dispersion plots

# auto_dir = Path("/home/arr65/data/nshm/auto_output/auto21")  ## For making the main GMCM dispersion plots

#auto_dir = Path("/home/arr65/data/nshm/auto_output/auto23")

# data_df = load_toshi_hazard_post_agg_stats_in_rungroup(auto_dir)



# vs30 = 400
#
#
# ims = ["PGA"]


# locations = ["AKL","WLG","CHC"]
#locations = ["WLG"]


# linestyle_dict = {
#     "CRU": "--",
#     "INTER": "-.",
#     "SLAB": ":"
# }


#


# location_to_full_location = {"AKL": "Auckland",
#                              "WLG": "Wellington",
#                              "CHC": "Christchurch"}



##################################################################################################
##################################################################################################
### Used function


def get_runs_sorted_by_model_name(rungroup_num,
                                  results_dir: Path = Path("/home/arr65/data/nshm/auto_output")):

    """
    Sort the runs in a rungroup by the model name that was isolated in that run.
    """

    run_list_label_tuple_list = []

    run_notes_df = pd.read_csv(results_dir / f"auto{rungroup_num}/run_notes.csv")

    run_nums = run_notes_df["run_counter"]


    for run_counter in run_nums:

        slt_note = f"{run_notes_df[run_notes_df["run_counter"] == run_counter]["slt_note"].values[0]}"
        glt_note = f"{run_notes_df[run_notes_df["run_counter"]==run_counter]["glt_note"].values[0]}"

        trts_from_note = slt_note.split(">")[-2].strip().split(":")[-1].strip("[]")
        glt_model_and_weight_str = glt_note.split(">")[-2].strip(" []")
        glt_model = glt_model_and_weight_str.split("*")[0]

        if "NZNSHM2022_" in glt_model:
            glt_model = glt_model.split("NZNSHM2022_")[1]

        ## Get tuples of (run_number, corresponding model name)
        run_list_label_tuple_list.append((run_counter, f"{trts_from_note}_{glt_model}"))

    sorted_run_list_label_tuple_list = natsort.natsorted(run_list_label_tuple_list, key=lambda x: x[1])

    print()

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

                print(f"run_idx: {run_idx}, run: {run_list[run_idx]}")

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

            run_list = [f"run_{x}" for x in filtered_run_notes_df["run_counter"].values]

            filtered_data_df = data_df[data_df["hazard_model_id"].isin(run_list)]

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
    # run_list = [f"run_{x}" for x in filtered_run_notes_df["run_counter"].values]
    #
    # filtered_data_df = data_df[data_df["hazard_model_id"].isin(run_list)]
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
def do_big_gmcm_subplot(run_num: int,
                        locations : list[str] = ["AKL", "WLG", "CHC"],
                        vs30 = 400,
                        im = "PGA",
                        plot_output_dir : Path = Path("/home/arr65/data/nshm/output_plots")):

    locations_nloc_dict = toml.load('resources/location_code_to_nloc_str.toml')
    tectonic_type_to_linestyle = toml.load('resources/tectonic_region_type_to_linestyle.toml')
    location_to_full_location = toml.load('resources/location_code_to_full_name.toml')
    model_to_plot_label = toml.load('resources/gmcm_name_plot_format.toml')
    glt_model_color = toml.load('resources/gmcm_plot_colors.toml')

    auto_dir = Path(f"/home/arr65/data/nshm/auto_output/auto{run_num}")

    loaded_run_results = load_toshi_hazard_post_agg_stats_in_rungroup(auto_dir)
    data_df = loaded_run_results.data_df
    run_notes_df = loaded_run_results.run_notes_df

    run_list_sorted_crust = get_runs_sorted_by_model_name(run_num)

    print()

    plt.close("all")
    fig, axes = plt.subplots(3, 3,figsize=(6,9))

    for location_row_idx, location in enumerate(locations):

        nloc_001_str = locations_nloc_dict[location]

        mean_list = []
        std_ln_list = []
        non_zero_run_list = []

        for run_counter in run_list_sorted:

            run = f"run_{run_counter}"

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
            slt_note = f"{run_notes_df[run_notes_df["run_counter"] == run_counter]["slt_note"].values[0]}"
            glt_note = f"{run_notes_df[run_notes_df["run_counter"]==run_counter]["glt_note"].values[0]}"

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
                #axes[0,1].set_title(f"{location_to_full_location[location]}\nsubduction interface", fontsize=11)
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
    plt.savefig(plot_output_dir / f"gmm_{auto_dir.name}_{im}_all_locations.png",dpi=500)

    return fig

def make_figure_srm_and_gmcm_model_dispersions(locations=["WLG", "CHC","AKL"],
                                               srm_models_group_run_num=20,
                                               gmcm_models_group_run_num=21,
                                               vs30=400, im="PGA"):

    """
    Makes a figure containing subplots of mean prediction on the vertical axis and the dispersion in
    predictions on the horizontal axis, following Bradley (2009).  All subplots are for a given vs30 value and
    intensity measure (im). Each column of subplots is for a different location so the figure will have a number of
    columns equal to the length of the locations list.  The figure will always have 3 rows of subplots.  The top row
    shows the crustal ground motion characterization models (GMCMs), the middle row shows the interface and intraslab
    GMCMs, and the bottom row shows the seismicity rate model (SRM) components.



    """

    num_plot_cols = len(locations)

    locations_nloc_dict = toml.load('resources/location_code_to_nloc_str.toml')
    location_to_full_location = toml.load('resources/location_code_to_full_name.toml')
    model_to_plot_label = toml.load('resources/model_name_lookup_for_plot.toml')
    glt_model_color = toml.load('resources/model_plot_colors.toml')

    autoruns = [srm_models_group_run_num, gmcm_models_group_run_num]

    autorun_num_to_data = {}
    for autorun in autoruns:
        auto_dir = Path(f"/home/arr65/data/nshm/auto_output/auto{autorun}")
        loaded_run_results = load_toshi_hazard_post_agg_stats_in_rungroup(auto_dir)
        autorun_num_to_data[autorun] = loaded_run_results

    ## relate plot row to data
    data_lookup_dict = {0:autorun_num_to_data[gmcm_models_group_run_num],
                        1:autorun_num_to_data[gmcm_models_group_run_num],
                        2:srm_models_group_run_num[20]}

    ## Get a list of
    sorted_gmm_run_nums = get_runs_sorted_by_model_name(21)

    needed_runs_dict = {}

    ## Relate the
    for row_index in range(3):
        if row_index == 0:
            needed_runs_dict[row_index] = data_lookup_dict[row_index].run_notes_df[data_lookup_dict[row_index].run_notes_df["slt_note"].str.contains("CRU")]["run_counter"]
        if row_index == 1:
            needed_runs_dict[row_index] = data_lookup_dict[row_index].run_notes_df[data_lookup_dict[row_index].run_notes_df["slt_note"].str.contains("INTER_HIK_and_PUY|SLAB")]["run_counter"]
        if row_index == 2:
            needed_runs_dict[row_index] = data_lookup_dict[row_index].run_notes_df[data_lookup_dict[row_index].run_notes_df["slt_note"].str.contains("CRU|INTER_HIK_and_PUY")]["run_counter"]


    ####################################################

    title_font_size = 12

    plt.close("all")
    #fig, axes = plt.subplots(3, num_plot_cols, figsize=(6, 9))
    fig, axes = plt.subplots(3, num_plot_cols, figsize=(3*num_plot_cols, 9))

    for row_index in range(3):

        for column_index in range(num_plot_cols):

            axes[row_index, column_index].set_prop_cycle(None)

            plot_location = locations[column_index]
            if row_index == 0:
                axes[row_index, column_index].set_title(location_to_full_location[plot_location],
                                                        fontsize=title_font_size)

            if row_index != 2:
                sorted_run_nums = sorted_gmm_run_nums
            if row_index == 2:
                sorted_run_nums = needed_runs_dict[row_index]

            for sorted_run_num in sorted_run_nums:

                if sorted_run_num in needed_runs_dict[row_index]:

                    run = f"run_{sorted_run_num}"

                    if row_index in [0,1]:
                        run_note = data_lookup_dict[row_index].run_notes_df[data_lookup_dict[row_index].run_notes_df["run_counter"] == sorted_run_num]["glt_note"].values[0]
                        short_note = run_note.split(">")[-2].split("*")[-2].strip(" [")
                        plot_label = model_to_plot_label[short_note]

                    if row_index == 2:
                        run_note = data_lookup_dict[row_index].run_notes_df[data_lookup_dict[row_index].run_notes_df["run_counter"] == sorted_run_num]["slt_note"].values[0]
                        short_note = run_note.split(">")[1].split(":")[-1].strip(" []") + "_" +\
                                     run_note.split(">")[2].strip()

                        plot_label = model_to_plot_label[short_note]

                    mean = data_lookup_dict[row_index].data_df[(data_lookup_dict[row_index].data_df["agg"] == "mean") &
                                                 (data_lookup_dict[row_index].data_df["vs30"] == vs30) &
                                                 (data_lookup_dict[row_index].data_df["imt"] == im) &
                                                 (data_lookup_dict[row_index].data_df["hazard_model_id"] == run) &
                                                 (data_lookup_dict[row_index].data_df["nloc_001"] == locations_nloc_dict[plot_location])]["values"].values[0]

                    std_ln = data_lookup_dict[row_index].data_df[(data_lookup_dict[row_index].data_df["agg"] == "std_ln") &
                                                 (data_lookup_dict[row_index].data_df["vs30"] == vs30) &
                                                 (data_lookup_dict[row_index].data_df["imt"] == im) &
                                                 (data_lookup_dict[row_index].data_df["hazard_model_id"] == run) &
                                                 (data_lookup_dict[row_index].data_df["nloc_001"] == locations_nloc_dict[plot_location])]["values"].values[0]
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

    ### Adjust the figure margins before using the figure axes positions to place text
    fig_margins = convert_edge_margin_in_pixels_to_fraction(fig,
                                                            100,
                                                            5,
                                                            45,
                                                            30)

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
    fig.text(row_titles_x0, axes[0, 0].get_position().y0,'Ground Motion Characterization Models', ha='center', va='center',rotation=90, fontsize=title_font_size)

    fig.text(row_titles_x0, (axes[2, 0].get_position().y0 + axes[2, 0].get_position().y1)/2.0,
             'Seismicity Rate Models',
             ha='center', va='center', rotation=90, fontsize=title_font_size)


    #plt.show()
    plt.savefig(f"/home/arr65/data/nshm/output_plots/{"_".join(locations)}_dispersion_poster_plot.png", dpi=500)
    print()



## A good plotting function
def make_cov_plots(rungroup_num = 15, location="WLG", im="PGA", vs30=400):

    rungroup_dir = Path(f"/home/arr65/data/nshm/auto_output/auto{rungroup_num}")

    locations_nloc_dict = toml.load('resources/location_code_to_nloc_str.toml')

    nloc_001_str = locations_nloc_dict[location]

    mean_list = []
    cov_list = []

    run_results = load_toshi_hazard_post_agg_stats_in_rungroup(rungroup_dir)

    data_df = run_results.data_df
    run_notes_df = run_results.run_notes_df

    run_list = [f"run_{x}" for x in run_notes_df["run_counter"].values]
    for run_idx, run in enumerate(run_list):

        mean = data_df[(data_df["agg"] == "mean") &
                  (data_df["vs30"] == vs30) &
                  (data_df["imt"] == im) &
                  (data_df["hazard_model_id"] == run) &
                  (data_df["nloc_001"] == nloc_001_str)]["values"].values[0]

        cov = data_df[(data_df["agg"] == "cov") &
                  (data_df["vs30"] == vs30) &
                  (data_df["imt"] == im) &
                  (data_df["hazard_model_id"] == run) &
                  (data_df["nloc_001"] == nloc_001_str)]["values"].values[0]

        mean_list.append(mean)
        cov_list.append(cov)

    plt.rcParams.update({'font.size': 12})

    lw = 5

    #plt.figure(figsize=(5.12,4.62))
    plt.figure(figsize=(7.3, 4.62))
    plt.semilogx(nshm_im_levels, cov_list[0], linestyle='--', linewidth=lw, label='source model')
    plt.semilogx(nshm_im_levels, cov_list[1], linestyle='-.', linewidth=lw, label='ground motion model')
    plt.semilogx(nshm_im_levels, cov_list[2], linestyle='-', linewidth=lw, label='both')
    plt.legend(handlelength=4)
    #plt.title(f"{location} {im}")
    #plt.ylabel("coefficient of variation (CoV) of\nannual probability of exceedance (APoE)")
    plt.ylabel("Modelling uncertainty\n(coefficient of variation of model predictions)")
    plt.xlabel('Peak ground acceleration (g)')
    plt.xlim(1e-2,5)
    plt.ylim(0.05,0.8)

    plt.grid(which='major',
           linestyle='--',
           linewidth='0.5',
           color='black',
           alpha=0.6)

    #plt.tight_layout()
    plt.subplots_adjust(left=0.11, right=0.99, bottom=0.12, top=0.97 )
    plt.savefig("/home/arr65/data/nshm/output_plots/cov_plot.png",dpi=500)
    print()





#
# plt.close()
#
# plt.figure()
# plt.semilogx(nshm_im_levels, std_list[2], linestyle='-', label='SRM & GMCM')
# plt.semilogx(nshm_im_levels, std_list[0], linestyle='--', label='SRM')
# plt.semilogx(nshm_im_levels, std_list[1], linestyle='-.', label='GMCM')
# plt.legend()
# #plt.title(f"{location} {im}")
# #plt.title("Wellington assuming Vs30 = 400 m/s")
# plt.ylabel("standard deviation of annual\nprobability of exceedance (APoE)")
# plt.xlabel('peak ground acceleration (g)')
# plt.tight_layout()
# plt.savefig("/home/arr65/data/nshm/output_plots/std_plot.png",dpi=400)

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

    run_list = [f"run_{x}" for x in filtered_df["run_counter"].values]

    fig, axes = plt.subplots(1, 3, figsize=(8,4))

    linestyle_lookup_dict = {"CRU":"-", "INTER":"--"}

    for location_idx, location in enumerate(locations):

        for run in run_list:

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
    plt.savefig(plot_output_dir / f"{auto_dir.name}_source_mean_vs_dispersion.png",dpi=1000)

    # pdf = PdfPages(plot_output_dir / f"{auto_dir.name}_source_mean_vs_dispersion.pdf")
    # pdf.savefig(fig)
    # pdf.close()

######################################
##############################
### Old plotting functions

def do_plots(over_plot_all=False):

    color_lookup_dict = {"AKL":"blue","WLG":"orange","CHC":"red"}
    linestyle_lookup_dict = {"run_0":"-", "run_1":"--", "run_2":":"}

    label_lookup_dict = {"run_0":"full SRM and full GMCM",
                         "run_1":"single highest weighted SRM branch and full GMCM",
                         "run_2":"full SRM & single highest weighted GMCM branch"}



    for im in ims:
        num_hazard_curves_on_plot = 0

        if not over_plot_all:
            plt.close("all")

        #plt.rc('axes', prop_cycle=custom_cycler_slt_nth_branch)
        for location in locations:

            nloc_001_str = locations_nloc_dict[location]

            for run in run_list:

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

                # plot_label = (f"{location} {run_notes_df[run_notes_df["run_counter"]==run_counter]["slt_note"].values[0]}, "
                #               f"{run_notes_df[run_notes_df["run_counter"]==run_counter]['glt_note'].values[0]}")

                plot_label = f"{location} {label_lookup_dict[run]}"

                print()



                plt.semilogy(std_ln, mean, color=color_lookup_dict[location],linestyle=linestyle_lookup_dict[run], label=plot_label)
                print(f"plotting: {im} {location} {run}")
                num_hazard_curves_on_plot += 1

        plt.ylim(1e-5,1e0)
        plt.xlim(-0.01, 0.7)


        plt.xlabel(r'Dispersion in hazard probability, $\sigma_{\ln P(IM=im)}$')
        plt.ylabel(r'Mean annual hazard probability, $\mu_{P(IM=im)}$')
        plt.grid(which='major', linestyle='--', linewidth='0.5', color='black')




        if not over_plot_all:

            #plt.title(f'Fixed: IM={im}, Vs30 = 400 m/s')
            plt.legend(prop={'size': 5})
            plt.savefig(plot_output_dir / f"{auto_dir.name}_{im}.png", dpi=500)
            #pdf_all_ims.savefig()


    if over_plot_all:
        plt.title(f'All IMs, fixed Vs30 = 400 m/s')
        #pdf_all_ims.savefig()


def do_plots_with_seperate_location_subplots(over_plot_all=False):

    #plt.rc('axes', prop_cycle=custom_cycler_location_subplot)

    if over_plot_all:
        fig, axes = plt.subplots(1, 3)
        plt.subplots_adjust(wspace=0.0)

    for im in ims:

        if not over_plot_all:
            fig, axes = plt.subplots(1, 3)
            plt.subplots_adjust(wspace=0.0)

        for location_idx, location in enumerate(locations):

            if not over_plot_all:
                plt.close("all")

            for run in run_list:

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

                slt_note = f"{run_notes_df[run_notes_df["run_counter"] == run_counter]["slt_note"].values[0]}"
                glt_note = f"{run_notes_df[run_notes_df["run_counter"]==run_counter]["glt_note"].values[0]}"

                trts_from_note = slt_note.split(">")[-2].strip().split(":")[-1].strip("[]")
                glt_model_and_weight_str = glt_note.split(">")[-2].strip(" []")
                glt_model = glt_model_and_weight_str.split("*")[0]
                glt_model_weight = 1/float(glt_model_and_weight_str.split("*")[1])


                #plot_label_short = plot_label.split(">")[-2].strip().split(":")[-1].strip("[]")

                #plot_label_short = f"{trts_from_note} {glt_model_and_weight_str}"

                plot_label_short = f"{trts_from_note} {glt_model} (w = {glt_model_weight:.3f})"

                # if plot_label_short != "INTER HIK":
                #     continue

                if ("HIK" in plot_label_short) or ("PUY" in plot_label_short):
                    ## only plot the inteface both
                    continue

                if "CRU" in plot_label_short:
                    linestyle = '--'
                if "INTER" in plot_label_short:
                    linestyle = "-."
                if "SLAB" in plot_label_short:
                    linestyle = ":"

                plot_label = plot_label_short

                if not over_plot_all:
                    axes[location_idx].semilogy(std_ln, mean, label=plot_label,
                                                linestyle=linestyle)

                axes[location_idx].set_ylim(1e-5,0.6)
                axes[location_idx].set_xlim(-0.01, 0.7)
                axes[location_idx].set_title(location)

                if location_idx == 0:
                    axes[location_idx].set_ylabel(r'Mean annual hazard probability, $\mu_{P(IM=im)}$')

                if location_idx == 1:
                    axes[location_idx].set_xlabel(r'Dispersion in hazard probability, $\sigma_{\ln P(IM=im)}$')
                    axes[location_idx].set_yticklabels([])
                if location_idx == 2:
                    axes[location_idx].set_yticklabels([])

            axes[location_idx].grid(which='major', linestyle='--', linewidth='0.5', color='black')
            if not over_plot_all:
                #if location_idx == 0:
                axes[location_idx].legend(loc="lower left", prop={'size': 3},
                                              handlelength=5)
            if over_plot_all:
                axes[location_idx].legend(loc="lower left", prop={'size': 3},
                                          handlelength=5, ncol=4)

            if over_plot_all:
                axes[location_idx].semilogy(std_ln, mean, label=f"{im} " + plot_label,
                                            linestyle=linestyle)

        if not over_plot_all:
            fig.suptitle(f'Fixed: IM={im}, Vs30 = 400 m/s')
            pdf_all_ims.savefig(fig)

    if over_plot_all:
        fig.suptitle(f'All IMs, Vs30 = 400 m/s')
        pdf_all_ims.savefig(fig)

def remove_special_characters(s):
    chars_to_remove = ["'", "[", "]", '"']

    translation_table = str.maketrans('', '', ''.join(chars_to_remove))
    return s.translate(translation_table)

@dataclass
class RealizationName():
    """
    Class to store the names of the seismicity rate model and ground motion
    characterization model used in a realization.
    """
    seismicity_rate_model_id: str
    ground_motion_characterization_models_id: str
    

def lookup_realization_name_from_hash(individual_realization_df):

    """

    Looks up the models used in the realization based on the branch hash ids in the output parquet file.

    Parameters
    ----------
    individual_realization_df : pd.DataFrame
        Dataframe containing the individual realizations
        (produced by the modified version of toshi_hazard_post with output_individual_realizations == True)

    Returns
    -------
    realization_names : List[RealizationName]
        List of RealizationName objects containing the seismicity rate model and ground motion characterization model ids

    """

    registry_dir = Path("/home/arr65/src/gns/modified_gns/nzshm-model/resources")
    #registry_dir = Path("/home/arr65/src/gns/nzshm-model/resources")
    gmm_registry_df = pd.read_csv(registry_dir / 'gmm_branches.csv')
    source_registry_df = pd.read_csv(registry_dir / 'source_branches.csv')

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
            
    realization_names = [RealizationName(seismicity_rate_model_id, ground_motion_characterization_models_id) for seismicity_rate_model_id, ground_motion_characterization_models_id in zip(seismicity_rate_model_ids, ground_motion_characterization_models_ids)]

    return realization_names

def load_individual_realizations(rungroup_num, run_num):

    realizations_path = Path(f"/home/arr65/data/nshm/auto_output/auto{rungroup_num}/run_{run_num}/individual_realizations")

    return ds.dataset(source=realizations_path, format="parquet").to_table().to_pandas()


def make_explanation_plot_for_poster(rungroup_num, run_num, loc_name="WLG",
                                     vs30=400, im="PGA"):


    run_name = f"run_{run_num}"

    plot_label_lookup = {"Stafford2022(mu_branch=Upper)":"Upper branch of Stafford 2022",
                        "Stafford2022(mu_branch=Central)":"Central branch of Stafford 2022",
                        "Stafford2022(mu_branch=Lower)":"Lower branch of Stafford 2022"}

    locations_nloc_dict = toml.load('resources/location_code_to_nloc_str.toml')

    plot_colors = ["tab:purple", "tab:orange", "tab:green"]
    plot_linestyles = [":", "-", "--"]

    individual_realization_df = load_individual_realizations(rungroup_num, run_num)

    individual_realizations_needed_indices = (individual_realization_df["hazard_model_id"] == run_name) & \
                     (individual_realization_df["nloc_001"] == locations_nloc_dict[loc_name])

    filtered_individual_realization_df = individual_realization_df[individual_realizations_needed_indices]

    realization_names = lookup_realization_name_from_hash(filtered_individual_realization_df)

    hazard_rate_array = np.zeros((len(filtered_individual_realization_df),44))

    for realization_index in range(len(filtered_individual_realization_df)):
        hazard_rate_array[realization_index,:] = filtered_individual_realization_df.iloc[realization_index]["branches_hazard_rates"]

    hazard_prob_of_exceedance = calculators.rate_to_prob(hazard_rate_array, 1.0)

    
    run_results = load_toshi_hazard_post_agg_stats_in_rungroup(Path("/home/arr65/data/nshm/auto_output/auto21"))

    agg_stats_df = run_results.data_df

    run_name = f"run_{run_num}"

    mean = agg_stats_df[(agg_stats_df["agg"] == "mean") &
                   (agg_stats_df["vs30"] == vs30) &
                   (agg_stats_df["imt"] == im) &
                   (agg_stats_df["hazard_model_id"] == run_name) &
                   (agg_stats_df["nloc_001"] == locations_nloc_dict[loc_name])]["values"].values[0]


    std_ln = agg_stats_df[(agg_stats_df["agg"] == "std_ln") &
                     (agg_stats_df["vs30"] == vs30) &
                     (agg_stats_df["imt"] == im) &
                     (agg_stats_df["hazard_model_id"] == run_name) &
                     (agg_stats_df["nloc_001"] == locations_nloc_dict[loc_name])]["values"].values[0]

    plot_ylims = (1e-5,1)
    fig, axes = plt.subplots(1, 2, figsize=(8, 5))

    for realization_index in range(len(hazard_prob_of_exceedance)):
        gmcm_name = realization_names[realization_index].ground_motion_characterization_models_id

        print(gmcm_name)


        axes[0].loglog(nshm_im_levels, hazard_prob_of_exceedance[realization_index],
                       label=plot_label_lookup[gmcm_name],
                       marker="o",
                       color=plot_colors[realization_index],
                       linestyle=plot_linestyles[realization_index])


    # axes[0].legend(loc="lower left",
    #                           #prop={'size': 6},
    #               framealpha=0.6,
    #               handlelength=2.2,
    #               handletextpad=0.2)



    # Customize the second subplot
    axes[1].semilogy(std_ln, mean,marker="o", linestyle="-", color="tab:blue", label="Stafford (2022)")

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
                 "PGA =  ",
                 ha='right',
                 va='center',
                 color="black")

    axes[0].text(5.8e-3,
                 plot_ylims[1]+0.5,
                 r"$\sigma_{\ln P(PGA=pga)} = $",
                 ha='right',
                 va='center',
                 color="black")

    plt.subplots_adjust(bottom=0.1, top=0.81,left=0.085, right=0.99,wspace=0.23)

    plt.savefig("/home/arr65/data/nshm/output_plots/explanation_plot.png", dpi=500)
    print()


if __name__ == "__main__":

    # get_interpolated_gmms()

    #plot_gmm_dispersion_ranges()

    #do_srm_model_plots_with_seperate_location_subplots("PGA")




    ### use autorun15 for these plots
    #make_cov_plots()#

    #print()

    ### use autorun21 for these plots
    # run_list_sorted = get_alphabetical_run_list()
    # for location in ["AKL", "WLG", "CHC"]:
    #     do_gmcm_plots_with_seperate_tectonic_region_type(run_list_sorted, location, "PGA")

    #do_big_gmcm_subplot(21)

    #do_plot_for_poster()

    #plot_srm_and_gmcm_model_dispersions(["WLG", "CHC"])
    plot_srm_and_gmcm_model_dispersions(["WLG","CHC","AKL"])


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
