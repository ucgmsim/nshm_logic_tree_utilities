import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path
import pandas as pd
import scipy

from cycler import cycler
import natsort
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt, ticker as mticker

nshm_im_levels = np.loadtxt("resources/nshm_im_levels.txt")


# named_color_list = list(matplotlib.colors.cnames.keys())
# named_color_list = named_color_list[::5]

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

def load_all_runs_in_rungroup(output_dir: Path) -> pd.DataFrame:

    results_df = pd.DataFrame()

    for run_dir in output_dir.iterdir():
        if run_dir.is_dir():
            results_df = (pd.concat
                          ([results_df,
                            load_locations_from_run(run_dir, ["AKL","WLG","CHC"])],
                           ignore_index=True))

    return results_df



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

def insert_ln_std(df):

    # Initialize an empty DataFrame to hold the new rows
    new_rows = []

    # Iterate over the DataFrame
    for index, row in df.iterrows():
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
#auto_dir = Path("/home/arr65/data/nshm/auto_output/auto13") ## For making the main GMCM dispersion plots
#auto_dir = Path("/home/arr65/data/nshm/auto_output/auto14")
auto_dir = Path("/home/arr65/data/nshm/auto_output/auto15")

#auto_dir = Path("/home/arr65/data/nshm/auto_output/auto17")

#auto_dir = Path("/home/arr65/data/nshm/auto_output/auto18")

df = load_all_runs_in_rungroup(auto_dir)

plot_output_dir = Path("/home/arr65/data/nshm/output_plots")

#location = "WLG"
vs30 = 400
#im = "PGA"

run_notes_df = pd.read_csv(auto_dir / "run_notes.csv")

#pdf_all_ims = PdfPages(plot_output_dir / f"{auto_dir.name}.pdf")
pdf_all_ims = PdfPages(plot_output_dir / f"{auto_dir.name}_mean_vs_dispersion.pdf")
#pdf_all_ims = PdfPages(plot_output_dir / f"{auto_dir.name}_mean_vs_dispersion_seperate_trt.pdf")

#ims = ["PGA", "SA(0.1)", "SA(0.5)", "SA(1.0)", "SA(3.0)", "SA(10.0)"]
ims = ["PGA"]


locations = ["AKL","WLG","CHC"]
#locations = ["WLG"]

locations_nloc_dict = {"AKL":"-36.870~174.770",
                       "WLG":"-41.300~174.780",
                       "CHC":"-43.530~172.630"}

run_list = natsort.natsorted((df["hazard_model_id"].unique()))

run_list_label_tuple_list = []

##################################################################################################
##################################################################################################

### For sorting the runs by the model name and run number for tectonic type plots

## Get the model name and run_number pairs
# for run_index, run_name in enumerate(run_list):
#
#     for location in ['WLG']:
#
#         run_counter = int(run_name.split("_")[-1])
#
#         slt_note = f"{run_notes_df[run_notes_df["run_counter"] == run_counter]["slt_note"].values[0]}"
#         glt_note = f"{run_notes_df[run_notes_df["run_counter"]==run_counter]["glt_note"].values[0]}"
#
#         trts_from_note = slt_note.split(">")[-2].strip().split(":")[-1].strip("[]")
#         glt_model_from_note = glt_note.split(">")[-2].strip(" []")
#         glt_model = glt_model_from_note.split("*")[0]
#         glt_model_weight = 1/float(glt_model_from_note.split("*")[1])
#
#         plot_label_short = f"{trts_from_note} {glt_model} (w = {glt_model_weight:.3f})"
#
#         run_list_label_tuple_list.append((run_name, plot_label_short))
#
#         print()
#
# sorted_run_list_label_tuple_list = natsort.natsorted(run_list_label_tuple_list, key=lambda x: x[1])
#
# print()
#
# run_list = [x[0] for x in sorted_run_list_label_tuple_list]

##################################################################################################
##################################################################################################

#run_list = ["run_0"]
#plt.rc('axes', prop_cycle=custom_cycler)

def do_plots(over_plot_all=False):

    for im in ims:
        num_hazard_curves_on_plot = 0

        if not over_plot_all:
            plt.close("all")

        plt.rc('axes', prop_cycle=custom_cycler_slt_nth_branch)
        for location in locations:

            nloc_001_str = locations_nloc_dict[location]

            for run in run_list:

                run_counter = int(run.split("_")[-1])

                mean = df[(df["agg"] == "mean") &
                          (df["vs30"] == vs30) &
                          (df["imt"] == im) &
                          (df["hazard_model_id"] == run) &
                          (df["nloc_001"] == nloc_001_str)]["values"].values[0]

                std_ln = df[(df["agg"] == "std_ln") &
                          (df["vs30"] == vs30) &
                          (df["imt"] == im) &
                          (df["hazard_model_id"] == run) &
                          (df["nloc_001"] == nloc_001_str)]["values"].values[0]

                plot_label = (f"{location} {run_notes_df[run_notes_df["run_counter"]==run_counter]["slt_note"].values[0]}, "
                              f"{run_notes_df[run_notes_df["run_counter"]==run_counter]['glt_note'].values[0]}")
                plt.semilogy(std_ln, mean, label=plot_label)
                print(f"plotting: {im} {location} {run}")
                num_hazard_curves_on_plot += 1

        plt.ylim(1e-5,1e0)
        plt.xlim(-0.01, 0.7)


        plt.xlabel(r'Dispersion in hazard probability, $\sigma_{\ln P(IM=im)}$')
        plt.ylabel(r'Mean annual hazard probability, $\mu_{P(IM=im)}$')
        plt.grid(which='major', linestyle='--', linewidth='0.5', color='black')




        if not over_plot_all:

            plt.title(f'Fixed: IM={im}, Vs30 = 400 m/s')
            plt.legend(loc="lower left", prop={'size': 6})
            pdf_all_ims.savefig()


    if over_plot_all:
        plt.title(f'All IMs, fixed Vs30 = 400 m/s')
        pdf_all_ims.savefig()

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

                mean = df[(df["agg"] == "mean") &
                          (df["vs30"] == vs30) &
                          (df["imt"] == im) &
                          (df["hazard_model_id"] == run) &
                          (df["nloc_001"] == nloc_001_str)]["values"].values[0]

                std_ln = df[(df["agg"] == "std_ln") &
                          (df["vs30"] == vs30) &
                          (df["imt"] == im) &
                          (df["hazard_model_id"] == run) &
                          (df["nloc_001"] == nloc_001_str)]["values"].values[0]

                slt_note = f"{run_notes_df[run_notes_df["run_counter"] == run_counter]["slt_note"].values[0]}"
                glt_note = f"{run_notes_df[run_notes_df["run_counter"]==run_counter]["glt_note"].values[0]}"

                trts_from_note = slt_note.split(">")[-2].strip().split(":")[-1].strip("[]")
                glt_model_from_note = glt_note.split(">")[-2].strip(" []")
                glt_model = glt_model_from_note.split("*")[0]
                glt_model_weight = 1/float(glt_model_from_note.split("*")[1])


                #plot_label_short = plot_label.split(">")[-2].strip().split(":")[-1].strip("[]")

                #plot_label_short = f"{trts_from_note} {glt_model_from_note}"

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



def do_plots_with_seperate_tectonic_region_type_per_location(location, im):

    plt.close("all")

    fig, axes = plt.subplots(1, 3)
    plt.subplots_adjust(wspace=0.0)

    have_plotted_im_labels = np.zeros(3,dtype=bool)

    mean_list = []
    std_ln_list = []
    non_zero_run_list = []

    for run in run_list:

        nloc_001_str = locations_nloc_dict[location]

        run_counter = int(run.split("_")[-1])

        mean = df[(df["agg"] == "mean") &
                  (df["vs30"] == vs30) &
                  (df["imt"] == im) &
                  (df["hazard_model_id"] == run) &
                  (df["nloc_001"] == nloc_001_str)]["values"].values[0]

        mean_max = np.max(mean)
        print(f'run {run} max mean: {mean_max}')

        std_ln = df[(df["agg"] == "std_ln") &
                  (df["vs30"] == vs30) &
                  (df["imt"] == im) &
                  (df["hazard_model_id"] == run) &
                  (df["nloc_001"] == nloc_001_str)]["values"].values[0]

        mean_list.append(mean)
        std_ln_list.append(std_ln)
        non_zero_run_list.append(run)

        slt_note = f"{run_notes_df[run_notes_df["run_counter"] == run_counter]["slt_note"].values[0]}"
        glt_note = f"{run_notes_df[run_notes_df["run_counter"]==run_counter]["glt_note"].values[0]}"

        trts_from_note = slt_note.split(">")[-2].strip().split(":")[-1].strip("[]")
        glt_model_from_note = glt_note.split(">")[-2].strip(" []")
        glt_model = glt_model_from_note.split("*")[0]
        glt_model_weight = 1/float(glt_model_from_note.split("*")[1])

        if "CRU" in trts_from_note:
            subplot_idx = 0
        if "INTER" in trts_from_note:
            subplot_idx = 1
        if "SLAB" in trts_from_note:
            subplot_idx = 2

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

        axes[subplot_idx].semilogy(std_ln, mean, label=plot_label,
                                    linestyle=linestyle)

        axes[subplot_idx].set_ylim(1e-5,0.6)
        #axes[subplot_idx].set_ylim(bottom=1e-5)
        axes[subplot_idx].set_xlim(-0.01, 0.7)

        axes[0].set_title("Active Shallow Crust",fontsize=11)
        axes[1].set_title("Subduction Interface",fontsize=11)
        axes[2].set_title("Subduction Intraslab",fontsize=11)

        if subplot_idx == 0:
            axes[subplot_idx].set_ylabel(r'Mean annual hazard probability, $\mu_{P(IM=im)}$')

        if subplot_idx == 1:
            axes[subplot_idx].set_xlabel(r'Dispersion in hazard probability, $\sigma_{\ln P(IM=im)}$')
            axes[subplot_idx].set_yticklabels([])
        if subplot_idx == 2:
            axes[subplot_idx].set_yticklabels([])

        axes[subplot_idx].grid(which='major', linestyle='--', linewidth='0.5', color='black')

        #if subplot_idx == 0:
        axes[subplot_idx].legend(loc="lower left", prop={'size': 3},
                                      handlelength=5)

    fig.suptitle(f'{location}, IM={im}, Vs30 = 400 m/s')
    pdf_all_ims.savefig(fig)

    mean_array = np.array(mean_list)
    std_ln_array = np.array(std_ln_list)

    num_mean_points = 1000
    mm = np.logspace(-6, -1, num_mean_points)

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

        if len(mean_vect3) == 0:
            interp_disp_array[run_idx, :] = np.nan

        else:
            plt.figure()

            print(f"run_idx: {run_idx}, run: {run_list[run_idx]}")

            #plt.semilogy(std_ln_array[run_idx], mean_array[run_idx],  '.')
            plt.semilogx(mean_vect, std_ln_vect, '.')

            #plt.semilogx(mean_vect2, std_ln_vect2, 'r.')

            plt.semilogx(mean_vect3, std_ln_vect3, 'r.')

            plt.show()

            mean_to_dispersion_func = scipy.interpolate.interp1d(mean_vect3, std_ln_vect3, kind='cubic')
            interp_disp = mean_to_dispersion_func(mm)
            interp_disp_array[run_idx, :] = interp_disp

            plt.show()
            print()
            # plt.semilogy(std_ln_array[run_idx], mean_array[run_idx],  '.')
            # plt.semilogy(interp_disp, mm, 'r--')
            plt.semilogx(mean_array[run_idx], std_ln_array[run_idx], '.')
            plt.semilogx(mm, interp_disp, 'r--')
            plt.show()

    return interp_disp_array, mm



def make_cov_plots(over_plot_all=False):

    for im in ims:

        locations = ["WLG"]

        for location in locations:

            nloc_001_str = locations_nloc_dict[location]

            mean_list = []
            std_list = []
            cov_list = []

            for run_idx, run in enumerate(run_list):

                print()

                run_counter = int(run.split("_")[-1])

                mean = df[(df["agg"] == "mean") &
                          (df["vs30"] == vs30) &
                          (df["imt"] == im) &
                          (df["hazard_model_id"] == run) &
                          (df["nloc_001"] == nloc_001_str)]["values"].values[0]

                cov = df[(df["agg"] == "cov") &
                          (df["vs30"] == vs30) &
                          (df["imt"] == im) &
                          (df["hazard_model_id"] == run) &
                          (df["nloc_001"] == nloc_001_str)]["values"].values[0]

                # std_ln = df[(df["agg"] == "std_ln") &
                #           (df["vs30"] == vs30) &
                #           (df["imt"] == im) &
                #           (df["hazard_model_id"] == run) &
                #           (df["nloc_001"] == nloc_001_str)]["values"].values[0]

                std = df[(df["agg"] == "std") &
                          (df["vs30"] == vs30) &
                          (df["imt"] == im) &
                          (df["hazard_model_id"] == run) &
                          (df["nloc_001"] == nloc_001_str)]["values"].values[0]

                mean_list.append(mean)
                std_list.append(std)
                cov_list.append(cov)
                #std_list.append(std_ln)

            #source_unc_plus_gmcm_unc = np.sqrt(np.array(std_list)**2 + np.array(cov_list)**2)
            source_std_plus_gmcm_std = np.array(std_list[0]) + np.array(std_list[1])

            cov_sum = np.array(cov_list[0]) + np.array(cov_list[1])
            cov_sum2 = np.sqrt(np.array(cov_list[0])**2 + np.array(cov_list[1])**2)

            mean_sum = np.array(mean_list[0]) + np.array(mean_list[1])
            mean_sum2 = np.sqrt(np.array(mean_list[0])**2 + np.array(mean_list[1])**2)

            std_sum = np.array(std_list[0]) + np.array(std_list[1])
            std_sum2 = np.sqrt(np.array(std_list[0])**2 + np.array(std_list[1])**2)

            plt.rcParams.update({'font.size': 14})

            lw = 5


            plt.semilogx(nshm_im_levels, cov_list[0], linestyle='--', linewidth=lw, label='source model')
            plt.semilogx(nshm_im_levels, cov_list[1], linestyle='-.', linewidth=lw, label='ground motion model')
            plt.semilogx(nshm_im_levels, cov_list[2], linestyle='-', linewidth=lw, label='both')
            plt.legend(handlelength=5)
            #plt.title(f"{location} {im}")
            plt.ylabel("coefficient of variation (CoV) of\nannual probability of exceedance (APoE)")
            plt.xlabel('peak ground acceleration (g)')
            plt.xlim(1e-2,5)
            plt.ylim(0.05,0.8)
            #plt.title("Wellington assuming Vs30 = 400 m/s")

            # plt.semilogx(nshm_im_levels, cov_sum, linestyle='-.', label='sum')
            # plt.semilogx(nshm_im_levels, cov_sum2, linestyle='-.', label='sum2')




            #plt.show()
            plt.tight_layout()
            plt.savefig("/home/arr65/data/nshm/output_plots/cov_plot.png",dpi=400)

            plt.close()

            plt.figure()
            plt.semilogx(nshm_im_levels, std_list[2], linestyle='-', label='SRM & GMCM')
            plt.semilogx(nshm_im_levels, std_list[0], linestyle='--', label='SRM')
            plt.semilogx(nshm_im_levels, std_list[1], linestyle='-.', label='GMCM')
            plt.legend()
            #plt.title(f"{location} {im}")
            #plt.title("Wellington assuming Vs30 = 400 m/s")
            plt.ylabel("standard deviation of annual\nprobability of exceedance (APoE)")
            plt.xlabel('peak ground acceleration (g)')
            plt.tight_layout()
            plt.savefig("/home/arr65/data/nshm/output_plots/std_plot.png",dpi=400)

def do_srm_model_plots_with_seperate_location_subplots(over_plot_all=False):

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

                mean = df[(df["agg"] == "mean") &
                          (df["vs30"] == vs30) &
                          (df["imt"] == im) &
                          (df["hazard_model_id"] == run) &
                          (df["nloc_001"] == nloc_001_str)]["values"].values[0]

                std_ln = df[(df["agg"] == "std_ln") &
                          (df["vs30"] == vs30) &
                          (df["imt"] == im) &
                          (df["hazard_model_id"] == run) &
                          (df["nloc_001"] == nloc_001_str)]["values"].values[0]

                slt_note = f"{run_notes_df[run_notes_df["run_counter"] == run_counter]["slt_note"].values[0]}"
                glt_note = f"{run_notes_df[run_notes_df["run_counter"]==run_counter]["glt_note"].values[0]}"

                note = slt_note + "," + glt_note



                # trts_from_note = slt_note.split(">")[-2].strip().split(":")[-1].strip("[]")
                # glt_model_from_note = glt_note.split(">")[-2].strip(" []")
                # glt_model = glt_model_from_note.split("*")[0]
                # glt_model_weight = 1/float(glt_model_from_note.split("*")[1])


                #plot_label_short = plot_label.split(">")[-2].strip().split(":")[-1].strip("[]")

                #plot_label_short = f"{trts_from_note} {glt_model_from_note}"

                # plot_label_short = f"{trts_from_note} {glt_model} (w = {glt_model_weight:.3f})"

                # if plot_label_short != "INTER HIK":
                #     continue


                if not over_plot_all:
                    axes[location_idx].semilogy(std_ln, mean, label=note),


                axes[location_idx].set_ylim(1e-12,0.6)
                axes[location_idx].set_xlim(-0.01, 0.6)
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
                axes[location_idx].semilogy(std_ln, mean, label=note)

        if not over_plot_all:
            fig.suptitle(f'Fixed: IM={im}, Vs30 = 400 m/s')
            pdf_all_ims.savefig(fig)

    if over_plot_all:
        fig.suptitle(f'All IMs, Vs30 = 400 m/s')
        pdf_all_ims.savefig(fig)


### use autorun15 for these plots
make_cov_plots()#

#print()

### use autorun13 for these plots
#do_plots_with_seperate_tectonic_region_type_per_location("AKL", "PGA")
#interp_disp_array, mm = do_plots_with_seperate_tectonic_region_type_per_location("WLG", "PGA")

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

#do_srm_model_plots_with_seperate_location_subplots()

#pdf_all_ims.close()

