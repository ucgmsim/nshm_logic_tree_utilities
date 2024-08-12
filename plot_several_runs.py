import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path
import pandas as pd

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



#auto_dir = Path("/home/arr65/data/nshm/auto_output/auto1")
#auto_dir = Path("/home/arr65/data/nshm/auto_output/auto2")
#auto_dir = Path("/home/arr65/data/nshm/auto_output/auto3")
#auto_dir = Path("/home/arr65/data/nshm/auto_output/auto4")
#auto_dir = Path("/home/arr65/data/nshm/auto_output/auto5")
#auto_dir = Path("/home/arr65/data/nshm/auto_output/auto6")
#auto_dir = Path("/home/arr65/data/nshm/auto_output/auto7")
#auto_dir = Path("/home/arr65/data/nshm/auto_output/auto8")
auto_dir = Path("/home/arr65/data/nshm/auto_output/auto9")
#auto_dir = Path("/home/arr65/data/nshm/auto_output/auto11")
#auto_dir = Path("/home/arr65/data/nshm/auto_output/auto12")

df = load_all_runs_in_rungroup(auto_dir)

plot_output_dir = Path("/home/arr65/data/nshm/output_plots")

#location = "WLG"
vs30 = 400
#im = "PGA"

run_notes_df = pd.read_csv(auto_dir / "run_notes.csv")

#pdf_all_ims = PdfPages(plot_output_dir / f"{auto_dir.name}_mean_vs_dispersion.pdf")
pdf_all_ims = PdfPages(plot_output_dir / f"{auto_dir.name}_mean_vs_dispersion_seperate_trt.pdf")

#ims = ["PGA", "SA(0.1)", "SA(0.5)", "SA(1.0)", "SA(3.0)", "SA(10.0)"]
ims = ["PGA"]


locations = ["AKL","WLG","CHC"]
#locations = ["WLG"]

locations_nloc_dict = {"AKL":"-36.870~174.770",
                       "WLG":"-41.300~174.780",
                       "CHC":"-43.530~172.630"}

run_list = natsort.natsorted((df["hazard_model_id"].unique()))
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

                plot_label = (f"{location} {run_notes_df[run_notes_df["run_counter"]==run_counter]["slt_note"].values[0]} "
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

    fig, axes = plt.subplots(1, 3)
    plt.subplots_adjust(wspace=0.0)

    have_plotted_im_labels = np.zeros(3,dtype=bool)

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

        # if not have_plotted_im_labels[subplot_idx]:
            # for datapoint_idx in range(len(mean)):
            #     axes[subplot_idx].text(std_ln[datapoint_idx], mean[datapoint_idx], f"{nshm_im_levels[datapoint_idx]:.1e}", fontsize=1)
            #     #axes[subplot_idx].scatter(std_ln[datapoint_idx], mean[datapoint_idx], color='black', s=1)
            #     have_plotted_im_labels[subplot_idx] = False

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

#do_plots_with_seperate_tectonic_region_type_per_location("AKL", "PGA")
do_plots_with_seperate_tectonic_region_type_per_location("WLG", "PGA")
#do_plots_with_seperate_tectonic_region_type_per_location("CHC", "PGA")


#do_plots(over_plot_all=True)
#do_plots(over_plot_all=False)

#do_plots_with_seperate_location_subplots(over_plot_all=True)
#do_plots_with_seperate_location_subplots(over_plot_all=False)

pdf_all_ims.close()

