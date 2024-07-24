from pathlib import Path
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import natsort
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as ticker

NSHM_IM_LEVELS = np.array([
    0.0001,
    0.0002,
    0.0004,
    0.0006,
    0.0008,
    0.001,
    0.002,
    0.004,
    0.006,
    0.008,
    0.01,
    0.02,
    0.04,
    0.06,
    0.08,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
    1.2,
    1.4,
    1.6,
    1.8,
    2.0,
    2.2,
    2.4,
    2.6,
    2.8,
    3.0,
    3.5,
    4,
    4.5,
    5.0,
    6.0,
    7.0,
    8.0,
    9.0,
    10.0])

def load_results(results_dir: Path):

    #data_dir = Path('/home/arr65/data/nshm/nshm_output/chc_vs30_275_full_full')

    for index, file in enumerate(results_dir.glob('*.parquet')):

        if index == 0:
            full_df = pd.read_parquet(file)

        full_df = pd.concat([full_df, pd.read_parquet(file)], ignore_index=True)

    return full_df


def overplot_hazard_curve(full_df, label, imt, agg):

    df = full_df[full_df["imt"] == imt]

    #print()

    hazard_curve = df[df["agg"] == agg]["values"].values[0]

    #print()

    plt.loglog(NSHM_IM_LEVELS, hazard_curve,label=label)

    return hazard_curve

    #print()

    #plt.show()

def get_data_from_df(full_df, imt, agg):

    df = full_df[full_df["imt"] == imt]

    data = df[df["agg"] == agg]["values"].values[0]

    return data

location_code = 'wlg'
vs30 = 275

full_full_df = load_results(results_dir = Path(f'/home/arr65/data/nshm/nshm_output/{location_code}_{vs30}_full_full'))
full_highest_df = load_results(results_dir = Path(f'/home/arr65/data/nshm/nshm_output/{location_code}_{vs30}_full_highest'))
highest_full_df = load_results(results_dir = Path(f'/home/arr65/data/nshm/nshm_output/{location_code}_{vs30}_highest_full'))

ims = natsort.natsorted(full_full_df["imt"].unique())
aggs = full_full_df["agg"].unique()


ims = ["PGA", "SA(0.1)","SA(0.5)", "SA(1.0)", "SA(3.0)", "SA(10.0)"]
#ims = ["SA(10.0)"]
#ims = ["PGA"]
aggs = ["mean", "std"]

# ims = ims[0:2]
# aggs = aggs[0:2]

pdf = PdfPages(Path(f"/home/arr65/data/nshm/output_plots") / f'{location_code}_{vs30}.pdf')


progress_counter = 0

for im in ims:

    for agg in aggs:
        progress_counter += 1

        print(f"Plotting {im} (agg = {agg}) ({progress_counter}/{len(ims) * len(aggs)})")

        full_full_data = get_data_from_df(full_full_df, im, agg)
        full_highest_data = get_data_from_df(full_highest_df, im, agg)
        highest_full_data = get_data_from_df(highest_full_df, im, agg)

        fh_resid = np.log(full_highest_data) - np.log(full_full_data)
        hf_resid = np.log(highest_full_data) - np.log(full_full_data)

        plt.close('all')

        plt.subplots(2, 1)
        plt.subplot(2, 1, 1)
        plt.loglog(NSHM_IM_LEVELS, full_full_data, linestyle='-', color="black", label='full logic tree')
        plt.loglog(NSHM_IM_LEVELS, full_highest_data, linestyle=':', label='full SRM, single GMCM')
        plt.loglog(NSHM_IM_LEVELS, highest_full_data, linestyle='--', label='single SRM, full GMCM')
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
        ax = plt.gca()
        ax.minorticks_on()
        ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.semilogx(NSHM_IM_LEVELS, fh_resid, linestyle=":", label=r'$\ln$(full SRM, single GMCM) $-$ $\ln$(full logic tree)')
        plt.semilogx(NSHM_IM_LEVELS, hf_resid, linestyle="--", label=r'$\ln$(single SRM, full GMCM) $-$ $\ln$(full logic tree)')
        plt.ylabel('residual')
        plt.legend()
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
        ax = plt.gca()  # Get current axis
        #ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.minorticks_on()
        ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
        plt.subplots_adjust(hspace=0.03, top=0.94)
        plt.xlabel('IM level')
        plt.ylim(-2,0.5)
        plt.suptitle(f"{im} (agg = {agg})")

        pdf.savefig()
        plt.close()

    pdf.close()

pdf2 = PdfPages(Path(f"/home/arr65/data/nshm/output_plots") / f'{location_code}_{vs30}_mean_plusminus.pdf')

for im in ims:

    progress_counter += 1

    print(f"Plotting {im} (agg = {agg}) ({progress_counter}/{len(ims) * len(aggs)})")

    full_full_data_mean = get_data_from_df(full_full_df, im, "mean")
    full_full_data_std = get_data_from_df(full_full_df, im, "std")

    full_highest_data_mean = get_data_from_df(full_highest_df, im, "mean")
    full_highest_data_std = get_data_from_df(full_highest_df, im, "std")

    highest_full_data_mean = get_data_from_df(highest_full_df, im, "mean")
    highest_full_data_std = get_data_from_df(highest_full_df, im, "std")

    plt.close('all')

    plt.subplots(2, 1)
    plt.subplot(2, 1, 1)

    plt.loglog(NSHM_IM_LEVELS, full_full_data_mean, linestyle='-', color="blue", linewidth = 2, label='full logic tree')
    # plt.loglog(NSHM_IM_LEVELS, full_full_data_mean + full_full_data_std, color="black", linestyle=':')
    # plt.loglog(NSHM_IM_LEVELS, full_full_data_mean - full_full_data_std, color="black", linestyle=':')
    plt.fill_between(NSHM_IM_LEVELS, full_full_data_mean - full_full_data_std, full_full_data_mean + full_full_data_std, color='blue', alpha=0.2)

    plt.loglog(NSHM_IM_LEVELS, full_highest_data_mean, linestyle='--', color="green", linewidth = 2, label='full SRM, single GMCM')
    # plt.loglog(NSHM_IM_LEVELS, full_highest_data_mean + full_highest_data_std, color="orange", linestyle=':')
    # plt.loglog(NSHM_IM_LEVELS, full_highest_data_mean - full_highest_data_std, color="orange", linestyle=':')
    plt.fill_between(NSHM_IM_LEVELS, full_highest_data_mean - full_highest_data_std, full_highest_data_mean + full_highest_data_std,
                     color='green', alpha=0.2)

    plt.ylim(1e-6,1e0)

    ax1 = plt.gca()
    x_limits = ax1.get_xlim()  # Get x-axis limits
    y_limits = ax1.get_ylim()  # Get y-axis limi
    plt.xlabel('IM level')
    ax1.set_xticklabels([])
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray')  # Add grid lines

    plt.legend()

    plt.subplot(2, 1, 2)
    plt.loglog(NSHM_IM_LEVELS, full_full_data_mean, color="blue", linewidth = 2, linestyle='-', label='full logic tree')
    #plt.loglog(NSHM_IM_LEVELS, full_full_data_mean + full_full_data_std, color="black", linestyle=':')
    #plt.loglog(NSHM_IM_LEVELS, full_full_data_mean - full_full_data_std, color="black", linestyle=':')
    plt.fill_between(NSHM_IM_LEVELS, full_full_data_mean - full_full_data_std, full_full_data_mean + full_full_data_std, color='blue', alpha=0.2)

    plt.loglog(NSHM_IM_LEVELS, highest_full_data_mean, linestyle='--', color="red", linewidth=2, label='single SRM, full GMCM')
    # plt.loglog(NSHM_IM_LEVELS, highest_full_data_mean + highest_full_data_std, color="blue", linestyle=':')
    # plt.loglog(NSHM_IM_LEVELS, highest_full_data_mean - highest_full_data_std, color="blue", linestyle=':')
    plt.fill_between(NSHM_IM_LEVELS, highest_full_data_mean - highest_full_data_std, highest_full_data_mean + highest_full_data_std,
                     color='red', alpha=0.2)
    plt.legend()
    plt.xlim(x_limits)  # Set x-axis limits to match the first plot
    plt.ylim(y_limits)
    plt.xlabel('IM level')
    plt.subplots_adjust(hspace=0.07, top=0.94)
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray')  # Add grid lines

    plt.suptitle(f"{im}")

    pdf2.savefig()
    plt.close()

pdf2.close()


progress_counter = 0

for im in ims:

    for agg in aggs:

        progress_counter += 1

        print(f"Plotting {im} (agg = {agg}) ({progress_counter}/{len(ims)*len(aggs)})")

        full_full_data = get_data_from_df(full_full_df, im, agg)
        full_highest_data = get_data_from_df(full_highest_df, im, agg)
        highest_full_data = get_data_from_df(highest_full_df, im, agg)
        print()

        fh_resid = np.log(full_highest_data) - np.log(full_full_data)
        hf_resid = np.log(highest_full_data) - np.log(full_full_data)

        plt.subplots(2,1)
        plt.subplot(2,1,1)
        plt.loglog(NSHM_IM_LEVELS, full_full_data, linestyle='-', color = "blue", label='full logic tree')
        plt.loglog(NSHM_IM_LEVELS, full_highest_data, linestyle=':', color="green", label='full SRM, single GMCM')
        plt.loglog(NSHM_IM_LEVELS, highest_full_data, linestyle='--', color="red", label='single SRM, full GMCM')
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
        ax = plt.gca()
        ax.minorticks_on()
        ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
        plt.legend()


        plt.subplot(2,1,2)
        plt.semilogx(NSHM_IM_LEVELS, fh_resid, color="green", label=r'$\ln$(full SRM, single GMCM) $-$ $\ln$(full logic tree)')
        plt.semilogx(NSHM_IM_LEVELS, hf_resid, color="red", label=r'$\ln$(full SRM, single GMCM) $-$ $\ln$(full logic tree)')
        plt.ylabel('residual')
        plt.xlabel('IM level')
        plt.legend()
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
        ax = plt.gca()  # Get current axis
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.minorticks_on()
        ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
        plt.ylim(-2,0.3)
        plt.suptitle(f"{im} (agg = {agg})")

        plt.subplots_adjust(hspace=0.04,top=0.94)

        pdf.savefig()
        plt.close()

pdf.close()











# #im_to_plot = "PGA"
# im_to_plot = "SA(1.0)"
#
#
# agg_to_plot = "std"

# ff = overplot_hazard_curve(full_full_df, "full_full", im_to_plot, agg_to_plot)
# fh = overplot_hazard_curve(full_highest_df, "full_highest", im_to_plot, agg_to_plot)
# hf = overplot_hazard_curve(highest_full_df, "highest_full", im_to_plot, agg_to_plot)
#
# plt.legend()
# plt.show()

print()

# df = pd.read_parquet(file)
# print()
# d = df["values"]
# print(d)

#df = pd.read_parquet(data_dir / 'b75b4f5b-7a18-404d-a29e-9b661c1fd893-part-0.parquet')