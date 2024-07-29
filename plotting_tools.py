from pathlib import Path
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import natsort
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as ticker


nshm_im_levels = np.loadtxt('resources/nshm_im_levels.txt')

# def load_results(results_dir: Path):
#
#     for index, file in enumerate(results_dir.glob('*.parquet')):
#
#         if index == 0:
#             full_df = pd.read_parquet(file)
#
#         full_df = pd.concat([full_df, pd.read_parquet(file)], ignore_index=True)
#     print()
#     return full_df


def load_resultsV2(output_dir: Path, location: str) -> pd.DataFrame:

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

        if index == 0:
            full_df = pd.read_parquet(file)

        full_df = pd.concat([full_df, pd.read_parquet(file)], ignore_index=True)

    return full_df

# def get_data_from_df(full_df, imt, agg):
#
#
#     df = full_df[full_df["imt"] == imt]
#
#     data = df[df["agg"] == agg]["values"].values[0]
#
#     return data


locations = ["AKL","WLG","CHC"]
vs30s = [225, 400, 750]
ims = ["PGA", "SA(0.1)","SA(0.5)", "SA(1.0)", "SA(3.0)", "SA(10.0)"]

plot_output_dir = "/home/arr65/data/nshm/output_plots"
plot_file_name = "mean_vs_dispersion.pdf"

plot_style_dict = {"color":{}, "linestyle":{}}

plot_style_dict["color"]["CHC"] = "blue"
plot_style_dict["color"]["WLG"] = "green"
plot_style_dict["color"]["AKL"] = "red"

plot_style_dict["linestyle"][225] = ":"
plot_style_dict["linestyle"][400] = "-"
plot_style_dict["linestyle"][750] = "--"

pdf_mean_vs_dispersion = PdfPages(Path(plot_output_dir) / plot_file_name)

for im in ims:

    for location in locations:

        df = load_resultsV2(output_dir = Path("/home/arr65/data/nshm/nshm_output"), location = location)

        for vs30 in vs30s:

            if location == "CHC":
                plot_color = "blue"
            if location == "WLG":
                plot_color = "green"
            if location == "AKL":
                plot_color = "red"

            if vs30 == 225:
                plot_style = ":"
            if vs30 == 400:
                plot_style = "-"
            if vs30 == 750:
                plot_style = "--"

            mean = df[(df["agg"] == "mean") & (df["vs30"] == vs30) & (df["imt"] == im)]["values"].values[0]
            cov = df[(df["agg"] == "cov") & (df["vs30"] == vs30) & (df["imt"] == im)]["values"].values[0]

            sln = np.sqrt(np.log(cov**2 + 1))

            plt.semilogy(sln, mean,label=f"{im} in {location} (Vs30 = {vs30} m/s)",color=plot_color,linestyle=plot_style)


    plt.legend(fontsize='small')
    plt.xlabel(r'Dispersion in hazard probability, $\sigma_{\ln P(IM=im)}$')
    plt.ylabel(r'Mean annual hazard probability, $\mu_{P(IM=im)}$')

    plt.ylim(1e-6, 1e0)
    
    pdf_mean_vs_dispersion.savefig()
    
    plt.close()
    
pdf_mean_vs_dispersion.close()


print()



# plt.savefig('/home/arr65/data/nshm/output_plots/mean_vs_dispersion.png',dpi=400)

plt.show()

print()

#data = df[df["agg"] == agg]["values"].values[0]



print()


# pdf3 = PdfPages(Path(f"/home/arr65/data/nshm/output_plots") / f'{city_code}_TRT.pdf')
#
# for im in ims:
#
#     for agg in aggs:
#
#         full_full_data = get_data_from_df(load_results(Path(f'/home/arr65/data/nshm/nshm_output/{city_code}_275_full_full/')), im, agg)
#
#         nocrust_data = get_data_from_df(load_results(Path(f'/home/arr65/data/nshm/nshm_output/275_HIK_PUY_SLAB/{nloc}')), im, agg)
#
#         crust_data = get_data_from_df(load_results(results_dir = Path(f'/home/arr65/data/nshm/nshm_output/275_CRU/{nloc}')), im, agg)
#
#         summed_c = nocrust_data + crust_data
#
#         plt.loglog(nshm_im_levels, full_full_data, linestyle='-', color="black", label='full logic tree')
#         plt.loglog(nshm_im_levels, summed_c, linestyle='--', color="blue", label='no_crust + crust')
#         plt.loglog(nshm_im_levels, nocrust_data, linestyle='-.', color="orange", label='no_crust')
#         plt.loglog(nshm_im_levels, crust_data, linestyle=':', color="green", label='crust')
#         plt.legend()
#         plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
#         plt.ylim(1e-6,1e0)
#         plt.ylabel('residual')
#         plt.xlabel('IM level')
#         plt.title(f"{city_code} {im} (agg = {agg})")
#
#         pdf3.savefig()
#         plt.close()
# pdf3.close()

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
        plt.loglog(nshm_im_levels, full_full_data, linestyle='-', color="black", label='full logic tree')
        plt.loglog(nshm_im_levels, full_highest_data, linestyle=':', label='full SRM, single GMCM')
        plt.loglog(nshm_im_levels, highest_full_data, linestyle='--', label='single SRM, full GMCM')
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
        ax = plt.gca()
        ax.minorticks_on()
        ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.semilogx(nshm_im_levels, fh_resid, linestyle=":", label=r'$\ln$(full SRM, single GMCM) $-$ $\ln$(full logic tree)')
        plt.semilogx(nshm_im_levels, hf_resid, linestyle="--", label=r'$\ln$(single SRM, full GMCM) $-$ $\ln$(full logic tree)')
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

    plt.loglog(nshm_im_levels, full_full_data_mean, linestyle='-', color="blue", linewidth = 2, label='full logic tree')
    # plt.loglog(nshm_im_levels, full_full_data_mean + full_full_data_std, color="black", linestyle=':')
    # plt.loglog(nshm_im_levels, full_full_data_mean - full_full_data_std, color="black", linestyle=':')
    plt.fill_between(nshm_im_levels, full_full_data_mean - full_full_data_std, full_full_data_mean + full_full_data_std, color='blue', alpha=0.2)

    plt.loglog(nshm_im_levels, full_highest_data_mean, linestyle='--', color="green", linewidth = 2, label='full SRM, single GMCM')
    # plt.loglog(nshm_im_levels, full_highest_data_mean + full_highest_data_std, color="orange", linestyle=':')
    # plt.loglog(nshm_im_levels, full_highest_data_mean - full_highest_data_std, color="orange", linestyle=':')
    plt.fill_between(nshm_im_levels, full_highest_data_mean - full_highest_data_std, full_highest_data_mean + full_highest_data_std,
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
    plt.loglog(nshm_im_levels, full_full_data_mean, color="blue", linewidth = 2, linestyle='-', label='full logic tree')
    #plt.loglog(nshm_im_levels, full_full_data_mean + full_full_data_std, color="black", linestyle=':')
    #plt.loglog(nshm_im_levels, full_full_data_mean - full_full_data_std, color="black", linestyle=':')
    plt.fill_between(nshm_im_levels, full_full_data_mean - full_full_data_std, full_full_data_mean + full_full_data_std, color='blue', alpha=0.2)

    plt.loglog(nshm_im_levels, highest_full_data_mean, linestyle='--', color="red", linewidth=2, label='single SRM, full GMCM')
    # plt.loglog(nshm_im_levels, highest_full_data_mean + highest_full_data_std, color="blue", linestyle=':')
    # plt.loglog(nshm_im_levels, highest_full_data_mean - highest_full_data_std, color="blue", linestyle=':')
    plt.fill_between(nshm_im_levels, highest_full_data_mean - highest_full_data_std, highest_full_data_mean + highest_full_data_std,
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
        plt.loglog(nshm_im_levels, full_full_data, linestyle='-', color = "blue", label='full logic tree')
        plt.loglog(nshm_im_levels, full_highest_data, linestyle=':', color="green", label='full SRM, single GMCM')
        plt.loglog(nshm_im_levels, highest_full_data, linestyle='--', color="red", label='single SRM, full GMCM')
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
        ax = plt.gca()
        ax.minorticks_on()
        ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
        plt.legend()


        plt.subplot(2,1,2)
        plt.semilogx(nshm_im_levels, fh_resid, color="green", label=r'$\ln$(full SRM, single GMCM) $-$ $\ln$(full logic tree)')
        plt.semilogx(nshm_im_levels, hf_resid, color="red", label=r'$\ln$(full SRM, single GMCM) $-$ $\ln$(full logic tree)')
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


# df = pd.read_parquet(file)
# print()
# d = df["values"]
# print(d)

#df = pd.read_parquet(data_dir / 'b75b4f5b-7a18-404d-a29e-9b661c1fd893-part-0.parquet')