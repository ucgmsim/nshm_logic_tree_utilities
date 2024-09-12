"""
This script provides examples of how to use the plotting functions in plotting_functions.py.
"""

import plotting_functions

plotting_functions.make_figure_of_coefficient_of_variation(
    results_directory="/home/arr65/data/nshm/output/",
    plot_output_directory="/home/arr65/data/nshm/plots",
)


input_locations = (("WLG", "CHC", "AKL"), ("WLG", "CHC"))
for locations in input_locations:
    plotting_functions.make_figure_of_srm_and_gmcm_model_dispersions(
        locations=locations,
        results_directory="/home/arr65/data/nshm/output",
        plot_output_directory="/home/arr65/data/nshm/plots",
    )


plotting_functions.make_figure_of_srm_model_components(
    results_directory="/home/arr65/data/nshm/output/",
    plot_output_directory="/home/arr65/data/nshm/plots/",
    locations=("WLG", "CHC", "AKL"),
)


plotting_functions.make_figure_of_gmcms(
    results_directory="/home/arr65/data/nshm/output",
    plot_output_directory="/home/arr65/data/nshm/plots",
    locations=("AKL", "WLG", "CHC"),
)


plotting_functions.make_figure_of_gmm_dispersion_ranges(
    results_directory="/home/arr65/data/nshm/output",
    plot_output_directory="/home/arr65/data/nshm/plots",
    locations=("AKL", "WLG", "CHC"),
    filter_strs=("CRU", "HIK_and_PUY", "SLAB"),
    vs30=400,
    im="PGA",
    plot_dpi=500,
    num_interp_mean_points=1000,
    min_log10_mean_for_interp=-6,
    max_log10_mean_for_interp=-2,
    plot_interpolations=False,
    min_mean_value_for_interp_plots=1e-9,
)


plotting_functions.make_figure_showing_bradley2009_method(
    results_directory="/home/arr65/data/nshm/output/logic_tree_index_4",
    plot_output_directory="/home/arr65/data/nshm/plots",
    registry_directory="/home/arr65/src/gns/modified_gns/nzshm-model/resources",
    location_short_name="WLG",
    vs30=400,
    im="PGA",
)


plotting_functions.make_figures_of_several_individual_realizations(
    results_directory="/home/arr65/data/nshm/output",
    plot_output_directory="/home/arr65/data/nshm/plots/individual_realizations",
    locations=("AKL", "WLG", "CHC"),
    im="PGA",
    vs30=400,
    im_xlims=(9e-5, 5),
    poe_min_plot=1e-5,
    plot_dpi=500,
    notes_to_exclude=(
        ("full > ", "full > "),
        ("full > 1 (nth) h.w.b. > ", "full > "),
        ("full > ", "full > 1 (nth) h.w.b. > "),
        (
            "full > tectonic_region_type_set:[SLAB] > slab_only_branch > ",
            "full > 1 (nth) h.w.b. > tectonic_region_type_set:[SLAB] > ",
        ),
    ),
)
