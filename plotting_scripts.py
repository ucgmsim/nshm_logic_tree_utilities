import plotting_functions

#plot_gmm_dispersion_ranges()

#do_srm_model_plots_with_seperate_location_subplots("PGA")


# plotting_functions.make_figures_of_all_individual_realizations(srm_or_gmcm="gmcm",
#     results_directory = "/home/arr65/data/nshm/output/gmcm_models",
#                                             plot_output_directory = "/home/arr65/data/nshm/plots/gmcm_individual_realizations",
#                                             locations = ["AKL", "WLG", "CHC"],
#                                             registry_directory = "/home/arr65/src/gns/modified_gns/nzshm-model/resources",)

plotting_functions.make_figures_of_individual_realizations_for_a_single_logic_tree(srm_or_gmcm="srm",
    logic_tree_index_dir = "/home/arr65/data/nshm/output/gmcm_models/logic_tree_index_0",
                                            plot_output_directory = "/home/arr65/data/nshm/plots/srm_individual_realizations",
                                            locations = ["AKL", "WLG", "CHC"],
                                            registry_directory = "/home/arr65/src/gns/modified_gns/nzshm-model/resources")


print()


# plotting_functions.make_figures_of_all_individual_realizations(srm_or_gmcm="srm",
#     results_directory = "/home/arr65/data/nshm/output/srm_models",
#                                             plot_output_directory = "/home/arr65/data/nshm/plots/srm_individual_realizations",
#                                             locations = ["AKL", "WLG", "CHC"],
#                                             registry_directory = "/home/arr65/src/gns/modified_gns/nzshm-model/resources")




print()


### Used function
plotting_functions.make_figure_of_gmm_dispersion_ranges(results_directory = "/home/arr65/data/nshm/output/gmcm_models",
                         plot_output_directory = "/home/arr65/data/nshm/plots",
                         locations = ["AKL", "WLG", "CHC"],
                         filter_strs = ["CRU", "HIK_and_PUY", "SLAB"],
                         vs30 = 400,
                         im = "PGA",
                         plot_dpi=500,
                        num_interp_mean_points=1000,
                        min_log10_mean_for_interp=-6,
                        max_log10_mean_for_interp=-2,
                         plot_interpolations=True,
                        min_mean_value_for_interp_plots=1e-9)

print()


plotting_functions.make_figure_of_srm_model_components(results_directory = "/home/arr65/data/nshm/output/srm_models",
                                                       plot_output_directory = "/home/arr65/data/nshm/plots",
                                                       locations = ["AKL", "WLG", "CHC"])




plotting_functions.make_figure_of_gmcms(results_directory = "/home/arr65/data/nshm/output/gmcm_models",
                     plot_output_directory = "/home/arr65/data/nshm/plots",
                     locations = ["AKL", "WLG", "CHC"])

plotting_functions.make_figure_showing_Bradley2009_method(results_directory = "/home/arr65/data/nshm/output/gmcm_models/logic_tree_index_3",
                                plot_output_directory = "/home/arr65/data/nshm/plots",
                                registry_directory = "/home/arr65/src/gns/modified_gns/nzshm-model/resources",
                                location_short_name = "WLG",
                                vs30 = 400,
                                im = "PGA")

plotting_functions.make_figure_of_coefficient_of_variation(results_directory="/home/arr65/data/nshm/output/full_component_logic_trees",
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
plotting_functions.make_figure_of_srm_and_gmcm_model_dispersions(
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
