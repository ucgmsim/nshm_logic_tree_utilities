from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.dataset as ds
import natsort
import copy
from matplotlib.backends.backend_pdf import PdfPages

from toshi_hazard_post import aggregation_calc
import toshi_hazard_post.calculators as calculators

def remove_special_characters(s):
    chars_to_remove = ["'", "[", "]", '"']

    translation_table = str.maketrans('', '', ''.join(chars_to_remove))
    return s.translate(translation_table)

group_id_to_color = {"AbrahamsonEtAl2014":"blue",
                     "Atkinson2022Crust":"orange",
                        "BooreEtAl2014":"green",
                        "Bradley2013":"red",
                        "CampbellBozorgnia2014":"purple",
                        "ChiouYoungs2014":"brown",
                        "Stafford2022":"pink"}


def plot_residuals(group_id, stat_from_realizations, stat_from_toshi_hazard_post, stat_name):

    residual = np.log(stat_from_realizations) - np.log(stat_from_toshi_hazard_post)

    label_font_size = 10

    plt.close("all")

    plt.subplot(2, 1, 1)
    plt.title(f"{group_id} {stat_name}")
    plt.loglog(nshm_im_levels, stat_from_toshi_hazard_post,label="toshi hazard post")
    plt.loglog(nshm_im_levels, stat_from_realizations, "r--",label="realizations")
    plt.legend()

    plt.ylabel('APoE',fontsize=label_font_size)
    plt.xlabel('acceleration (g)',fontsize=label_font_size)

    plt.subplot(2, 1, 2)
    plt.title(f'{group_id} residual in {stat_name}')
    plt.semilogx(nshm_im_levels, residual, '.-')
    plt.xlabel('acceleration (g)',fontsize=label_font_size)
    plt.ylabel('ln(real.) - ln(thp)',fontsize=label_font_size)
    plt.subplots_adjust(left=0.15,hspace=0.6)


import nzshm_model.branch_registry

plot_all_residuals = True
plot_poe_comparisons = True

if plot_all_residuals:
    residual_pdf = PdfPages(Path("/home/arr65/data/nshm/output_plots") / "residuals_realizations_thp.pdf")

nshm_im_levels = np.loadtxt("resources/nshm_im_levels.txt")


#registry = nzshm_model.branch_registry.Registry()

registry_dir = Path("/home/arr65/src/gns/modified_gns/nzshm-model/resources")
#registry_dir = Path("/home/arr65/src/gns/nzshm-model/resources")
gmm_registry_df = pd.read_csv(registry_dir / 'gmm_branches.csv')
source_registry_df = pd.read_csv(registry_dir / 'source_branches.csv')


### registry can be accessed like:
## entry = registry.source_registry.get_by_hash("af9ec2b004d7")


# data_dir = Path("/home/arr65/data/nshm/auto_output/auto10/run_0/nloc_0=-41.0~175.0")
# realization_dir = Path("/home/arr65/data/nshm/auto_output/auto10/run_0/individual_realizations/nloc_0=-41.0~175.0")

#base_dir = Path("/home/arr65/data/nshm/auto_output/auto10")

base_dir = Path("/home/arr65/data/nshm/auto_output/auto11")

#realization_dir = Path("/home/arr65/data/nshm/auto_output/auto10/run_0/individual_realizations/nloc_0=-41.0~175.0")


run_dirs = [x for x in base_dir.iterdir() if x.is_dir()]
run_dirs = natsort.natsorted(run_dirs)

statistical_aggregation_df = pd.DataFrame()
individual_realization_df = pd.DataFrame()

for run_dir in run_dirs:

    temp_df = ds.dataset(source=run_dir / "individual_realizations", format="parquet").to_table().to_pandas()

    individual_realization_df = pd.concat([individual_realization_df, temp_df], ignore_index=True)

    location_dirs = [x for x in run_dir.iterdir() if (x.is_dir() & ("nloc" in str(x)))]

    for location_dir in location_dirs:

        temp_df = ds.dataset(source=location_dir, format="parquet").to_table().to_pandas()

        statistical_aggregation_df = pd.concat([statistical_aggregation_df, temp_df],ignore_index=True)

source_ids = []
gmm_ids = []

print()


### Get all realization ids
for idx, row in individual_realization_df[individual_realization_df["nloc_001"] == "-41.300~174.780"].iterrows():

    contributing_branches_hash_ids = row["contributing_branches_hash_ids"]
    contributing_branches_hash_ids_clean = remove_special_characters(contributing_branches_hash_ids).split(", ")

    for contributing_branches_hash_id in contributing_branches_hash_ids_clean:

        source_id = contributing_branches_hash_id[0:12]
        gmm_id = contributing_branches_hash_id[12:24]

        gmm_reg_idx = gmm_registry_df["hash_digest"] == gmm_id
        gmm_id = gmm_registry_df[gmm_reg_idx]["identity"].values[0]

        source_reg_idx = source_registry_df["hash_digest"] == source_id
        source_id = source_registry_df[source_reg_idx]["extra"].values[0]

        source_ids.append(source_id)
        gmm_ids.append(gmm_id)

source_ids = np.array(source_ids)
assert np.all(source_ids == source_ids[0]), "All source ids should be the same"
print()

gmm_id_groups = []

for gmm_id in gmm_ids:

    gmm_group_id = gmm_id.split("(")[0]
    if gmm_group_id not in gmm_id_groups:
        gmm_id_groups.append(gmm_group_id)

gmm_id_groups = natsort.natsorted(gmm_id_groups)

id_to_upper_central_lower_dict = {"Bradley2013":{"upper":"sigma_mu_epsilon=1.28155","central":"sigma_mu_epsilon=0.0","lower":"sigma_mu_epsilon=-1.28155"},
                                  "Stafford2022":{"upper":"mu_branch=Upper","central":"mu_branch=Central","lower":"mu_branch=Lower"},
                                 "BooreEtAl2014":{"upper":"sigma_mu_epsilon=1.28155","central":"sigma_mu_epsilon=0.0","lower":"sigma_mu_epsilon=-1.28155"},
                                 "Atkinson2022Crust":{"upper":"epistemic=Upper, modified_sigma=true","central":"epistemic=Central, modified_sigma=true","lower":"epistemic=Lower, modified_sigma=true"},
                                 "AbrahamsonEtAl2014":{"upper":"sigma_mu_epsilon=1.28155","central":"sigma_mu_epsilon=0.0","lower":"sigma_mu_epsilon=-1.28155"},
                                 "CampbellBozorgnia2014":{"upper":"sigma_mu_epsilon=1.28155","central":"sigma_mu_epsilon=0.0","lower":"sigma_mu_epsilon=-1.28155"},
                                 "ChiouYoungs2014":{"upper":"sigma_mu_epsilon=1.28155","central":"sigma_mu_epsilon=0.0","lower":"sigma_mu_epsilon=-1.28155"}}


realization_arm_index_to_name = {0:"upper", 1:"central", 2:"lower"}

id_to_rate_array_dict = copy.deepcopy(id_to_upper_central_lower_dict)
id_to_weight_dict = copy.deepcopy(id_to_upper_central_lower_dict)
id_to_rate_agg_stats_dict = copy.deepcopy(id_to_upper_central_lower_dict)
id_to_prob_agg_stats_dict = {}
id_to_prob_array_dict = {}

group_id_to_run_dict = {}

for key in id_to_rate_array_dict.keys():
    id_to_rate_array_dict[key] = np.zeros((3, len(nshm_im_levels)))

for key in id_to_weight_dict.keys():
    id_to_weight_dict[key] = np.zeros(3)

location_code_str = "-41.300~174.780"

for idx, row in individual_realization_df[individual_realization_df["nloc_001"] == location_code_str].iterrows():

    hazard_rate = row["branches_hazard_rates"]

    contributing_branches_hash_ids = row["contributing_branches_hash_ids"]
    contributing_branches_hash_ids_clean = remove_special_characters(contributing_branches_hash_ids).split(", ")

    realization_id = ""

    # get the ids of all branches that contributed to this realization
    assert len(contributing_branches_hash_ids_clean) == 1, "realizations from multiple branches are not yet fully supported"
    for contributing_branches_hash_id in contributing_branches_hash_ids_clean:

        source_id = contributing_branches_hash_id[0:12]
        gmm_id = contributing_branches_hash_id[12:24]

        gmm_reg_idx = gmm_registry_df["hash_digest"] == gmm_id
        gmm_id = gmm_registry_df[gmm_reg_idx]["identity"].values[0]

        source_reg_idx = source_registry_df["hash_digest"] == source_id
        source_id = source_registry_df[source_reg_idx]["extra"].values[0]

        #realization_id += f"{source_id}_{gmm_id}_"
        realization_id += f"{gmm_id}_"

    gmm_group_id = realization_id.split("(")[0]
    id_within_group = realization_id.split("(")[1].strip("_)(")

    if gmm_group_id not in group_id_to_run_dict.keys():
        group_id_to_run_dict[gmm_group_id] = row["hazard_model_id"]

    upper_key = id_to_upper_central_lower_dict[gmm_group_id]["upper"]
    central_key = id_to_upper_central_lower_dict[gmm_group_id]["central"]
    lower_key = id_to_upper_central_lower_dict[gmm_group_id]["lower"]

    if id_within_group == upper_key:
        id_to_rate_array_dict[gmm_group_id][0] = hazard_rate
        id_to_weight_dict[gmm_group_id][0] = row["branch_weight"]

    elif id_within_group == central_key:
        id_to_rate_array_dict[gmm_group_id][1] = hazard_rate
        id_to_weight_dict[gmm_group_id][1] = row["branch_weight"]

    elif id_within_group == lower_key:
        id_to_rate_array_dict[gmm_group_id][2] = hazard_rate
        id_to_weight_dict[gmm_group_id][2] = row["branch_weight"]

for gmm_group_id in gmm_id_groups:

    rate_array = id_to_rate_array_dict[gmm_group_id]

    id_to_rate_agg_stats_dict[gmm_group_id] = aggregation_calc.calculate_aggs(rate_array, id_to_weight_dict[gmm_group_id], ["mean", "std", "cov"])
    id_to_prob_agg_stats_dict[gmm_group_id] = calculators.rate_to_prob(id_to_rate_agg_stats_dict[gmm_group_id], 1.0)

    id_to_prob_array_dict[gmm_group_id] = calculators.rate_to_prob(id_to_rate_array_dict[gmm_group_id], 1.0)

    if plot_all_residuals:

        run_name = group_id_to_run_dict[gmm_group_id]

        agg_stat_toshi_hazard_post = statistical_aggregation_df[(statistical_aggregation_df["nloc_001"] == location_code_str)
        & (statistical_aggregation_df["hazard_model_id"] == run_name)]

        ### The aggregate statistics provided directly from the GNS package toshi-hazard-post
        thp_agg_stats_mean = agg_stat_toshi_hazard_post[agg_stat_toshi_hazard_post["agg"] == "mean"]["values"].values[0]
        thp_agg_stats_std = agg_stat_toshi_hazard_post[agg_stat_toshi_hazard_post["agg"] == "std"]["values"].values[0]
        thp_agg_stats_cov = agg_stat_toshi_hazard_post[agg_stat_toshi_hazard_post["agg"] == "cov"]["values"].values[0]

        print()

        plot_residuals(gmm_group_id, id_to_prob_agg_stats_dict[gmm_group_id][0], thp_agg_stats_mean, "mean")
        residual_pdf.savefig()
        plot_residuals(gmm_group_id, id_to_prob_agg_stats_dict[gmm_group_id][1], thp_agg_stats_std, "std")
        residual_pdf.savefig()
        plot_residuals(gmm_group_id, id_to_prob_agg_stats_dict[gmm_group_id][2], thp_agg_stats_mean, "cov")
        residual_pdf.savefig()

if plot_all_residuals:
    residual_pdf.close()

if plot_poe_comparisons:

    comparison_pdf = PdfPages(Path("/home/arr65/data/nshm/output_plots") / "realization_comparison_plots.pdf")

    comparison_base = "Bradley2013"
    #xlims = [1e-2, 5e0]
    xlims = [1e-3, 5e0]
    ylims = [1e-6, 1e0]

    # Overplot central realizations
    for realization_arm_index in range(3):

        plt.close("all")

        for gmm_group_id in gmm_id_groups:

            residuals = np.log10(id_to_prob_array_dict[gmm_group_id]) - np.log10(id_to_prob_array_dict[comparison_base])

            idxs = np.where((nshm_im_levels >= xlims[0]) & (nshm_im_levels <= xlims[1]))[0]

            plt.semilogx(nshm_im_levels[idxs], residuals[realization_arm_index,idxs], '.-', label=gmm_group_id)

            plt.xlabel("PGA (g)")
            #plt.ylabel(f"log({gmm_group_id}) - log({comparison_base})")
            plt.ylabel(r"$\ln$(APoE$_1$)-$\ln$(APoE$_2$)")
            plt.legend()
            plt.title(f"{realization_arm_index_to_name[realization_arm_index]} residuals relative to {comparison_base}")
            #plt.ylim([1e-6, 1e0])
            #plt.subplots_adjust(hspace=0)
        plt.grid()
        comparison_pdf.savefig()

    for gmm_group_id in gmm_id_groups:

        residuals = np.log10(id_to_prob_array_dict[gmm_group_id]) - np.log10(id_to_prob_array_dict[comparison_base])

        plt.close("all")

        ax1 = plt.subplot(2,1,1)
        ax1.set_xticklabels([])
        plt.loglog(nshm_im_levels, id_to_prob_array_dict[gmm_group_id][1],linewidth=3, label=gmm_group_id)
        plt.fill_between(nshm_im_levels, id_to_prob_array_dict[gmm_group_id][0], id_to_prob_array_dict[gmm_group_id][2], alpha=0.3)

        plt.loglog(nshm_im_levels, id_to_prob_array_dict[comparison_base][0], linestyle=':', color="black",linewidth=3)
        plt.loglog(nshm_im_levels, id_to_prob_array_dict[comparison_base][1],linestyle='--', color="black",linewidth=3,label=comparison_base)
        plt.loglog(nshm_im_levels, id_to_prob_array_dict[comparison_base][2], linestyle=':', color="black",linewidth=3)

        plt.legend(handlelength=4)
        #plt.ylabel("Annual probability of exceedance")
        plt.ylabel("APoE")
        plt.xlabel("PGA (g)")
        plt.title("Range indicates upper and lower realizations")

        ax1.set_xticklabels([])


        ax2 = plt.subplot(2, 1, 2)

        idxs = np.where((nshm_im_levels >= xlims[0]) & (nshm_im_levels <= xlims[1]))[0]

        plt.semilogx(nshm_im_levels[idxs], residuals[0,idxs], '.-', label="upper residuals")
        plt.semilogx(nshm_im_levels[idxs], residuals[1,idxs], '.-', label="central residuals")
        plt.semilogx(nshm_im_levels[idxs], residuals[2,idxs], '.-', label="lower residuals")

        plt.xlabel("PGA (g)")
        #plt.ylabel(f"log({gmm_group_id}) - log({comparison_base})")
        plt.ylabel(r"$\ln$(APoE$_1$)-$\ln$(APoE$_2$)")
        plt.legend()
        plt.xlim([1e-2, 5e0])
        #plt.ylim([1e-6, 1e0])
        plt.subplots_adjust(hspace=0)
        comparison_pdf.savefig()

    comparison_pdf.close()











print()




#     print()
#
#
#     #plt.fill_between(nshm_im_levels, rate_array[0], rate_array[2], alpha=0.3, label=f"{gmm_group_id}")
#
#     plt.loglog(nshm_im_levels, rate_array[1], color=group_id_to_color[gmm_group_id], linewidth = 1, linestyle="--", label=f"{gmm_group_id}")
#     plt.loglog(nshm_im_levels, rate_array[0], color=group_id_to_color[gmm_group_id], linewidth = 1, linestyle=":")
#     plt.loglog(nshm_im_levels, rate_array[2], color=group_id_to_color[gmm_group_id], linewidth=1, linestyle=":")
#
# plt.xlim([1e-2, 5e0])
# plt.ylim([1e-6, 1e0])
#
# plt.legend(prop={'size': 3})
# plt.savefig('test.pdf')
# plt.show()
#
# print()





















        #plt.loglog(nshm_im_levels, prob_of_exceedance, label=gmm_id)




# plt.xlim([1e-2, 5e0])
# plt.ylim([1e-6, 1e0])
#
# plt.legend(prop={'size': 3})
# plt.savefig('test.pdf')
# plt.show()

print()














#realization_dir = Path("/home/arr65/data/nshm/auto_output/auto11/run_0/individual_realizations")



# = pd.read_parquet(realization_dir / '9e9650eb-7e50-4290-adc7-fcc9aa2fdb36-part-0.parquet')

print()

#df = ds.dataset(source=realization_dir,format="parquet")

#df = ds.dataset(source=base_dir, format="parquet").to_table().to_pandas()

gmm_id_col_names = [str(x) for x in df.columns if (("component" in str(x)) & ("gmm" in str(x)))]
source_id_col_names = [str(x) for x in df.columns if (("component" in str(x)) & ("source" in str(x)))]

print()



desired_gmm_name = "Bradley2013"

needed_gmm_hash_ids = gmm_registry_csv[gmm_registry_csv["identity"].str.contains(desired_gmm_name)]["hash_digest"].values

# mask = df[id_col_names] == needed_gmm_hash_ids[0]
# for needed_gmm_id in needed_gmm_hash_ids[1:]:
#     mask |= df[id_col_names] == needed_gmm_id

mask_list = []

mask_2d = np.zeros((len(df), 4), dtype=bool)



for col_idx, id_col_name in enumerate(gmm_id_col_names):

    mask_2d[:,col_idx] = df[id_col_name].str.contains(needed_gmm_hash_ids[0]).values

mask = np.any(mask_2d, axis=1)
masked_df = df[mask]

highest_weighted_row_masked_idx = masked_df["branch_weight"].argmax()

highest_weighted_branch_in_selection = masked_df.loc[highest_weighted_row_masked_idx]




print()











hwbidx = np.argmax(df["branch_weight"])

#"gsim_name": "Bradley2013",


needed_gmm_hash_ids = []




# for idx, row in gmm_registry_csv.iterrows():
#
#     if desired_gmm_name in row["identity"]:
#         needed_gmm_hash_ids.append(row["hash_digest"])

print()

composite_list = df["branches_as_hash_composites"]

num_components = []
component_hash_id_lens = []


for branch_list in composite_list:

    for component_hash_id in branch_list:
        component_hash_id_lens.append(len(component_hash_id))



    #num_components.append(len(i))

#num_components = np.array(num_components)
component_hash_id_lens = np.array(component_hash_id_lens)


#
# filtered_realizations_df = pd.DataFrame()

# for idx, row in df.iterrows():
#
#     if idx % 1000 == 0:
#         print(f"Processing realization {idx+1} of {len(df)} ({100*(idx+1)/len(df):.2f}%)")
#
#     for component_hash_id in row["branches_as_hash_composites"]:
#
#         assert len(component_hash_id) == 24, "should be 24 characters long"
#
#         if component_hash_id[12:24] in needed_gmm_hash_ids:
#             pd.concat([filtered_realizations_df, row])
#
# print()
#
#








rate_array = np.zeros((len(df), len(nshm_im_levels)))

print()

for i in range(len(df)):
    if i % 1000 == 0.0:
        print(i)
    rate_array[i] = df['branches_hazard_rates'][i]

weights = df["branch_weight"].values

weighted_avg_rate = np.average(rate_array, weights=weights, axis=0)

weighted_avg_prob_from_rate = calculators.rate_to_prob(weighted_avg_rate, 1.0)


df_agg = pd.read_parquet(data_dir / '16dd2556-b1b9-4bb4-a173-0a82e004655e-part-0.parquet')

agg_mean = df_agg[df_agg["agg"] == "mean"]["values"][0]

print()

plt.loglog(nshm_im_levels, weighted_avg_prob_from_rate, label='weighted_avg_prob')
plt.loglog(nshm_im_levels, agg_mean, label='agg_mean')
plt.ylabel("Annual exceedance rate")
plt.xlabel("PGA")
plt.legend()
plt.show()

residual = np.log(weighted_avg_prob_from_rate) - np.log(agg_mean)

plt.figure()
plt.semilogx(nshm_im_levels, residual, '.')
plt.ylabel("log10(weighted_avg_prob) - log10(agg_mean)")
plt.xlabel("PGA")
plt.show()

print()

###################
## Trying the actual functions from aggregation_calc.py

from toshi_hazard_post import aggregation_calc



probs = calculators.rate_to_prob(hazard, 1.0)

plt.loglog(nshm_im_levels, probs[0], label='hazard_mean')
plt.loglog(nshm_im_levels, agg_mean, label='agg_mean')
plt.show()

residual2 = np.log(probs[0]) - np.log(agg_mean)

plt.figure()
plt.semilogx(nshm_im_levels, residual2, '.')
plt.ylabel("log10(weighted_avg_prob) - log10(agg_mean)")
plt.xlabel("PGA")
plt.show()





# #df = pd.read_parquet(data_dir / '2dbaf8f2-9f19-4d33-b852-366dda120b54-part-0.parquet')
# df = pd.read_parquet(data_dir / 'b75b4f5b-7a18-404d-a29e-9b661c1fd893-part-0.parquet')
#
# print()
#
# d = df["values"]
#
#
# mean_loc = df["agg"] == "mean"
#
# print()
#
#
# m = df[df["agg"] == "mean"]["values"][0]
#
# #w = pd.read_csv("/home/arr65/src/logic_tree_study/hazard_curves_from_website/hazard-curves.csv",skiprows=1)
# w = pd.read_csv("/home/arr65/data/nshm/csv_from_website/chch_vs30_400_hazard-curves.csv",skiprows=1)
#
#
#
# w2 = w[(w["period"] == "PGA") & (w["statistic"]=="mean")]
#
#
#
# w3 = w2.drop(columns=["lat", "lon", "vs30", "period","statistic"])
#
# w4 = w3.values.flatten()
#
# boolmask = w4 > 0
#
# w5 = w4[boolmask]
# m2 = m[boolmask]

# log_ratio = np.log10(m2)/np.log10(w5)
#
# plt.loglog(nshm_im_levels[boolmask],m2,label='parquet_file')
# plt.loglog(nshm_im_levels[boolmask],w5,'--',label='website')
#
# plt.ylabel("Annual exceedance rate (agg = mean)")
# plt.xlabel("PGA")
# plt.title('CHCH Vs30=400 m/s')
# plt.legend()
#
#
# plt.savefig("/home/arr65/src/logic_tree_study/test_output/test2.png",dpi=400)
# plt.close()
#
#
# plt.semilogx(nshm_im_levels[boolmask][0:-4], log_ratio[0:-4],'.')
# plt.show()
#
# print()