from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.dataset as ds

import toshi_hazard_post.calculators as calculators
import nzshm_model.branch_registry

nshm_im_levels = np.array([
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

nshm_im_levels = np.loadtxt("resources/nshm_im_levels.txt")


#registry = nzshm_model.branch_registry.Registry()

registry_dir = Path("/home/arr65/src/gns/modified_gns/nzshm-model/resources")
gmm_registry_csv = pd.read_csv(registry_dir / 'gmm_branches.csv')
source_registry_csv = pd.read_csv(registry_dir / 'source_branches.csv')


### registry can be accessed like:
## entry = registry.source_registry.get_by_hash("af9ec2b004d7")

print()

data_dir = Path("/home/arr65/data/nshm/auto_output/auto11/run_0/nloc_0=-41.0~175.0")
realization_dir = Path("/home/arr65/data/nshm/auto_output/auto12/run_0/individual_realizations/nloc_0=-41.0~175.0")

#realization_dir = Path("/home/arr65/data/nshm/auto_output/auto11/run_0/individual_realizations")



# = pd.read_parquet(realization_dir / '9e9650eb-7e50-4290-adc7-fcc9aa2fdb36-part-0.parquet')

print()

#df = ds.dataset(source=realization_dir,format="parquet")

df = ds.dataset(source=realization_dir, format="parquet").to_table().to_pandas()

gmm_id_col_names = [str(x) for x in df.columns if (("component" in str(x)) & ("gmm" in str(x)))]
source_id_col_names = [str(x) for x in df.columns if (("component" in str(x)) & ("source" in str(x)))]



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

hazard = aggregation_calc.calculate_aggs(rate_array, weights, ["mean"])

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