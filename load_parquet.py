from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import toshi_hazard_post.calculators as calculators

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

data_dir = Path("/home/arr65/data/nshm/auto_output/auto11/run_0/nloc_0=-41.0~175.0")
realization_dir = Path("/home/arr65/data/nshm/auto_output/auto11/run_0/individual_realizations/nloc_0=-41.0~175.0")


df = pd.read_parquet(realization_dir / '9e9650eb-7e50-4290-adc7-fcc9aa2fdb36-part-0.parquet')

rate_array = np.zeros((len(df), len(NSHM_IM_LEVELS)))

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

plt.loglog(NSHM_IM_LEVELS, weighted_avg_prob_from_rate, label='weighted_avg_prob')
plt.loglog(NSHM_IM_LEVELS, agg_mean, label='agg_mean')
plt.ylabel("Annual exceedance rate")
plt.xlabel("PGA")
plt.legend()
plt.show()

residual = np.log(weighted_avg_prob_from_rate) - np.log(agg_mean)

plt.figure()
plt.semilogx(NSHM_IM_LEVELS, residual, '.')
plt.ylabel("log10(weighted_avg_prob) - log10(agg_mean)")
plt.xlabel("PGA")
plt.show()

print()

###################
## Trying the actual functions from aggregation_calc.py

from toshi_hazard_post import aggregation_calc

hazard = aggregation_calc.calculate_aggs(rate_array, weights, ["mean"])

probs = calculators.rate_to_prob(hazard, 1.0)

plt.loglog(NSHM_IM_LEVELS, probs[0], label='hazard_mean')
plt.loglog(NSHM_IM_LEVELS, agg_mean, label='agg_mean')
plt.show()

residual2 = np.log(probs[0]) - np.log(agg_mean)

plt.figure()
plt.semilogx(NSHM_IM_LEVELS, residual2, '.')
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
# plt.loglog(NSHM_IM_LEVELS[boolmask],m2,label='parquet_file')
# plt.loglog(NSHM_IM_LEVELS[boolmask],w5,'--',label='website')
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
# plt.semilogx(NSHM_IM_LEVELS[boolmask][0:-4], log_ratio[0:-4],'.')
# plt.show()
#
# print()