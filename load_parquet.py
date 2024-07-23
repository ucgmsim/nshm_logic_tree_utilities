from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

data_dir = Path('/home/arr65/data/nshm/nshm_output/nloc_0=-44.0~173.0')

#df = pd.read_parquet(data_dir / '2dbaf8f2-9f19-4d33-b852-366dda120b54-part-0.parquet')
df = pd.read_parquet(data_dir / 'b75b4f5b-7a18-404d-a29e-9b661c1fd893-part-0.parquet')

print()

d = df["values"]


mean_loc = df["agg"] == "mean"

print()


m = df[df["agg"] == "mean"]["values"][0]

#w = pd.read_csv("/home/arr65/src/logic_tree_study/hazard_curves_from_website/hazard-curves.csv",skiprows=1)
w = pd.read_csv("/home/arr65/data/nshm/csv_from_website/chch_vs30_400_hazard-curves.csv",skiprows=1)



w2 = w[(w["period"] == "PGA") & (w["statistic"]=="mean")]



w3 = w2.drop(columns=["lat", "lon", "vs30", "period","statistic"])

w4 = w3.values.flatten()

boolmask = w4 > 0

w5 = w4[boolmask]
m2 = m[boolmask]

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