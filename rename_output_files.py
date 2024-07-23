from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


output_dir = Path('/home/arr65/data/nshm/nshm_output/nloc_0=-44.0~173.0')

files_to_rename = output_dir.glob("*.parquet")

for file_name in files_to_rename:
    df = pd.read_parquet(file_name)

    d = df["values"]

    ## location, vs30, im








#df = pd.read_parquet(data_dir / '2dbaf8f2-9f19-4d33-b852-366dda120b54-part-0.parquet')


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