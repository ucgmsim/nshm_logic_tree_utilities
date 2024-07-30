import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path
import pandas as pd

from cycler import cycler
import natsort

# named_color_list = list(matplotlib.colors.cnames.keys())
# named_color_list = named_color_list[::5]

#named_color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

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

# default_cycler = (cycler(color=['r', 'g', 'b', 'y']) +
#                   cycler(linestyle=['-', '--', ':', '-.']))

def load_resultsV2(output_dir: Path, location: str) -> pd.DataFrame:
    full_df = pd.DataFrame()

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

        full_df = pd.concat([full_df, pd.read_parquet(file)], ignore_index=True)

    return full_df


#auto_dir = Path("/home/arr65/data/nshm/auto_output/auto1")
auto_dir = Path("/home/arr65/data/nshm/auto_output/auto2")

location = "WLG"
vs30 = 400
im = "PGA"



plt.rc('axes', prop_cycle=custom_cycler)
#for location in ["AKL","WLG","CHC"]:
for location in [location]:

    dir_list = natsort.natsorted([x for x in auto_dir.iterdir() if x.is_dir()])

    for run_dir in dir_list:

        # Get all the files in the results directory
        df = load_resultsV2(output_dir = run_dir, location = location)

        mean = df[(df["agg"] == "mean") & (df["vs30"] == vs30) & (df["imt"] == im)]["values"].values[0]
        cov = df[(df["agg"] == "cov") & (df["vs30"] == vs30) & (df["imt"] == im)]["values"].values[0]

        sln = np.sqrt(np.log(cov ** 2 + 1))
        # skip the single branch run with no variance
        if np.all(np.isclose(sln,0.0)):
            continue
        # ax = plt.gca()
        # ax.set_prop_cycle(default_cycler)
        plt.semilogy(sln, mean, label=f"{run_dir.name} {im} in {location} (Vs30 = {vs30} m/s)")

plt.legend(prop={'size': 8})
plt.show()

