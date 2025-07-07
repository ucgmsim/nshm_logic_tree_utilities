from setuptools import find_packages, setup

setup(
    name="nshm_logic_tree_utilities",  # Your package name
    version="1.0",
    packages=find_packages(),  # Automatically finds all packages
    include_package_data=True,  # Ensures additional files (like data) are included
    url="https://github.com/ucgmsim/nshm_logic_tree_utilities",
    description="Utilities for working with the logic tree used in New Zealand's National Seismic Hazard Model (NSHM) 2022",
)