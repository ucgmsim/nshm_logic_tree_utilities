"""
Module for helping to run toshi_hazard_post with modified logic trees.

Classes
-------
CustomLogicTreePair
    A dataclass to store a pair of logic trees for use with toshi_hazard_post.

Functions
---------
run_with_modified_logic_trees
    Runs toshi_hazard_post with modified logic trees.
"""

import copy
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import toml
from nzshm_model.logic_tree import GMCMLogicTree, SourceLogicTree

from toshi_hazard_post.aggregation import run_aggregation
from toshi_hazard_post.aggregation_args import AggregationArgs

import logic_tree_tools


@dataclass
class CustomLogicTreePair:
    """
    A dataclass to store a pair of logic trees for use with toshi_hazard_post.

    The pair must consist of one SourceLogicTree for the seismicity rate model (SRM) and one GMCMLogicTree
    for the ground motion characterization model (GMCM).

    slt: SourceLogicTree, optional
        The seismicity rate model (SRM) logic tree to be used in the run.
    glt: GMCMLogicTree, optional
        The ground motion characterization model (GMCM) logic tree to be used in the run
    slt_note: str, optional
        A human-readable note describing changes to the SourceLogicTree.
    glt_note: str, optional
        A human-readable note describing changes to the GMCMLogicTree.
    other_notes: str, optional
        Any other notes that are relevant.
    """

    slt: Optional[SourceLogicTree] = None
    glt: Optional[GMCMLogicTree] = None

    slt_note: Optional[str] = ""
    glt_note: Optional[str] = ""
    other_notes: Optional[str] = ""

    def notes_to_toml(self, path: Path):
        data = {
            "slt_note": self.slt_note,
            "glt_note": self.glt_note,
            "other_notes": self.other_notes,
        }
        with path.open("w") as f:
            toml.dump(data, f)

    def notes_to_pandas_df(self):
        data = {
            "slt_note": self.slt_note,
            "glt_note": self.glt_note,
            "other_notes": self.other_notes,
        }
        return pd.DataFrame(data, index=[0])


def run_with_modified_logic_trees(
    args: AggregationArgs,
    output_dir: Path,
    logic_tree_index: int,
    custom_logic_tree_pair: CustomLogicTreePair,
    locations: list[str],
    output_staging_dir: Path,
):
    """
    Runs toshi_hazard_post with modified logic trees.

    Parameters
    ----------
    args : AggregationArgs
        Contains the arguments for toshi_hazard_post to run.
    output_dir : Path
        The directory where the output files will be saved.
    logic_tree_index : int
        The index of the logic_tree_set in the input list (used for naming the output directory).
    custom_logic_tree_pair : CustomLogicTreePair
        The logic tree set to run toshi_hazard_post with.
    locations : list[str]
        The locations to run toshi_hazard_post for.
    output_staging_dir : Path
        The output directory used internally by toshi_hazard_post. After toshi_hazard_post has finished,
        these files are moved to the output_dir.
    """

    run_start_time = time.time()

    modified_slt = copy.deepcopy(custom_logic_tree_pair.slt)
    modified_glt = copy.deepcopy(custom_logic_tree_pair.glt)

    logic_tree_tools.print_info_about_logic_tree_sets(custom_logic_tree_pair)

    # check the validity of the weights
    logic_tree_tools.check_weight_validity(custom_logic_tree_pair.slt)
    logic_tree_tools.check_weight_validity(custom_logic_tree_pair.glt)

    ### Save a copy of the logic trees for later inspection
    modified_slt.to_json(output_staging_dir / f"srm_logic_tree.json")
    modified_glt.to_json(output_staging_dir / f"gmcm_logic_tree.json")

    ### Save human-readable notes describing the changes to the logic tree
    custom_logic_tree_pair.notes_to_toml(output_staging_dir / f"notes.toml")

    ### While several locations can be passed into the same toshi_hazard_post run,
    ### this requires more memory than is available in a typical desktop workstation.
    ### We therefore loop over the locations and run toshi_hazard_post for each one.

    for location in locations:
        print(f"doing run {logic_tree_index} and location {location}")

        args.locations = [location]
        args.hazard_model_id = f"logic_tree_index_{logic_tree_index}"

        args.srm_logic_tree = modified_slt
        args.gmcm_logic_tree = modified_glt

        run_aggregation(args)

    run_output_dir = output_dir / f"logic_tree_index_{logic_tree_index}"
    run_output_dir.mkdir(parents=True, exist_ok=False)

    ### Move the output files from the staging directory to the run output directory
    for file in output_staging_dir.iterdir():
        shutil.move(file, run_output_dir)

    run_end_time = time.time()
    print(
        f"Time taken for run {logic_tree_index}: {(run_end_time - run_start_time)/60} mins"
    )
