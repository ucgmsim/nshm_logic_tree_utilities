import copy
import shutil
import time
from pathlib import Path

from toshi_hazard_post.aggregation import (
    AggregationArgs,
    run_aggregation,
)

import logic_tree_tools


def run_with_modified_logic_trees(
    args: AggregationArgs,
    output_dir: Path,
    logic_tree_index: int,
    custom_logic_tree_set: logic_tree_tools.CustomLogicTreeSet,
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
    custom_logic_tree_set : logic_tree_tools.CustomLogicTreeSet
        The logic tree set to run toshi_hazard_post with.
    locations : list[str]
        The locations to run toshi_hazard_post for.
    output_staging_dir : Path
        The output directory used internally by toshi_hazard_post. After toshi_hazard_post has finished,
        these files are moved to the output_dir.
    """

    run_start_time = time.time()

    modified_slt = copy.deepcopy(custom_logic_tree_set.slt)
    modified_glt = copy.deepcopy(custom_logic_tree_set.glt)

    logic_tree_tools.print_info_about_logic_tree_sets(custom_logic_tree_set)

    # check the validity of the weights
    logic_tree_tools.check_weight_validity(custom_logic_tree_set.slt)
    logic_tree_tools.check_weight_validity(custom_logic_tree_set.glt)

    ### Save a copy of the logic trees for later inspection
    modified_slt.to_json(output_staging_dir / f"slt_{logic_tree_index}.json")
    modified_glt.to_json(output_staging_dir / f"glt_{logic_tree_index}.json")

    ### Save human-readable notes describing the changes to the logic tree
    custom_logic_tree_set.notes_to_toml(
        output_staging_dir / f"run_{logic_tree_index}_notes.toml"
    )

    ### While several locations can be passed into the same toshi_hazard_post run,
    ### this requires more memory than is available in a typical desktop workstation.
    ### We therefore loop over the locations and run toshi_hazard_post for each one.

    for location in locations:
        print(f"doing run {logic_tree_index} and location {location}")

        args.locations = [location]
        args.hazard_model_id = f"run_{logic_tree_index}"

        args.srm_logic_tree = modified_slt
        args.gmcm_logic_tree = modified_glt

        run_aggregation(args)

    run_output_dir = output_dir / f"run_{logic_tree_index}"
    run_output_dir.mkdir(parents=True, exist_ok=False)

    ### Move the output files from the staging directory to the run output directory
    for file in output_staging_dir.iterdir():
        shutil.move(file, run_output_dir)

    run_end_time = time.time()
    print(
        f"Time taken for run {logic_tree_index}: {(run_end_time - run_start_time)/60} mins"
    )
