import numpy as np

from src.branch_and_bound import OptimalMDPTree
from src.storm_solver import StormSolver
from src.env_solver import SparseMDPSolver

import pickle

import click

import pandas as pd

import time


def filename_to_environment(filename: str):
    if filename.endswith(".pickle"):
        return filename.split("/")[-1][:-7]

    if filename.endswith("/model.prism"):
        filename = filename.removesuffix("/model.prism")

    return filename.split("/")[-1]


@click.command()
@click.argument(
    "filename"
)
@click.option(
    "--prop",
    default=None,
    help="Property for model checking (not necessary for OMDT instances).",
)
@click.option(
    "--param",
    "model_parameters",
    type=(str, int),
    multiple=True,
    help="Optional model parameters for PRISM files",
)
@click.option(
    "--max_depth", default=3, type=int, help="Depth of the decision tree policy."
)
@click.option("--time_limit", default=None, type=int, help="Time limit in seconds.")
@click.option(
    "--add_dont_care_action",
    is_flag=True,
    help="Whether to add a random don't care action to each state",
)
@click.option(
    "--skip_branch_heuristic",
    is_flag=True,
    help="Skip the branch (state-action) heuristic (not recommended)",
)
@click.option(
    "--skip_node_selection_heuristic",
    is_flag=True,
    help="Skip the node selection heuristic (not recommended)",
)
@click.option(
    "--skip_all_tree_heuristics",
    is_flag=True,
    help="Skip the decision tree learning heuristics (not recommended)",
)
@click.option(
    "--skip_perfect_tree_heuristic",
    is_flag=True,
    help="Skip the perfect policy mimic heuristic (not recommended)",
)
def run(
    filename,
    prop,
    model_parameters,
    max_depth,
    time_limit,
    add_dont_care_action,
    skip_branch_heuristic,
    skip_node_selection_heuristic,
    skip_all_tree_heuristics,
    skip_perfect_tree_heuristic,
):
    model_parameters = dict(model_parameters)
    print(
        filename,
        prop,
        model_parameters,
        max_depth,
        time_limit,
        add_dont_care_action,
        skip_branch_heuristic,
        skip_node_selection_heuristic,
        skip_all_tree_heuristics,
        skip_perfect_tree_heuristic,
    )
    if prop is None:
        # Assume that without property we have a path to a pickled OMDT MDP
        print("Using python sparse MDP solver")

        with open(filename, "rb") as file:
            mdp = pickle.load(file)

        trans_probs = np.transpose(mdp.trans_probs, (0, 2, 1))
        rewards = np.transpose(mdp.rewards, (0, 2, 1))

        mdp_solver = SparseMDPSolver(
            trans_probs,
            rewards,
            mdp.initial_state_p,
            mdp.observations,
            feature_names=mdp.feature_names,
            action_names=mdp.action_names,
        )
    else:
        # Assume that if a property is given we have a DRN or PRISM file
        print("Using STORM model checker")
        mdp_solver = StormSolver(filename, prop, model_parameters, add_dont_care_action)

    print("n_states:", mdp_solver.n_states_)
    print("n_actions:", mdp_solver.n_actions_)
    print("max depth:", max_depth)

    start_time = time.time()

    tree = OptimalMDPTree(
        max_depth,
        time_limit,
        track_progress=True,
        skip_branch_heuristic=skip_branch_heuristic,
        skip_node_selection_heuristic=skip_node_selection_heuristic,
        skip_all_tree_heuristics=skip_all_tree_heuristics,
        skip_perfect_tree_heuristic=skip_perfect_tree_heuristic,
    )
    tree.solve(mdp_solver)

    runtime = time.time() - start_time

    print(tree.to_string())
    print(
        {
            "filename": filename,
            "property": prop,
            "max_depth": max_depth,
            "time_limit": time_limit,
            "add_dont_care_action": add_dont_care_action,
            "n_states": mdp_solver.n_states_,
            "n_actions": mdp_solver.n_actions_,
            "score": tree.score_,
            "bound": tree.bound_,
            "optimal": tree.optimal_,
            "runtime": runtime,
        }
    )

    history_df = pd.DataFrame(tree.history_, columns=["time", "objective", "bound"])
    environment = filename_to_environment(filename)
    history_filename = (
        "out/lipa_"
        + environment
        + "_"
        + str(max_depth)
        + "_"
        + ("-" if skip_branch_heuristic else "+")
        + ("-" if skip_node_selection_heuristic else "+")
        + ("-" if skip_all_tree_heuristics else "+")
        + ("-" if skip_perfect_tree_heuristic else "+")
        + "_history.csv"
    )
    print("output file:", history_filename)
    history_df.to_csv(history_filename, index=False)


if __name__ == "__main__":
    run()
