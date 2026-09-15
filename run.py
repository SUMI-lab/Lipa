import numpy as np

from src.branch_and_bound import OptimalMDPTree
from src.storm_solver import StormSolver
from src.env_solver import SparseMDPSolver

import pickle

import click

import pandas as pd

from pathlib import Path

import os

import time

def filename_to_environment(filename: str):
    if filename.endswith(".pickle"):
        return filename.split("/")[-1][:-7]
    
    if filename.endswith("/model.prism"):
        filename = filename.removesuffix("/model.prism")

    if filename.endswith(".drn"):
        filename = filename.removesuffix(".drn")
    
    return filename.split("/")[-1]

@click.command()
@click.argument('filename')
@click.option('--prop', default=None, help='Property for model checking (not necessary for OMDT instances).')
@click.option("--param", "model_parameters", type=(str, int), multiple=True, help='Optional model parameters for PRISM files')
@click.option('--max_depth', default=3, type=int, help='Depth of the decision tree policy.')
@click.option('--time_limit', default=None, type=int, help='Time limit in seconds.')
@click.option('--add_dont_care_action', is_flag=True, help="Whether to add a random don't care action to each state")
@click.option('--output_dir', default="out", help="Directory to output result files in")
def run(filename, prop, model_parameters, max_depth, time_limit, add_dont_care_action, output_dir):
    model_parameters = dict(model_parameters)
    print(filename, prop, model_parameters, max_depth, time_limit, add_dont_care_action, output_dir)
    if prop is None:
        # Assume that without property we have a path to a pickled OMDT MDP
        print("Using python sparse MDP solver")

        with open(filename, "rb") as file:
            mdp = pickle.load(file)

        trans_probs = np.transpose(mdp.trans_probs, (0, 2, 1))
        rewards = np.transpose(mdp.rewards, (0, 2, 1))
        
        mdp_solver = SparseMDPSolver(trans_probs, rewards, mdp.initial_state_p, mdp.observations, feature_names=mdp.feature_names, action_names=mdp.action_names)
    else:
        # Assume that if a property is given we have a DRN or PRISM file
        print("Using STORM model checker")
        mdp_solver = StormSolver(filename, prop, model_parameters, add_dont_care_action)

    print("n_states:", mdp_solver.n_states_)
    print("n_actions:", mdp_solver.n_actions_)
    print("max depth:", max_depth)
    
    start_time = time.time()

    tree = OptimalMDPTree(max_depth, time_limit)
    tree.solve(mdp_solver)

    runtime = time.time() - start_time

    tree_string = tree.to_string()

    environment = filename_to_environment(filename)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_filename_base = f"{output_dir}/lipa_" + environment + "_" + str(max_depth)
    output_filename = output_filename_base + ".txt"
    print("Writing resulting tree in:", output_filename)
    with open(output_filename, "w") as file:
        file.write(tree_string)

    results = {
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
    new_row = pd.DataFrame([results])

    results_filename = f"{output_dir}/results_lipa.csv"
    if os.path.isfile(results_filename):
        df = pd.read_csv(results_filename)
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = new_row
    df.to_csv(results_filename, index=False)

if __name__ == "__main__":
    run()
