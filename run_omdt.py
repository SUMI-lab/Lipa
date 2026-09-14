import numpy as np

import pickle

import click

import pandas as pd

import os

from pathlib import Path

import time

from omdt.solver import OmdtSolver

@click.command()
@click.argument('filename')
@click.option('--max_depth', default=3, type=int, help='Depth of the decision tree policy.')
@click.option('--discount_factor', default=0.999, type=float, help='Discount factor (gamma) for discounted MDPs.')
@click.option('--time_limit', default=None, type=int, help='Time limit in seconds.')
def run(filename, max_depth, discount_factor, time_limit):
    print(filename, max_depth, time_limit)
    
    with open(filename, "rb") as file:
        mdp = pickle.load(file)

    mdp_name = filename.split("/")[-1][:-7]
    output_dir = f"out/omdt_{mdp_name}/"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    solver = OmdtSolver(depth=max_depth, gamma=discount_factor, verbose=True, time_limit=time_limit, output_dir=output_dir)
    solver.solve(mdp)

    runtime = time.time() - start_time

    results = {
        "filename": filename,
        "max_depth": max_depth,
        "time_limit": time_limit,
        "score": solver.objective_,
        "bound": solver.bound_,
        "optimal": solver.optimal_,
        "runtime": runtime,
    }
    new_row = pd.DataFrame([results])

    results_filename = "out/results_omdt.csv"
    if os.path.isfile(results_filename):
        df = pd.read_csv(results_filename)
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = new_row
    df.to_csv(results_filename, index=False)

if __name__ == "__main__":
    run()
