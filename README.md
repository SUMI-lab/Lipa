# Lipa

**Lipa** computes optimal depth-limited **decision tree policies for Markov Decision Processes**. Given an MDP
and a maximum depth, it returns a decision tree over the state features that maps states to actions, together
with a proof of optimality (or a bound and an optimality gap if the time limit runs out first).

Lipa searches over *state-action constraints* rather than over tree structures. It solves a relaxation of the
MDP for an upper bound, fits decision trees to the relaxed optimal policy for feasible incumbents, and branches
on states where the tree and the relaxed policy disagree. Both explicit MDPs and PRISM models (via
[Storm](https://www.stormchecker.org/)) are supported.

This repository also contains two methods from the literature, used as baselines:

- **OMDT** — a MILP encoding of the problem, solved with Gurobi (source vendored in `omdt/`).
- **dtPaynt** — decision tree policy synthesis in the PAYNT tool (invoked through the `paynt` CLI).

## Installation

Python 3.10. The three methods have conflicting native dependencies, so pick the requirements file matching
what you want to run:

```bash
pip install -r requirements.txt
```

Generate the OMDT MDPs if you want to run on those with

```bash
python generate_mdps.py
```

## Usage

Run Lipa on an explicit MDP:

```bash
python run.py mdps/frozenlake_4x4.pickle --max_depth 3 --time_limit 300
```

Run Lipa on a PRISM model, passing the property to verify:

```bash
python run.py dts-uai/undiscounted/wlan-4/model.prism \
  --prop '"cost_max": R{"cost"}max=? [ F "both_twelve" ];' \
  --max_depth 3 --time_limit 1200 --add_dont_care_action
```

`--add_dont_care_action` gives every state an extra action that behaves like a uniformly random choice among
the real actions, letting the tree express "any action is fine here" and falling back to it if the intended
action is not available (this behavior is based on dtPaynt).

The baselines take the same MDPs:

```bash
python run_omdt.py mdps/frozenlake_4x4.pickle --max_depth 3 --time_limit 300
python run_dtpaynt.py dts-uai/undiscounted/wlan-4 --max_depth 3 --time_limit 1200
```

Each run appends a row to `out/results_{lipa,omdt,dtpaynt}.csv` (score, bound, optimality, runtime) and Lipa
writes the tree itself to `out/lipa_<environment>_<depth>.txt`.

## Reproducing the experiments

`generate_run_all.py` defines the benchmark grid — the MDPs, the PRISM models with their properties, the
depths, and the time limits — and writes one shell script per method:

```bash
python generate_run_all.py
./run_all_lipa.sh
./run_all_omdt.sh
./run_all_dtpaynt.sh
```

`run_anytime.py` is a variant of `run.py` that records the incumbent and bound over time and exposes
`--skip_*` flags to disable individual heuristics, for anytime and ablation experiments.

## Layout

```
src/                  Lipa: branch and bound + the two MDP solver backends
omdt/                 OMDT baseline (MILP) and the benchmark environments
mdps/                 Pickled benchmark MDPs (from mdps.zip or generate_mdps.py)
dts-uai/              PRISM benchmark models and properties
out/                  Results, trees and logs
```
