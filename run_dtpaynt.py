import click

import subprocess

import re

import time

import os

import pandas as pd

import psutil

@click.command()
@click.argument('model_dir')
@click.option('--output_filename', default=None, type=str)
@click.option('--max_depth', default=3, type=int, help='Depth of the decision tree policy.')
@click.option('--time_limit', default=None, type=int, help='Time limit in seconds.')
@click.option('--max_memory', default=8, type=float, help='Physical memory limit in GB')
def run(model_dir, output_filename, max_depth, time_limit, max_memory):
    print(model_dir, max_depth, time_limit)

    model_dir = model_dir.rstrip("/")

    if output_filename is None:
        output_filename = "out/dtpaynt_" + model_dir.replace("/", "_") + "_" + str(max_depth)

    command = ["paynt", "--sketch", f"{model_dir}/model.prism", "--props", f"{model_dir}/model.props", "--tree-depth", str(max_depth), "--timeout", str(time_limit), "--export-synthesis", str(output_filename), "--add-dont-care-action", "."]
    log_filename = output_filename + ".log"

    MAX_MEM = max_memory * 1024**3
    kill_time_limit = 1.5 * time_limit + 30
    with open(log_filename, "w") as file:
        start_time = time.time()

        proc = subprocess.Popen(
            command,
            stdout=file,
            stderr=subprocess.STDOUT,
            text=True
        )

        p = psutil.Process(proc.pid)

        status = "done"
        try:
            while proc.poll() is None:
                # Check time limit
                if time.time() - start_time > int(kill_time_limit):
                    proc.kill()
                    print("Process timed out, output saved so far.")
                    file.write("HARD TIME LIMIT EXCEEDED\n")
                    status = "time_limit"
                    break

                # Check memory usage
                try:
                    mem = p.memory_info().rss
                    if mem > MAX_MEM:
                        proc.kill()
                        print(f"Process killed: memory limit exceeded ({mem / 1e9:.2f} GB)")
                        file.write("MEMORY LIMIT EXCEEDED\n")
                        status = "memory_limit"
                        break
                except psutil.NoSuchProcess:
                    break  # process already exited

                time.sleep(1.0)

            proc.wait()

        except Exception as e:
            proc.kill()
            raise e

    runtime = time.time() - start_time

    with open(log_filename, "r") as file:
        log = file.read()
    
    tree_score_regex = r"the synthesized tree has value (\d+.?\d*)"
    matches = re.findall(tree_score_regex, log)
    if len(matches) != 1:
        print("Could not parse dtpaynt log accurately")
    score = float(matches[-1])

    random_score_regex = r"the random scheduler has value: (\d+.?\d*)"
    optimal_score_regex = r"the optimal scheduler has value: (\d+.?\d*)"
    random_score = float(re.findall(random_score_regex, log)[-1])
    optimal_score = float(re.findall(optimal_score_regex, log)[-1])
    optimal = "explored: 100 %" in log

    results = {
        "filename": model_dir,
        "max_depth": max_depth,
        "time_limit": time_limit,
        "score": score,
        "random_score": random_score,
        "optimal_score": optimal_score,
        "runtime": runtime,
        "optimal": optimal,
        "status": status,
    }
    new_row = pd.DataFrame([results])

    results_filename = "out/results_dtpaynt.csv"
    if os.path.isfile(results_filename):
        df = pd.read_csv(results_filename)
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = new_row
    df.to_csv(results_filename, index=False)
    

if __name__ == "__main__":
    run()
