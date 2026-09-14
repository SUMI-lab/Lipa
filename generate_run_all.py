mdps = [
    "mdps/3d_navigation.pickle",
    "mdps/blackjack.pickle",
    "mdps/frozenlake_12x12.pickle",
    "mdps/frozenlake_4x4.pickle",
    "mdps/frozenlake_8x8.pickle",
    "mdps/inventory_management.pickle",
    "mdps/system_administrator_1.pickle",
    "mdps/system_administrator_2.pickle",
    "mdps/system_administrator_tree.pickle",
    "mdps/tictactoe_vs_random.pickle",
    "mdps/tiger_vs_antelope.pickle",
    "mdps/traffic_intersection.pickle",
    "mdps/xor.pickle",
]

prism_files_properties = [
    ("dts-uai/undiscounted/consensus-4-2/model.prism", '"steps_max": R{"steps"}max=? [ F "finished" ];'),
    ("dts-uai/undiscounted/csma-3-2-some/model.prism", '"some_before": Pmin=? [ F "target_some_before" ];'),
    ("dts-uai/undiscounted/csma-3-4/model.prism", '"time_max": R{"time"}max=? [ F "all_delivered" ];'),
    ("dts-uai/undiscounted/firewire-false-36-200/model.prism", '"time_max": R{"time"}max=? [ F "done" ];'),
    ("dts-uai/undiscounted/firewire-true-3-600/model.prism", '"time_max": R{"time"}max=? [ F "done" ];'),
    ("dts-uai/undiscounted/ij-14-single/model.prism", '"stable": Pmax=? [ F  "two_true" ];'),
    ("dts-uai/discounted/maze-7/model.prism", 'R{"reward"}max=? [Cdiscount=0.99]'),
    ("dts-uai/undiscounted/wlan-4/model.prism", '"cost_max": R{"cost"}max=? [ F "both_twelve" ];'),
]

depths = [1, 2, 3, 4, 5]

time_limit = 300
prism_time_limit = 1200

with open("run_all_lipa.sh", "w") as file:
    for mdp in mdps:
        for depth in depths:
            file.write(f"python run.py {mdp} --max_depth {depth} --time_limit {time_limit}\n")

    for prism_model, prop in prism_files_properties:
        for depth in depths:
            file.write(f"python run.py {prism_model} --prop '{prop}' --max_depth {depth} --time_limit {prism_time_limit} --add_dont_care_action\n")

with open("run_all_omdt.sh", "w") as file:
    for mdp in mdps:
        for depth in depths:
            file.write(f"python run_omdt.py {mdp} --max_depth {depth} --time_limit {time_limit}\n")

with open("run_all_dtpaynt.sh", "w") as file:
    for prism_model, _ in prism_files_properties:
        directory = prism_model.removesuffix("/model.prism")
        for depth in depths:
            file.write(f"python run_dtpaynt.py {directory} --max_depth {depth} --time_limit {prism_time_limit}\n")
