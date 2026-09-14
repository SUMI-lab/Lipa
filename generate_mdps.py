import importlib
from pathlib import Path

names = [
    "3d_navigation",
    "blackjack",
    "frozenlake_12x12",
    "frozenlake_4x4",
    "frozenlake_8x8",
    "inventory_management",
    "system_administrator_1",
    "system_administrator_2",
    "system_administrator_tree",
    "tictactoe_vs_random",
    "tiger_vs_antelope",
    "traffic_intersection",
    "xor",
]

output_dir = Path("mdps")
output_dir.mkdir(exist_ok=True)

for name in names:
    module_path = f"omdt.environments.{name}"
    
    try:
        module = importlib.import_module(module_path)
        mdp = module.generate_mdp()
        
        output_path = output_dir / f"{name}.pickle"
        mdp.export(output_path)
        
        print(f"Exported {name} to {output_path}")
    
    except Exception as e:
        print(f"Failed for {name}: {e}")
