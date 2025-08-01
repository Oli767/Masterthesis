print("Hello")

import papermill as pm

# List of parameter sets to try
param_variants = [
    {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5},
    {"a": 10, "b": 20, "c": 30, "d": 40, "e": 50},
    {"a": 100, "b": 200, "c": 300, "d": 400, "e": 500},
    {"a": -1, "b": -2, "c": -3, "d": -4, "e": -5},
    {"a": 0, "b": 1, "c": 0, "d": 1, "e": 0},
]

# Run the notebook with each parameter set
for i, param in enumerate(param_variants, start=1):
    output_name = f"MT_run_{i}.ipynb"
    print(f"Running MT.ipynb with param set {i}: {param}")
    pm.execute_notebook(
        input_path="MT.ipynb", output_path=output_name, parameters={"param": param}
    )
    print(f"✅ Finished run {i} → {output_name}\n")
