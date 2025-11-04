"""
nohup python sweep.py > sweep_parameter/sweep.log 2>&1 &
"""

from tqdm import tqdm
from calculate_distance_difference import *
from concurrent.futures import ProcessPoolExecutor
import json
import os
import numpy as np
# Save results to JSON
os.makedirs("sweep_parameter", exist_ok=True)
output_path = os.path.join("sweep_parameter", "results.json")



# w_EI_range = np.arange(0.12, 0.15, 0.01) # 0.143
# w_EP_range = np.arange(0.009, 0.014, 0.001) # 0.012
# w_PE_range = np.arange(0.70, 0.75, 0.01) # 0.71

w_EI_range = np.arange(0.01, 0.4, 0.01) # 0.143
w_EP_range = np.arange(0.0, 0.01, 0.001) # 0.012
w_PE_range = np.arange(0.60, 1.00, 0.01) # 0.71

total_simulations = len(w_EI_range) * len(w_EP_range) * len(w_PE_range)


def run_simulation(w_EI, w_EP, w_PE):
    conf = {
    "w_EE": 0.72, # 0.72
    "w_EI": w_EI, # 0.143
    "w_IE": 0.74, # 0.74
    "w_II": 0.01, # 0.01
    "w_PP": 0.01, # 0.01
    "w_EP": w_EP, # 0.012
    "w_PE": w_PE, # .709
    "sigma": 0.0001,
    "shifter_strength": 0.018,
    }
    speeds, distance = compute_phase_speed(conf)
    return {"config": conf, "speeds": speeds, "distance": distance}

# Set up the progress bar and the process pool.
def _to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    return obj


results = []
with tqdm(total=total_simulations) as pbar:
    with ProcessPoolExecutor(max_workers=20) as executor:
        futures = []
        for w_EI in w_EI_range:
            for w_EP in w_EP_range:
                for w_PE in w_PE_range:
                    futures.append(executor.submit(run_simulation, w_EI, w_EP, w_PE))

        for future in futures:
            res = future.result()
            print(f'speeds: {res["speeds"]}, distance: {res["distance"]}')
            results.append(res)
            pbar.update(1)

# Save results to JSON
os.makedirs("sweep_parameter", exist_ok=True)
output_path = os.path.join("sweep_parameter", "results.json")
with open(output_path, "w") as f:
    json.dump(_to_serializable(results), f, indent=2)

