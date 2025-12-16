from tqdm import tqdm, trange
from multiprocessing import Pool
import multiprocessing as mp
import random
from brian2 import *
import numpy as np
from tqdm import tqdm, trange
from brian2 import *
import matplotlib.pyplot as plt
from sheet_attractor import simulator
from util import *
from fit import *
from visualization import create_3d_animation
FIG_PATH = 'figures/'
os.makedirs(FIG_PATH, exist_ok=True)



parameters = {
  "w_EE": 0.72,
  "w_EI": 0.143, # 0.143
  "w_IE": 0.76, # 0.74
  "w_II": 0.01, # 0.01
  "w_PP": 0.01,
  "w_EP": 0.013, # 0.012
  "w_PE": 0.71, # .709
  "sigma": 0.001,
}

events = [
        {'type': 'visual_cue_on', 'x': 2, 'y': 6, 'strength': 0.5, 'duration': 300*ms},
        {'type': 'visual_cue_off', 'duration': 300*ms},
        {'type': 'shift', 'direction': 'left', 'strength': 0.018, 'duration': 2000*ms},
        {'type': 'run', 'duration': 100*ms},
        {'type': 'shift', 'direction': 'up', 'strength': 0.018, 'duration': 2000*ms},
    ]


def run_single_simulation(seed):
    """
    Run a single simulation with a given random seed
    
    Args:
        seed (int): Random seed for the simulation
    
    Returns:
        tuple: (speed, bump_positions) from the simulation
    """
    t, fr, fr_penx, fr_peny = simulator(w_EE = parameters['w_EE'], # EB <-> EB
                    w_EI = parameters['w_EI'], # EPG -> R # 0.15
                    w_IE = parameters['w_IE'], # R -> EPG
                    w_II = parameters['w_II'], # R <-> R
                    w_PP = parameters['w_PP'], # PEN <-> PEN
                    w_EP = parameters['w_EP'], # EB -> PEN 
                    w_PE = parameters['w_PE'], # PEN -> EB
                    sigma = parameters['sigma'], # noise level
                    
                    events = events,
                    defaultclock_dt=0.1,
                    seed=seed)

    processed_data = process_data(fr)
    bp, fit_params = bump_position(processed_data)
    trajectory_results = analyze_trajectory(bp, fit_params, processed_data, events, plot=False)
    direction = events[2]['direction'].capitalize() 
    speed = trajectory_results['phase_results'][direction]['speed']
    bump_pos = trajectory_results['bump_positions']
    return speed, bump_pos


def run_multiple_simulations(parameters, events, num_simulations=10, n_processes=None):
    """
    Run multiple simulations with different random seeds and collect results using parallel computation
    
    Args:
        parameters (dict): Parameters for the simulation
        events (list): List of events for the simulation
        num_simulations (int): Number of simulations to run
        n_processes (int): Number of processes to use. If None, uses all available CPUs
    
    Returns:
        tuple: (speeds, bump_positions) lists from all simulations
    """ 
    if n_processes is None:
        n_processes = mp.cpu_count()
    
    print(f"Running {num_simulations} simulations using {n_processes} processes")
    
    # Create list of seeds for each simulation
    seeds = list(range(num_simulations))
    
    # Run simulations in parallel
    with Pool(processes=n_processes) as pool:
        results = list(tqdm(
            pool.imap(run_single_simulation, seeds),
            total=num_simulations,
            desc="Running simulations"
        ))
    
    # Separate speeds and bump_positions
    speeds = [result[0] for result in results]
    bump_positions = [result[1] for result in results]
    
    return speeds, bump_positions

speeds, bump_positions = run_multiple_simulations(parameters, events, num_simulations=100, n_processes=10)

# Calculate statistics for speeds
speeds_array = np.array(speeds)
speed_mean = np.mean(speeds_array)
speed_std = np.std(speeds_array)
speed_median = np.median(speeds_array)
speed_min = np.min(speeds_array)
speed_max = np.max(speeds_array)

# Calculate statistics for bump positions
pos_mean = np.mean(bump_positions, axis=0)
pos_std = np.std(bump_positions, axis=0)
pos_median = np.median(bump_positions, axis=0)
pos_min = np.min(bump_positions, axis=0)
pos_max = np.max(bump_positions, axis=0)

# Save statistics data
statistics_data = {
    'speeds': {
        'raw_data': speeds_array,
        'mean': speed_mean,
        'std': speed_std,
        'median': speed_median,
        'min': speed_min,
        'max': speed_max
    },
    'bump_positions': {
        'raw_data': np.array(bump_positions),
        'mean': pos_mean,
        'std': pos_std,
        'median': pos_median,
        'min': pos_min,
        'max': pos_max
    },
    'parameters': parameters,
    'events': events,
    'num_simulations': len(speeds)
}

# Save to file
import pickle
with open('simulation_statistics_left.pkl', 'wb') as f:
    pickle.dump(statistics_data, f)
print(f"Statistics data saved to 'simulation_statistics.pkl'")

# Plot speed statistics
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Histogram of speeds
ax1.hist(speeds_array * 2 * np.pi / 16, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
ax1.axvline(speed_mean * 2 * np.pi / 16, color='red', linestyle='--', linewidth=2, label=f'Mean: {speed_mean*2*np.pi/16:.2f}')
ax1.axvline(speed_median * 2 * np.pi / 16, color='green', linestyle='--', linewidth=2, label=f'Median: {speed_median*2*np.pi/16:.2f}')
ax1.set_xlabel('Speed [rad/s]')
ax1.set_ylabel('Frequency')
ax1.set_title('Distribution of Speeds')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Box plot of speeds
ax2.boxplot(speeds_array * 2 * np.pi / 16, patch_artist=True, 
            boxprops=dict(facecolor='lightblue', alpha=0.7),
            medianprops=dict(color='red', linewidth=2))
ax2.set_ylabel('Speed [rad/s]')
ax2.set_title('Speed Distribution Box Plot')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIG_PATH, 'speed_statistics_left.png'), dpi=300, bbox_inches='tight')
plt.show()

print("Speed Statistics:")
print(f"Mean: {speed_mean*2*np.pi/16:.2f} [rad/s]")
print(f"Standard Deviation: {speed_std*2*np.pi/16:.2f} [rad/s]")
print(f"Median: {speed_median*2*np.pi/16:.2f} [rad/s]")
print(f"Min: {speed_min*2*np.pi/16:.2f} [rad/s]")
print(f"Max: {speed_max*2*np.pi/16:.2f} [rad/s]")
print()

import matplotlib.pyplot as plt

# Since pos_mean has shape (2600, 2), we need to handle the 2D position data
# Let's plot the trajectory of mean positions over time
time_steps = np.arange(pos_mean.shape[0])

# Create subplots for x and y positions
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5))

# Plot x-position over time
ax1.plot(time_steps, pos_mean[:, 0], 'b-', label='Mean X Position')
ax1.fill_between(time_steps, 
                 pos_mean[:, 0] - pos_std[:, 0], 
                 pos_mean[:, 0] + pos_std[:, 0], 
                 alpha=0.3, color='blue', label='± 1 Std')
ax1.set_xlabel('Time Step')
ax1.set_ylabel('X Position')
ax1.set_title('X Position Over Time: Mean ± Standard Deviation')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot y-position over time
ax2.plot(time_steps, pos_mean[:, 1], 'r-', label='Mean Y Position')
ax2.fill_between(time_steps, 
                 pos_mean[:, 1] - pos_std[:, 1], 
                 pos_mean[:, 1] + pos_std[:, 1], 
                 alpha=0.3, color='red', label='± 1 Std')
ax2.set_xlabel('Time Step')
ax2.set_ylabel('Y Position')
ax2.set_title('Y Position Over Time: Mean ± Standard Deviation')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIG_PATH, 'bump_positions_over_time_left.png'), dpi=300, bbox_inches='tight')
plt.show()

