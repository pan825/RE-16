import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
from tqdm import tqdm
from numba import prange

def gaussian_2d_periodic(xy, amplitude, x0, y0, sigma_x, sigma_y, theta, offset):
    """2D Gaussian function with periodic boundary conditions"""
    x, y = xy
    
    # Apply periodic boundary conditions
    dx = x - x0
    dy = y - y0
    
    # Find the shortest distance considering periodicity (16x16 grid)
    dx = np.where(dx > 8, dx - 16, dx)
    dx = np.where(dx < -8, dx + 16, dx)
    dy = np.where(dy > 8, dy - 16, dy)
    dy = np.where(dy < -8, dy + 16, dy)
    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    
    return offset + amplitude * np.exp(-(a*dx**2 + 2*b*dx*dy + c*dy**2))


def bump_position(processed_data):

    # Create coordinate grids once
    x = np.arange(processed_data.shape[1])
    y = np.arange(processed_data.shape[0])
    X, Y = np.meshgrid(x, y)
    xy_data = np.vstack([X.ravel(), Y.ravel()])

    # Pre-allocate arrays for better performance
    n_frames = processed_data.shape[2] // 10
    bump_positions = np.zeros((n_frames, 2))
    fit_params = np.zeros((n_frames, 7))

    # Define bounds once
    bounds = ([0, 0, 0, 0.1, 0.1, -np.pi, -np.inf],
            [np.inf, 16, 16, 10, 10, np.pi, np.inf])

    # Pre-compute common values for efficiency
    sigma_init = 2.0
    theta_init = 0.0
    max_iterations = 500  # Reduced from 1000 for faster fitting

    # Vectorized preprocessing for all frames
    time_indices = np.arange(0, processed_data.shape[2], 10)
    data_slices = processed_data[:, :, time_indices]

    # Find all max positions at once
    max_indices = np.unravel_index(np.argmax(data_slices.reshape(data_slices.shape[0] * data_slices.shape[1], -1), axis=0), 
                                data_slices.shape[:2])
    max_vals = data_slices[max_indices[0], max_indices[1], np.arange(len(time_indices))]
    min_vals = np.min(data_slices.reshape(-1, data_slices.shape[2]), axis=0)

    # Fit Gaussian for each time step with optimized loop
    for i in range(n_frames):
        data_slice = data_slices[:, :, i]
        
        # Use pre-computed values
        max_val = max_vals[i]
        min_val = min_vals[i]
        max_y, max_x = max_indices[0][i], max_indices[1][i]
        
        # Initial guess for parameters (vectorized)
        initial_guess = np.array([
            max_val,             # amplitude
            max_x,               # x0 (center x)
            max_y,               # y0 (center y)
            sigma_init,          # sigma_x
            sigma_init,          # sigma_y
            theta_init,          # theta (rotation)
            min_val              # offset
        ])
        
        try:
            # Flatten the data for fitting
            z_data = data_slice.ravel()
            
            # Fit the periodic 2D Gaussian with reduced iterations
            popt, _ = curve_fit(gaussian_2d_periodic, xy_data, z_data, 
                            p0=initial_guess, bounds=bounds, maxfev=max_iterations)
            
            bump_positions[i] = [popt[1], popt[2]]  # x0, y0
            fit_params[i] = popt
            
        except Exception:
            # If fitting fails, use the maximum position (clamp to bounds)
            max_x_clamped = np.clip(max_x, 0, 15)
            max_y_clamped = np.clip(max_y, 0, 15)
            bump_positions[i] = [max_x_clamped, max_y_clamped]
            fit_params[i] = initial_guess

    return bump_positions, fit_params


def analyze_trajectory(bump_positions, fit_params, processed_data, events, plot=True):
    """
    Analyze and plot trajectory from bump positions with phase-based velocity analysis.
    
    Parameters:
    -----------
    bump_positions : array-like
        Array of bump positions over time
    fit_params : array-like
        Array of fit parameters from Gaussian fitting
    processed_data : array-like
        Processed data array with shape (x, y, time)
    events : list
        List of event dictionaries with timing and type information
    
    Returns:
    --------
    dict : Dictionary containing trajectory analysis results
    """
    # Optimized, vectorized trajectory analysis and plotting
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from brian2 import second

    # Ensure arrays
    bump_positions = np.asarray(bump_positions)
    fit_params = np.asarray(fit_params)

    # Vectorized periodic unwrapping (period 16, threshold 8)
    def unwrap_periodic(series: np.ndarray, threshold: float = 8.0, period: float = 16.0) -> np.ndarray:
        diffs = np.diff(series, prepend=series[0])
        jumps = np.zeros_like(diffs, dtype=float)
        pos_jump = diffs[1:] > threshold
        neg_jump = diffs[1:] < -threshold
        jumps[1:][pos_jump] = -period
        jumps[1:][neg_jump] = period
        correction = np.cumsum(jumps)
        return series + correction

    bp = np.empty_like(bump_positions, dtype=float)
    bp[:, 0] = unwrap_periodic(bump_positions[:, 0], threshold=8.0, period=16.0)
    bp[:, 1] = unwrap_periodic(bump_positions[:, 1], threshold=8.0, period=16.0)

    # Time points (seconds)
    time_points = np.arange(0, processed_data.shape[2], 10) / 10000.0

    # Calculate distance between start and end points
    start_point = bp[0]
    end_point = bp[-1]
    total_distance = np.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)

    # # Dynamic phase calculation based on events
    # events = [
    #     {'type': 'visual_cue_on', 'x': 2, 'y': 6, 'strength': 0.5, 'duration': 300},  # ms
    #     {'type': 'visual_cue_off', 'duration': 300},  # ms
    #     {'type': 'shift', 'direction': 'right', 'strength': 0.015, 'duration': 1100},  # ms
    #     {'type': 'shift', 'direction': 'up', 'strength': 0.015, 'duration': 1100},  # ms
    #     {'type': 'shift', 'direction': 'left', 'strength': 0.015, 'duration': 1100},  # ms
    #     {'type': 'shift', 'direction': 'down', 'strength': 0.015, 'duration': 1100},  # ms
    # ]

    # Calculate cumulative timing from events (convert ms to seconds)
    current_time = 0.0
    phases = []
    phase_colors = {'visual_cue_on': 'red', 'visual_cue_off': 'gray', 'right': 'blue', 'up': 'green', 'left': 'orange', 'down': 'purple'}

    for event in events:
        start_time = current_time
        end_time = current_time + event['duration'] / second  # Convert ms to seconds
        
        if event['type'] == 'visual_cue_on':
            phase_name = 'Visual cue on'
            color = phase_colors['visual_cue_on']
        elif event['type'] == 'visual_cue_off':
            phase_name = 'Visual cue off'
            color = phase_colors['visual_cue_off']
        elif event['type'] == 'shift':
            phase_name = event['direction'].capitalize()
            color = phase_colors[event['direction']]
        
        phases.append((start_time, end_time, phase_name, color))
        current_time = end_time

    # Compute per-phase slopes once
    phase_results = {}
    for start_time, end_time, phase_name, color in phases:
        # Convert time_points to plain numpy array for comparison
        time_array = np.asarray(time_points)
        phase_mask = (time_array >= start_time) & (time_array <= end_time)
        if np.sum(phase_mask) > 1:
            t = time_array[phase_mask]
            x = bp[phase_mask, 0]
            y = bp[phase_mask, 1]
            x_slope, x_intercept = np.polyfit(t, x, 1)
            y_slope, y_intercept = np.polyfit(t, y, 1)
            phase_results[phase_name] = {
                'color': color,
                'start': start_time,
                'end': end_time,
                'x_slope': float(x_slope),
                'x_intercept': float(x_intercept),
                'y_slope': float(y_slope),
                'y_intercept': float(y_intercept),
                'speed': float(np.hypot(x_slope, y_slope)),
                't': t,
            }

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 1) Trajectory with LineCollection
    points = bp
    segments = np.stack([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='jet', linewidths=2, alpha=0.8)
    lc.set_array(time_points[:-1])
    ax1.add_collection(lc)
    ax1.set_xlim(points[:, 0].min(), points[:, 0].max())
    ax1.set_ylim(points[:, 1].min(), points[:, 1].max())

    # Scatter markers every 10th point
    time_mask = (np.arange(len(time_points)) % 10) == 0
    ax1.scatter(points[time_mask, 0], points[time_mask, 1],
                c=time_points[time_mask], cmap='jet', s=30, alpha=0.7,
                edgecolors='white', linewidth=0.5)

    # Start/End markers
    ax1.scatter(points[0, 0], points[0, 1], c='green', s=100, marker='o',
                label='Start', edgecolors='black', linewidth=2, zorder=10)
    ax1.scatter(points[-1, 0], points[-1, 1], c='red', s=100, marker='s',
                label='End', edgecolors='black', linewidth=2, zorder=10)
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_title(f'Bump Trajectory (2D Gaussian Fit)\nDistance between start and end: {total_distance:.2f} units')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    fig.colorbar(lc, ax=ax1, label='Time (s)')

    # 2) Position vs time with reused linear fits
    ax2.plot(time_points, points[:, 0], 'b-', label='X position', linewidth=2)
    ax2.plot(time_points, points[:, 1], 'r-', label='Y position', linewidth=2)
    for phase_name, res in phase_results.items():
        t = res['t']
        ax2.plot(t, res['x_slope'] * t + res['x_intercept'], '--', color='darkblue', alpha=0.7, linewidth=1)
        ax2.plot(t, res['y_slope'] * t + res['y_intercept'], '--', color='darkred', alpha=0.7, linewidth=1)

    # Dynamic phase markers and spans
    for start_time, end_time, phase_name, color in phases:
        if end_time <= time_points[-1]:
            ax2.axvline(x=end_time, color=color, linestyle='--', alpha=0.7)
            ax2.text(end_time, ax2.get_ylim()[1] * 0.9, f'{phase_name} end', rotation=90,
                     verticalalignment='top', fontsize=8, color=color)
        ax2.axvspan(start_time, min(end_time, time_points[-1]), alpha=0.1, color=color, label=phase_name)

    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position')
    ax2.set_title('Bump Position vs Time (with Phase Slopes)')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if plot:
        plt.show()
    else:
        plt.close()
    # # Print distance and velocity information
    # print(f"\nTrajectory Analysis:")
    # print("=" * 50)
    # print(f"Start point: ({start_point[0]:.3f}, {start_point[1]:.3f})")
    # print(f"End point: ({end_point[0]:.3f}, {end_point[1]:.3f})")
    # print(f"Total distance between start and end: {total_distance:.3f} units")

    # print("\nPhase-based velocity analysis:")
    # print("=" * 50)
    # for start_time, end_time, phase_name, color in phases:
    #     if phase_name in phase_results:
    #         res = phase_results[phase_name]
    #         print(f"{phase_name:15s}: vx={res['x_slope']:6.3f}, vy={res['y_slope']:6.3f}, speed={res['speed']*np.pi/16:6.3f} rad/s")

    # print("\nPhase summary:")
    # print("=" * 50)
    # for start_time, end_time, phase_name, color in phases:
    #     if phase_name in phase_results:
    #         res = phase_results[phase_name]
    #         duration = end_time - start_time
    #         print(f"{phase_name:15s}: {start_time:.1f}-{end_time:.1f}s ({duration:.1f}s duration), speed={res['speed']*np.pi/16:.3f} rad/s")

    # print(f"Distance between start and end: {total_distance:.3f} units")
    
    # Return analysis results
    return {
        'bump_positions': bp,
        'time_points': time_points,
        'start_point': start_point,
        'end_point': end_point,
        'total_distance': total_distance,
        'phase_results': phase_results,
        'phases': phases
    }
