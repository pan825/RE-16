import io
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from brian2 import *
import RE16 as RE16
import utils
import fit
import visualize

@dataclass
class Parameters:
    """Parameters for the ring attractor network model.
    
    All weights represent connection strengths between neural populations.
    """
    w_EE: float = 0.719  # EB <-> EB 
    w_EI: float = 0.143  # R -> EB 
    w_IE: float = 0.74   # EB -> R 
    w_II: float = 0.01   # R <-> R 
    w_EP: float = 0.012  # EB -> PEN 
    w_PE: float = 0.709  # PEN -> EB 
    w_PP: float = 0.01   # PEN <-> PEN 
    sigma: float = 0.001  # Noise level

class Simulator:
    """Implementation of a ring attractor neural network model.
    
    This class handles simulation, data processing, analysis, and visualization
    of a ring attractor network based on the Drosophila central complex.
    """
    
    def __init__(self, parameters: Parameters = None):
        """Initialize the ring attractor network with given parameters.
        
        Args:
            parameters: Configuration parameters for the network
        """
        self.parameters = parameters or Parameters()
        
        # Simulation results
        self.time: Optional[np.ndarray] = None
        self.fr: Optional[np.ndarray] = None
        self.fr_pen: Optional[np.ndarray] = None
        self.fr_r: Optional[np.ndarray] = None
        
        # Processed results
        self.t_proc: Optional[np.ndarray] = None
        self.fr_proc: Optional[np.ndarray] = None
        
        # Gaussian fit results
        self.gt: Optional[np.ndarray] = None
        self.gx: Optional[np.ndarray] = None
        self.gfr: Optional[np.ndarray] = None
        self.gw: Optional[np.ndarray] = None
        
        # Analysis results
        self.slope: Optional[float] = None
        self.r_squared: Optional[float] = None
        self.std_err: Optional[float] = None
        self.coefficient_variation: Optional[float] = None
        self.angular_velocity: Optional[float] = None
        self.rotations_per_second: Optional[float] = None
        
        # Per-event analysis results
        self.event_timings: Optional[List[Dict[str, float]]] = None
        self.event_velocities: Optional[List[Dict[str, Any]]] = None
        
        # Events for simulation
        self.events: Optional[List[Dict[str, Any]]] = None
        
    def set_events(self, events: List[Dict[str, Any]]):
        """Set the events for the network simulation.
        Example:
        events = [
            {'type': 'visual_cue_on', 'location': 0, 'strength': 0.05, 'duration': 300*ms},
            {'type': 'visual_cue_off', 'duration': 300*ms},
            {'type': 'shift', 'direction': 'right', 'strength': 0.015, 'duration': 1000*ms},
            {'type': 'shift', 'direction': 'left', 'strength': 0.015, 'duration': 1000*ms},
        ]
        Args:
            events: List of event dictionaries for event-driven simulation
        """
        self.events = events
        return 
        
    def run(self, events: List[Dict[str, Any]] = None):
        """Run the simulation with events.
        
        Args:
            events: Optional events list. If provided, overrides the events set in setup()
        """
        # Use provided events or fall back to setup events
        simulation_events = events if events is not None else self.events
        
        if simulation_events is None:
            raise ValueError("No events provided. Use setup() or pass events to run()")
        
        t, fr_epg, fr_pen, fr_r = RE16.simulator(
            **self.parameters.__dict__,
            events=simulation_events,
        )
        
        
        # Store results (append if multiple simulations)
        self.time = utils.add_array(self.time, t, axis=0) 
        self.fr = utils.add_array(self.fr, fr_epg, axis=1)
        self.fr_pen = utils.add_array(self.fr_pen, fr_pen, axis=1)
        self.fr_r = utils.add_array(self.fr_r, fr_r, axis=1)


    def process_data(self):
        """Process raw simulation data for analysis.
        
        Transforms the raw firing rates into a format suitable for analysis,
        including conversion to ellipsoid body (EB) representation.
        """
        # Re-use shared preprocessing helper
        self.t_proc, self.fr_proc = self._expand_and_convert(self.fr)
    
    def save(self, file_path='simulation_results.dat', folder=None):
        """Save simulation results to a file.
        
        Args:
            file_path: Name of the file to save results to
            folder: Optional folder path for the file
        """
        if folder is not None:
            if not os.path.exists(folder):
                os.makedirs(folder)
            file_path = os.path.join(folder, file_path)
    
        t = self.time
        fr = self.fr
            
        with open(file_path, 'w') as file:
            for i in range(len(t)):
                row = f'{t[i]} '
                row += ' '.join([f'{fr[j,i]}' for j in range(fr.shape[0])])
                file.write(row + '\n')
                
        print(f'\n{time.strftime("%Y-%m-%d %H:%M:%S")}: file saved as {file_path}')

    def load(self, file_path='simulation_results.dat'):
        """Load simulation results from a file and preprocess them."""
        data = np.loadtxt(file_path)
        self.time = data[:, 0]
        # Transpose so that shape is (neurons, time)
        self.fr = data[:, 1:].T
        # Shared preprocessing
        self.t_proc, self.fr_proc = self._expand_and_convert(self.fr)

    def fit_gaussian(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fit a Gaussian to the processed firing rate data.
        
        Returns:
            Tuple containing time points, positions, amplitudes, and widths
        """
        # Ensure data is processed before fitting
        self._ensure_processed_data()
        
        t, fr = self.t_proc, self.fr_proc
        gt, gx, gfr, gw = fit.gau_fit(t, fr)
        
        # Translate the Gaussian parameters to a consistent coordinate system
        gt, gx, gfr, gw = fit.translate_gau(
            gt, gx, gfr, gw
        )
        
        self.gx = gx
        self.gt = gt
        self.gw = gw
        self.gfr = gfr
        
        return gt, gx, gfr, gw
        
    def fit_velocity(self, time_threshold=None, time_end=None):
        """Calculate the angular velocity from Gaussian position data.
        
        Args:
            time_threshold: Starting time for the fit (defaults to None)
            time_end: Ending time for the fit (defaults to end of simulation)
            
        Returns:
            Tuple of (slope, r_squared, std_err, CV)
        """
        self._ensure_gaussian_fit()
            
        slope, r2 , std_err, CV = fit.fit_slope(
            self.gt, 
            self.gx, 
            time_threshold, 
            time_end
        )
        
        self.slope = slope
        self.r_squared = r2
        self.std_err = std_err
        self.coefficient_variation = CV
        self.angular_velocity = self.slope * 2 * np.pi / 16
        
        # Rotations per second
        self.rotations_per_second = self.angular_velocity / (2 * np.pi)
        
        return slope, r2, std_err, CV
        
    def calculate_event_timings(self):
        """Calculate start and end times for each event."""
        if self.events is None:
            raise ValueError("No events available. Run setup() first.")
            
        event_timings = []
        current_time = 0.0
        
        for i, event in enumerate(self.events):
            duration = event.get('duration', 0)
            
            # Convert brian2 units to seconds if needed
            if hasattr(duration, 'value'):
                duration_sec = duration.value
            else:
                duration_sec = float(duration)
                
            start_time = current_time
            end_time = current_time + duration_sec
            
            event_timings.append({
                'event_index': i,
                'event_type': event.get('type', 'unknown'),
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration_sec,
                'event_data': event,
                'strength': event.get('strength', None)
            })
            
            current_time = end_time
            
        self.event_timings = event_timings
        return event_timings
        
    def fit_velocity_per_event(self):
        """Fit velocity for each individual event period."""
        if self.event_timings is None:
            self.calculate_event_timings()
            
        self._ensure_gaussian_fit()
        
        event_velocities = []
        
        for timing in self.event_timings:
            event_index = timing['event_index']
            event_type = timing['event_type']
            start_time = timing['start_time']
            end_time = timing['end_time']
            
            try:
                # Fit velocity for this specific time window
                slope, r2, std_err, CV = fit.fit_slope(
                    self.gt, 
                    self.gx, 
                    start_time, 
                    end_time
                )
                
                angular_velocity = slope * 2 * np.pi / 16
                rotations_per_second = angular_velocity / (2 * np.pi)
                
                event_velocity = {
                    'event_index': event_index,
                    'event_type': event_type,
                    'start_time': start_time,
                    'end_time': end_time,
                    'slope': slope,
                    'r_squared': r2,
                    'std_err': std_err,
                    'coefficient_variation': CV,
                    'angular_velocity': angular_velocity,
                    'angular_velocity_deg': np.rad2deg(angular_velocity),
                    'rotations_per_second': rotations_per_second,
                    'event_data': timing['event_data'],
                    'strength': timing['strength']
                }
                
            except Exception as e:
                # Handle cases where fitting fails (e.g., insufficient data)
                event_velocity = {
                    'event_index': event_index,
                    'event_type': event_type,
                    'start_time': start_time,
                    'end_time': end_time,
                    'slope': np.nan,
                    'r_squared': np.nan,
                    'std_err': np.nan,
                    'coefficient_variation': np.nan,
                    'angular_velocity': np.nan,
                    'angular_velocity_deg': np.nan,
                    'rotations_per_second': np.nan,
                    'error': str(e),
                    'event_data': timing['event_data'],
                    'strength': timing['strength']
                }
                
            event_velocities.append(event_velocity)
            
        self.event_velocities = event_velocities
        # return event_velocities

    
    def reset(self):
        """Clear all simulation and analysis results."""
        self.fr = None
        self.time = None
        self.t_proc = None
        self.fr_proc = None
        self.gx = None
        self.gt = None
        self.gw = None
        self.gfr = None
        self.slope = None
        self.r_squared = None
        self.std_err = None
        self.angular_velocity = None
        self.events = None
        self.event_timings = None
        self.event_velocities = None
        
    def _ensure_gaussian_fit(self):
        """Ensure that Gaussian fit has been performed."""
        if self.gx is None:
            self.fit_gaussian()
            
    def _ensure_processed_data(self):
        """Ensure that data processing has been performed."""
        if self.t_proc is None:
            self.process_data()

        
    def plot_raw(self, title=None, file_name=None, region='EB', y_label='Time (s)', 
                         cmap='Blues', save=False, folder='figures', plot_gaussian=True, 
                         figsize=(10, 2.5), eip2eb=True):
        """Visualize the raw neural activity.
        
        Args:
            title: Plot title
            file_name: Filename for saving the plot
            region: Brain region to label ('EB' or other)
            y_label: Label for y-axis
            cmap: Colormap for the plot
            save: Whether to save the plot
            folder: Folder for saving
            plot_gaussian: Whether to plot the Gaussian fit
            figsize: Figure size
            eip2eb: Whether to convert EIP to EB coordinates
        """
        t = self.time
        fr = self.fr.T
        
        if eip2eb:
            fr_with_zeros = np.zeros((fr.shape[0], fr.shape[1] + 2))
            fr_with_zeros[:, :8] = fr[:, :8]
            fr_with_zeros[:, 10:] = fr[:, 8:]
            eb_fr = utils.eip_to_eb_fast(fr_with_zeros)
            eb_fr = eb_fr.T
        else:
            eb_fr = fr
            
        plt.figure(figsize=figsize)
        plt.pcolormesh(t, range(eb_fr.shape[0]), eb_fr, cmap=cmap, shading='nearest')
        plt.colorbar(label='Firing Rate [Hz]')    
        plt.title(title)
        plt.xlabel(y_label)
        plt.ylabel('EB region' if region == 'EB' else 'Neuron ID')
        plt.yticks([0, 4, 11, 15], ['R8', 'R4', 'L4', 'L8'] if region == 'EB' else [0, 5, 10, 15])
        
        if plot_gaussian:
            self._ensure_processed_data()
            g_t, g_x, g_y, g_w = fit.gau_fit(self.t_proc, self.fr_proc)
            plt.plot(g_t, g_x, 'r', linewidth=3)
        
        if save:
            plt.savefig(os.path.join(folder, file_name))
            plt.close()
            
    def plot(self, show_events=True, show_velocity=True, figsize=(12, 6)):
        self.fit_velocity_per_event()

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=figsize, height_ratios=[2, 1], sharex=True
        )

        im = ax1.pcolormesh(
            self.t_proc, range(self.fr_proc.shape[0]), self.fr_proc,
            cmap='Blues', shading='nearest'
        )

        try:
            g_t, g_x, g_y, g_w = fit.gau_fit(self.t_proc, self.fr_proc)
            ax1.plot(g_t, g_x, 'r-', linewidth=3, label='Bump position')
            ax1.legend()
        except:
            pass

        ax1.set_ylabel('Neuron ID')
        ax1.set_title('Ring Attractor Activity with Event Phases')

        if hasattr(self, 'gx') and hasattr(self, 'gt'):
            ax2.plot(self.gt, self.gx * 2 * np.pi/16, 'b-', linewidth=2)
            ax2.set_ylabel('Position (rad)')
        ax2.set_xlabel('Time (s)')

        if hasattr(self, 'gt'):
            tmin = min(np.min(self.t_proc), np.min(self.gt))
            tmax = max(np.max(self.t_proc), np.max(self.gt))
        else:
            tmin, tmax = np.min(self.t_proc), np.max(self.t_proc)
        ax1.set_xlim(tmin, tmax)  

        if show_events and hasattr(self, 'event_velocities') and self.event_velocities:
            for ev in self.event_velocities:
                start_time, end_time = ev['start_time'], ev['end_time']
                for ax in (ax1, ax2):
                    ax.axvline(start_time, color='red', linestyle='--', alpha=0.7)
                    ax.axvline(end_time, color='red', linestyle='--', alpha=0.7)
                if show_velocity and not np.isnan(ev['angular_velocity']):
                    mid =  (start_time + end_time) / 2
                    if ev['event_type'] == 'shift':
                        txt = f"{ev['event_type']}\n{ev['angular_velocity_deg']:.1f}°/s\nStrength: {ev.get('strength', 'N/A')}"
                    else:
                        txt = f"{ev['event_type']}\n{ev['angular_velocity_deg']:.1f}°/s"
                    y_top = ax1.get_ylim()[1] * 0.9
                    ax1.annotate(txt, (mid, y_top), ha='center', va='top', fontsize=9,
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

        plt.tight_layout()
        plt.show()

            
    def xt_plot(self, v=False, a=False):
        """Plot the position (gx) over time."""
        self._ensure_gaussian_fit()
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.gt, self.gx * 2 * np.pi/16, 'b-', linewidth=2, label='Position')
        plt.xlabel('Time (s)')
        plt.ylabel('Position (rad)')
        if v:
            v = np.gradient(self.gx * 2 * np.pi/16, self.gt)
            plt.plot(self.gt, v, linewidth=2, label='Velocity')
        if a:
            a = np.gradient(v, self.gt)
            plt.plot(self.gt, a, linewidth=2, label='Acceleration')
        plt.legend()        
        plt.title('Position vs Time')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def print_event_velocities(self):
        """Print detailed velocity analysis for each event."""
        if self.event_velocities is None:
            self.fit_velocity_per_event()
            
        print("\n" + "="*60)
        print("PER-EVENT VELOCITY ANALYSIS")
        print("="*60)
        
        for i, ev in enumerate(self.event_velocities):
            print(f"\nEvent {ev['event_index'] + 1}: {ev['event_type']}")
            print(f"Time: {ev['start_time']:.3f}s - {ev['end_time']:.3f}s (duration: {ev['end_time'] - ev['start_time']:.3f}s)")
            
            if not np.isnan(ev['angular_velocity']):
                print(f"Angular velocity: {ev['angular_velocity']:.3f} ± {ev['std_err']*2*np.pi/16:.3f} rad/s")
                print(f"                  {ev['angular_velocity_deg']:.3f} ± {np.rad2deg(ev['std_err']*2*np.pi/16):.3f} deg/s")
                print(f"Rotations/sec:    {ev['rotations_per_second']:.3f} Hz")
                
                # Color code r_squared based on quality
                if abs(ev['r_squared']) >= 0.95:
                    print('\033[92m' + f"R-squared: {ev['r_squared']:.3f}" + '\033[0m')
                elif np.isnan(ev['r_squared']):
                    print('\033[91m' + f"R-squared: {ev['r_squared']:.3f}" + '\033[0m')
                else: 
                    print(f"R-squared: {ev['r_squared']:.3f}")
            else:
                print('\033[91m' + "Velocity fitting failed" + '\033[0m')
                if 'error' in ev:
                    print(f"Error: {ev['error']}")
                    
            # Show event parameters
            event_data = ev['event_data']
            if 'strength' in event_data:
                print(f"Strength: {event_data['strength']}")
            if 'location' in event_data:
                print(f"Location: {event_data['location']}")
            if 'direction' in event_data:
                print(f"Direction: {event_data['direction']}")
                
        print("="*60)
            
    def summary(self):
        """Print a summary of the simulation and analysis results."""
        self.fit_velocity()
        self._ensure_gaussian_fit()
        
        speed_rad = f'{self.angular_velocity:.3f}'
        speed_deg = f'{np.rad2deg(self.angular_velocity):.3f}'
        err_rad = f'{self.std_err*2*np.pi/16:.3f}'
        err_deg = f'{np.rad2deg(self.std_err*2*np.pi/16):.3f}'
        
        print('='* 40)
        print(f'Angular velocity: {speed_rad:>8} ± {err_rad:>8} [rad/s]')
        print(f'                  {speed_deg:>8} ± {err_deg:>8} [deg/s]')
        print(f'Rotations/sec:    {self.rotations_per_second:>8.3f} ± {self.std_err/16:>8.3f} [Hz]')
        
        # Color code r_squared based on quality
        if abs(self.r_squared) >= 0.95:
            print('\033[92m' + f'R-squared: {self.r_squared:.3f}' + '\033[0m')
        elif np.isnan(self.r_squared):
            print('\033[91m' + f'R-squared: {self.r_squared:.3f}' + '\033[0m')
        else: 
            print(f'R-squared: {self.r_squared:.3f}')
        
        # Color code bump width based on quality
        mean_bump_width = np.rad2deg(np.mean(self.gw*np.pi/8))
        std_bump_width = np.rad2deg(np.std(self.gw*np.pi/8))
        
        if mean_bump_width >= 360:
            print('\033[91m' + f'Average bump width: {mean_bump_width:.3f} ± {std_bump_width:.3f} [deg]' + '\033[0m')
        else:
            print(f'Average bump width: {mean_bump_width:.3f} ± {std_bump_width:.3f} [deg]')
        
        print(f'Average firing rate: {np.mean(self.gfr):.3f} ± {np.std(self.gfr):.3f} [Hz]')
        
        # Print the parameters
        print('Network parameters:')
        for param, value in self.parameters.__dict__.items():
            print(f'  {param}: {value}')
            
        if self.events is not None:
            print(f'Events: {len(self.events)} event(s) configured')
            for i, event in enumerate(self.events):
                print(f'  Event {i+1}: {event}')
        print('='*40)
        
        # Add per-event velocity analysis
        self.print_event_velocities()
        
    # ---------------------------------------------------------------------
    # Helper utilities (private)
    # ---------------------------------------------------------------------

    def _expand_and_convert(self, fr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Insert missing neuron positions and convert to EB representation.

        Args:
            fr: Firing rates with shape (neurons, time)
        
        Returns:
            Tuple (processed_time, processed_firing_rates)
        """
        # Insert zeros for the two absent neurons in the circuit
        expanded = np.insert(fr, 8, 0, axis=0)
        expanded = np.insert(expanded, 9, 0, axis=0)
        # Temporal convolution
        conv_rates, conv_time = utils.conv(expanded)
        # Convert to ellipsoid body coordinates
        eb_fr = utils.eip_to_eb_fast(conv_rates.T)
        return conv_time, eb_fr.T

if __name__ == '__main__':
    from brian2 import ms
    
    # Define events
    events = [
        {'type': 'visual_cue_on', 'location': 0, 'strength': 0.05, 'duration': 300*ms},
        {'type': 'visual_cue_off', 'duration': 300*ms},
        {'type': 'shift', 'direction': 'right', 'strength': 0.015, 'duration': 1000*ms},
        {'type': 'shift', 'direction': 'left', 'strength': 0.015, 'duration': 1000*ms},
    ]
    
    # Create and run simulation
    network = Simulator()
    network.setup(events)
    network.run()
    network.process_data()
    network.save(file_path='simulation_results.dat', folder='results')
    network.plot(title='Activity', file_name='activity.png', region='EB', save=True, folder='figures')
    
