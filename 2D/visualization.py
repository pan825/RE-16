import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
from IPython.display import HTML
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

def create_3d_animation(processed_data, vmin=0, vmax=300, frame_stride=200, frame_duration=0.1, transition_duration=200):
    """
    Create an interactive 3D surface plot animation with Plotly.
    
    Parameters:
    -----------
    processed_data : numpy.ndarray
        3D array of shape (height, width, time_frames) containing neural activity data
    vmin : float, optional
        Minimum value for colorbar range (default: 0)
    vmax : float, optional
        Maximum value for colorbar range (default: 300)
    frame_stride : int, optional
        Sample every nth frame to reduce file size (default: 200)
    frame_duration : float, optional
        Duration of each frame in milliseconds (default: 0.1)
    transition_duration : int, optional
        Transition duration between frames in milliseconds (default: 200)
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive 3D animation figure
    """
    
    # Create interactive 3D surface plot with Plotly
    fig = go.Figure()

    # Add initial frame
    fig.add_trace(go.Surface(
        z=processed_data[:, :, 0],
        colorscale='Blues',
        cmin=vmin,
        cmax=vmax,
        showscale=True,
        colorbar=dict(title="Firing Rate [Hz]")
    ))

    # Update layout for better visualization
    fig.update_layout(
        title='Neural Activity Evolution Over Time',
        scene=dict(
            xaxis_title='X Position',
            yaxis_title='Y Position', 
            zaxis_title='Firing Rate [Hz]',
            xaxis=dict(range=[0, 16]),
            yaxis=dict(range=[0, 16]),
            zaxis=dict(range=[vmin, vmax]),
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=800,
        height=600
    )

    # Create frames for animation
    frames = []
    slider_steps = []

    frame_names = []
    for i in range(0, processed_data.shape[2], frame_stride):
        frame_name = f'frame_{i}'
        frame_names.append(frame_name)
        frame = go.Frame(
            data=[go.Surface(
                z=processed_data[:, :, i],
                colorscale='Blues',
                cmin=vmin,
                cmax=vmax,
                showscale=True
            )],
            name=frame_name,
            layout=go.Layout(
                title=f'Neural Activity at t = {i/10000:.2f} s'
            )
        )
        frames.append(frame)
        slider_steps.append(
            dict(
                method='animate',
                label=f'{i/10000:.2f}s',
                args=[[frame_name], dict(mode='immediate', frame=dict(duration=frame_duration, redraw=True), transition=dict(duration=transition_duration))],
            )
        )

    # Add frames to the 3D figure
    fig.frames = frames
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(
                        args=[None, {"frame": {"duration": frame_duration, "redraw": True},
                                    "fromcurrent": True, 
                                    "transition": {"duration": transition_duration},
                                    "mode": "immediate"}],
                        label="Play",
                        method="animate"
                    ),
                    dict(
                        args=[[None], {"frame": {"duration": 0, "redraw": False},
                                      "mode": "immediate",
                                      "transition": {"duration": 0}}],
                        label="Pause",
                        method="animate"
                    )
                ]),
                pad={"r": 10, "t": 87},
                showactive=False,
                x=0.011,
                xanchor="right",
                y=0,
                yanchor="top"
            ),
        ],
        sliders=[dict(
            active=0,
            currentvalue={"prefix": "Time: "},
            steps=slider_steps
        )]
    )

    return fig

