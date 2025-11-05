import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from typing import Optional
from io import BytesIO

try:
    import imageio.v2 as imageio
except Exception:  # fallback if imageio.v2 is unavailable
    import imageio

try:
    from PIL import Image
except Exception:
    Image = None


def animate_3d_volume(
    data: np.ndarray,
    title: str = "3D Volume Animation",
    value_threshold: float = 5.0,
    frame_stride: int = 1,
    width: int = 800,
    height: int = 600,
    point_size: int = 3,
    opacity: float = 0.8,
    colorscale: str = 'Blues',
    output_html: Optional[str] = None,
    output_gif: Optional[str] = None,
    output_mp4: Optional[str] = None,
    fps: int = 10,
    show: bool = True,
) -> go.Figure:
    """Create an interactive 3D time animation and optionally export to HTML/GIF/MP4.

    Args:
        data: 4D array shaped (nx, ny, nz, nt).
        title: Plot title.
        value_threshold: Only plot points with values greater than this.
        frame_stride: Use every Nth frame along time.
        width: Figure width in pixels.
        height: Figure height in pixels.
        point_size: Marker size for scatter points.
        opacity: Marker opacity.
        colorscale: Plotly colorscale name.
        output_html: If provided, save interactive HTML to this path.
        output_gif: If provided, save a GIF to this path (requires kaleido + imageio + Pillow).
        output_mp4: If provided, save an MP4 to this path (requires kaleido + imageio + ffmpeg).
        fps: Frames per second for GIF/MP4 exports.

    Returns:
        The constructed Plotly Figure with animation frames and controls.
    """
    if data.ndim != 4:
        raise ValueError("data must be 4D (x, y, z, t)")

    nx, ny, nz, nt = data.shape

    # Create coordinate grids once
    x, y, z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
    x_flat = x.ravel()
    y_flat = y.ravel()
    z_flat = z.ravel()

    def make_trace(t_index: int) -> go.Scatter3d:
        volume_t = data[:, :, :, t_index]
        values_flat = volume_t.ravel()
        mask = values_flat > value_threshold
        x_vals = x_flat[mask]
        y_vals = y_flat[mask]
        z_vals = z_flat[mask]
        c_vals = values_flat[mask]
        hover_text = [f'({xi},{yi},{zi}): {val:.3f}' for xi, yi, zi, val in zip(x_vals, y_vals, z_vals, c_vals)]
        return go.Scatter3d(
            x=x_vals,
            y=y_vals,
            z=z_vals,
            mode='markers',
            marker=dict(
                size=point_size,
                color=c_vals,
                colorscale=colorscale,
                opacity=opacity,
                colorbar=dict(title="Value"),
            ),
            text=hover_text,
            hovertemplate='<b>Position:</b> (%{x}, %{y}, %{z})<br><b>Value:</b> %{marker.color:.3f}<extra></extra>',
        )

    # Choose an initial time index with any points above threshold if possible
    initial_t = 0
    while initial_t < nt and not np.any(data[:, :, :, initial_t] > value_threshold):
        initial_t += 1
    if initial_t >= nt:
        initial_t = 0

    # Build base figure
    fig = go.Figure(
        data=[make_trace(initial_t)],
        layout=go.Layout(
            title=f'{title} (t={initial_t})',
            scene=dict(
                xaxis_title='X Index',
                yaxis_title='Y Index',
                zaxis_title='Z Index',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            ),
            width=width,
            height=height,
            updatemenus=[
                dict(
                    type='buttons',
                    showactive=False,
                    buttons=[
                        dict(
                            label='Play',
                            method='animate',
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=int(1000 / max(fps, 1)), redraw=True),
                                    fromcurrent=True,
                                    transition=dict(duration=0),
                                ),
                            ],
                        ),
                        dict(
                            label='Pause',
                            method='animate',
                            args=[
                                [None],
                                dict(
                                    frame=dict(duration=0, redraw=False),
                                    mode='immediate',
                                    transition=dict(duration=0),
                                ),
                            ],
                        ),
                    ],
                )
            ],
            sliders=[dict(steps=[], currentvalue=dict(prefix='t = '))],
        ),
        frames=[],
    )

    # Create frames and slider steps
    frame_indices = list(range(0, nt, max(frame_stride, 1)))
    frames = []
    slider_steps = []
    for t in frame_indices:
        frame = go.Frame(data=[make_trace(t)], name=str(t), layout=go.Layout(title=f'{title} (t={t})'))
        frames.append(frame)
        slider_steps.append(
            dict(
                method='animate',
                label=str(t),
                args=[[str(t)], dict(mode='immediate', frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
            )
        )

    fig.frames = frames
    fig.update_layout(sliders=[dict(active=0, steps=slider_steps, currentvalue=dict(prefix='t = '))])

    # Optional exports
    if output_html:
        pio.write_html(fig, file=output_html, auto_play=True, include_plotlyjs='cdn')

    if output_gif or output_mp4:
        # Rendering frames -> PNGs via kaleido, then encode with imageio
        try:
            images = []
            for fr in frames:
                # Apply frame data to fig for consistent camera/layout
                fig.update(data=fr.data)
                # Render current state to PNG bytes
                png_bytes = pio.to_image(fig, format='png', width=width, height=height)
                if Image is not None:
                    img = Image.open(BytesIO(png_bytes)).convert('RGBA')
                    images.append(np.array(img))
                else:
                    # Fallback: try imageio to decode from bytes
                    images.append(imageio.imread(BytesIO(png_bytes)))

            if output_gif:
                imageio.mimsave(output_gif, images, fps=fps)
            if output_mp4:
                # Requires ffmpeg in PATH
                writer = imageio.get_writer(output_mp4, fps=fps, codec='libx264', quality=8)
                for im in images:
                    writer.append_data(im)
                writer.close()
        except Exception as exc:
            print(
                "Export failed. Ensure 'kaleido', 'imageio', 'Pillow' and 'ffmpeg' (for MP4) are installed.\n",
                f"Details: {exc}",
            )

    if show:
        fig.show()
    return fig


# # Example usage
# # data shape: (nx, ny, nz, nt)
# fig = animate_3d_volume(
#     processed_data,
#     title="3D Volume Animation",
#     value_threshold=5.0,
#     frame_stride=1,
#     output_html="/home/pajucg/2D/animation.html",      # optional
#     output_gif="/home/pajucg/2D/animation.gif",        # optional (needs kaleido, imageio, Pillow)
#     output_mp4="/home/pajucg/2D/animation.mp4",        # optional (needs ffmpeg)
#     fps=10
# )