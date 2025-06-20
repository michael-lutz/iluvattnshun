"""Visualization functions for attention matrices"""

import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots


def viz_attn(
    attn_weights: list[np.ndarray],
    token_labels: list[str],
    plot_size: tuple[int, int] = (1000, 1000),
    font_size: int = 10,
) -> None:
    """Visualize attention weights using Plotly.

    Args:
        attn_weights: num_layers list of (num_heads, seq_len, seq_len) arrays
        token_labels: list of token labels
        plot_size: tuple of plot size
        font_size: font size
    """
    num_layers = len(attn_weights)
    num_heads = len(attn_weights[0])
    num_tokens = len(attn_weights[0][0])
    tick_vals = list(range(num_tokens))

    fig = make_subplots(
        rows=num_layers,
        cols=num_heads,
        subplot_titles=[f"L{l}H{h}" for l in range(num_layers) for h in range(num_heads)],
        horizontal_spacing=0.03,
        vertical_spacing=0.06,
    )

    for layer_idx in range(num_layers):
        for head_number in range(num_heads):
            attn = attn_weights[layer_idx][head_number]

            fig.add_trace(
                go.Heatmap(
                    z=attn,
                    x=tick_vals,
                    y=tick_vals,
                    colorscale="Viridis",
                    colorbar=dict(len=0.4),
                    zmin=0,
                    zmax=1,
                ),
                row=layer_idx + 1,
                col=head_number + 1,
            )

    # apply axis formatting
    for row in range(1, num_layers + 1):
        for col in range(1, num_heads + 1):
            fig.update_xaxes(
                tickmode="array",
                tickvals=tick_vals,
                ticktext=token_labels,
                type="linear",
                tickfont=dict(size=font_size),
                row=row,
                col=col,
            )
            fig.update_yaxes(
                tickmode="array",
                tickvals=tick_vals,
                ticktext=token_labels,
                type="linear",
                tickfont=dict(size=font_size),
                autorange="reversed",  # flip y-axis
                row=row,
                col=col,
            )

    fig.update_layout(
        height=num_layers * plot_size[1],
        width=num_heads * plot_size[0],
        title_text="Attention Weights by Layer and Head",
        title_x=0.5,
        showlegend=False,
    )

    fig.show()
