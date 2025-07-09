"""Visualization functions for attention matrices"""

import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots


def get_fig(
    attn_weights: list[np.ndarray],
    token_labels: list[str],
    base_height: int = 1000,
    font_size: int = 10,
    selected_layers: list[int] | None = None,
    selected_heads: list[int] | None = None,
    selected_x_toks: list[int] | None = None,
    selected_y_toks: list[int] | None = None,
) -> go.Figure:
    """Visualize attention weights using Plotly.

    Args:
        attn_weights: num_layers list of (num_heads, seq_len, seq_len) arrays
        token_labels: list of token labels
        plot_size: tuple of plot size
        font_size: font size
        selected_layers: list of layer indices to visualize (None for all)
        selected_heads: list of head indices to visualize (None for all)
        selected_x_toks: list of x token indices to visualize (None for all)
        selected_y_toks: list of y token indices to visualize (None for all)
    """
    num_layers = len(attn_weights)
    num_heads = len(attn_weights[0])
    num_tokens = len(attn_weights[0][0])

    # determine which layers and heads to visualize
    if selected_layers is None:
        layers_to_plot = list(range(num_layers))
    else:
        layers_to_plot = [l for l in selected_layers if 0 <= l < num_layers]

    if selected_heads is None:
        heads_to_plot = list(range(num_heads))
    else:
        heads_to_plot = [h for h in selected_heads if 0 <= h < num_heads]

    # determine which tokens to visualize
    if selected_x_toks is None:
        x_tokens_to_plot = list(range(num_tokens))
    else:
        x_tokens_to_plot = [t for t in selected_x_toks if 0 <= t < num_tokens]
    if selected_y_toks is None:
        y_tokens_to_plot = list(range(num_tokens))
    else:
        y_tokens_to_plot = [t for t in selected_y_toks if 0 <= t < num_tokens]

    if not layers_to_plot:
        raise ValueError("No valid layers selected for visualization")
    if not heads_to_plot:
        raise ValueError("No valid heads selected for visualization")
    if not x_tokens_to_plot:
        raise ValueError("No valid x tokens selected for visualization")
    if not y_tokens_to_plot:
        raise ValueError("No valid y tokens selected for visualization")

    # filter token labels and create tick values
    filtered_x_token_labels = [token_labels[i] for i in x_tokens_to_plot]
    filtered_y_token_labels = [token_labels[i] for i in y_tokens_to_plot]
    x_tick_vals = list(range(len(x_tokens_to_plot)))
    y_tick_vals = list(range(len(y_tokens_to_plot)))

    # create subplot titles
    subplot_titles = []
    for l in layers_to_plot:
        for h in heads_to_plot:
            subplot_titles.append(f"L{l+1}H{h+1}")

    fig = make_subplots(
        rows=len(layers_to_plot),
        cols=len(heads_to_plot),
        subplot_titles=subplot_titles,
        horizontal_spacing=0.03,
        vertical_spacing=0.06,
    )

    for row_idx, layer_idx in enumerate(layers_to_plot):
        for col_idx, head_number in enumerate(heads_to_plot):
            # extract the selected token subset from attention matrix
            attn = attn_weights[layer_idx][head_number]
            attn_subset = attn[np.ix_(y_tokens_to_plot, x_tokens_to_plot)]

            # set upper triangle to NaN to make it white (causal masking visualization)
            attn_subset = attn_subset.copy()
            for i in range(len(y_tokens_to_plot)):
                for j in range(len(x_tokens_to_plot)):
                    # If y_token_index > x_token_index, we're above the diagonal
                    if y_tokens_to_plot[i] < x_tokens_to_plot[j]:
                        attn_subset[i, j] = np.nan

            # create custom colorscale that starts with black for zeros
            colorscale = [[0, "black"], [0.001, "rgb(68,1,84)"], [1, "rgb(253,231,37)"]]

            fig.add_trace(
                go.Heatmap(
                    z=attn_subset,
                    x=x_tick_vals,
                    y=y_tick_vals,
                    colorscale=colorscale,
                    colorbar=dict(len=0.4),
                    zmin=0,
                    zmax=1,
                ),
                row=row_idx + 1,
                col=col_idx + 1,
            )

    # apply axis formatting
    for row in range(1, len(layers_to_plot) + 1):
        for col in range(1, len(heads_to_plot) + 1):
            fig.update_xaxes(
                tickmode="array",
                tickvals=x_tick_vals,
                ticktext=filtered_x_token_labels,
                type="linear",
                tickfont=dict(size=font_size),
                row=row,
                col=col,
            )
            fig.update_yaxes(
                tickmode="array",
                tickvals=y_tick_vals,
                ticktext=filtered_y_token_labels,
                type="linear",
                tickfont=dict(size=font_size),
                autorange="reversed",  # flip y-axis
                row=row,
                col=col,
            )

    # Calculate proportional dimensions based on token counts
    aspect_ratio = len(x_tokens_to_plot) / len(y_tokens_to_plot)
    proportional_width = base_height * aspect_ratio

    fig.update_layout(
        height=len(layers_to_plot) * base_height,
        width=len(heads_to_plot) * proportional_width,
        title_text="Attention Weights by Layer and Head",
        title_x=0.5,
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig
