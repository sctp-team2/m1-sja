"""Shared Plotly theming and palette.

Centralizes color choices and layout defaults so every chart in the
dashboard has the same look. Page modules import `PALETTE`, `themed`,
and the colormap helpers — they should not hardcode hex values.
"""

from __future__ import annotations

import plotly.graph_objects as go

PALETTE = {
    "primary": "#1E3A8A",
    "secondary": "#64748B",
    "success": "#10B981",
    "warning": "#F59E0B",
    "critical": "#DC2626",
    "background": "#F8FAFC",
    "text": "#0F172A",
    "muted": "#64748B",
}

AGENCY_COLORS = {True: PALETTE["warning"], False: PALETTE["primary"]}

# Diverging for z-scores / premiums (red = high, blue = low) per spec.
DIVERGING_SCALE = "RdBu_r"
SEQUENTIAL_SCALE = "Blues"


def themed(fig: go.Figure, title: str | None = None) -> go.Figure:
    """Apply consistent typography and spacing to a Plotly figure.

    Defaults bumped to 13–17pt and dark text so annotations and tick
    labels remain readable against coloured bars/cells.
    """
    fig.update_layout(
        font=dict(family="Inter, system-ui, sans-serif", size=13, color=PALETTE["text"]),
        title=dict(text=title, font=dict(size=17, color=PALETTE["text"])) if title else None,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=20, t=60 if title else 30, b=40),
        hoverlabel=dict(font_family="Inter, system-ui, sans-serif", font_size=13),
        legend=dict(font=dict(size=12, color=PALETTE["text"])),
    )
    fig.update_xaxes(gridcolor="#E2E8F0", zerolinecolor="#CBD5E1",
                     tickfont=dict(size=12, color=PALETTE["text"]),
                     title_font=dict(size=13, color=PALETTE["text"]))
    fig.update_yaxes(gridcolor="#E2E8F0", zerolinecolor="#CBD5E1",
                     tickfont=dict(size=12, color=PALETTE["text"]),
                     title_font=dict(size=13, color=PALETTE["text"]))
    return fig


def annotation_style(color: str = None) -> dict:
    """Common bgcolor + bordered style for vline / hline annotations.

    Plotly's default annotation text is unstyled and tends to vanish
    against busy histogram bars. This wraps it in a small pill.
    """
    return dict(
        font=dict(size=13, color=color or PALETTE["text"]),
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor=color or PALETTE["secondary"],
        borderwidth=1, borderpad=4,
    )


def empty_chart(message: str = "No data matches current filters") -> go.Figure:
    """Placeholder figure shown when the filtered df is empty."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        showarrow=False,
        font=dict(size=14, color=PALETTE["muted"]),
        x=0.5, y=0.5, xref="paper", yref="paper",
    )
    fig.update_layout(
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=20, r=20, t=20, b=20), height=300,
    )
    return fig


def fmt_sgd(value: float | int | None) -> str:
    """Format a SGD salary value: S$X,XXX. NaN-safe."""
    if value is None:
        return "—"
    try:
        if value != value:  # NaN check
            return "—"
    except TypeError:
        return "—"
    return f"S${int(round(value)):,}"


def fmt_int(value: float | int | None) -> str:
    """Format an int with comma separators. NaN-safe."""
    if value is None:
        return "—"
    try:
        if value != value:
            return "—"
    except TypeError:
        return "—"
    return f"{int(round(value)):,}"
