"""
Schematic architecture diagram for the proposed model (ResNet34 encoder +
U-Net++ nested decoder + CBAM). This is a static topology drawing, not a
traced forward pass, so it needs only matplotlib -- no torch, dataset, or
GPU. Install with: pip install matplotlib

Not part of the original tree in your diagram, but kept as its own file
(rather than mixed into models/resunetpp_cbam.py, which should only
contain the actual nn.Module) since it has a completely different
dependency (matplotlib vs. torch) and purpose (visualization vs. model
definition).
"""

from __future__ import annotations

from pathlib import Path

_DIAG_ENCODER_COLOR = "#E0785A"
_DIAG_ENCODER_SHADOW = "#B85B41"
_DIAG_DECODER_COLOR = "#8C7FC9"
_DIAG_DECODER_SHADOW = "#6C5FA3"
_DIAG_IO_COLOR = "#9A9A9A"
_DIAG_DOWN_COLOR = "#1A1A1A"
_DIAG_UP_COLOR = "#D6431F"
_DIAG_SKIP_COLOR = "#D9A62A"
_DIAG_CBAM_BADGE = "#F2B705"


def _diagram_node_xy(i: int, j: int, col_w: float, row_h: float) -> tuple[float, float]:
    """Column = i + j, row = i -- matches the U-Net++ nested grid layout."""
    return (i + j) * col_w, -i * row_h


def save_architecture_diagram(output_path: Path, depth: int = 5) -> None:
    """
    Render a schematic PNG of the model architecture (ResNet-34 encoder +
    U-Net++ nested decoder + CBAM on the skip connections), styled after
    the WBC-Net figure: nested dense skip connections drawn as arced
    dashed lines (arc height grows with span so longer arcs don't cross
    shorter ones), a diagonal encoder backbone (solid, downsampling), and
    vertical upsampling arrows one row at a time.

    This is a static schematic, not a traced forward pass -- it doesn't
    need a model instance, dataset, checkpoint, or GPU, and the diagram
    is identical for every DatasetConfig (input size/channels don't
    change the topology). `depth` is 5 for a standard ResNet backbone
    (stem + 4 stages); it would only change if you swapped in a
    differently-staged encoder.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required to render the architecture diagram. "
            "Install it with: pip install matplotlib"
        ) from exc

    col_w, row_h = 1.9, 2.35
    node_w, node_h = 0.95, 0.62
    shadow_dx, shadow_dy = 0.10, 0.10

    def draw_block(ax, cx, cy, color, shadow, label=None):
        ax.add_patch(Rectangle(
            (cx - node_w / 2 + shadow_dx, cy - node_h / 2 - shadow_dy),
            node_w, node_h, facecolor=shadow, edgecolor="none", zorder=2,
        ))
        ax.add_patch(Rectangle(
            (cx - node_w / 2, cy - node_h / 2),
            node_w, node_h, facecolor=color, edgecolor="#2b2b2b", linewidth=0.8, zorder=3,
        ))
        if label:
            ax.text(cx, cy, label, ha="center", va="center", fontsize=7.2,
                    color="white", zorder=4, fontweight="bold")

    def draw_arrow(ax, p1, p2, color, style="-", lw=1.4, z=1):
        ax.add_patch(FancyArrowPatch(
            p1, p2, connectionstyle="arc3,rad=0", arrowstyle="-|>", mutation_scale=10,
            linewidth=lw, linestyle=style, color=color, zorder=z, shrinkA=6, shrinkB=6,
        ))

    def draw_skip_arc(ax, x1, x2, y, span):
        # span == 1 (adjacent, same-row) is the only hop CBAM is actually
        # applied to -- draw it bolder/darker gold so it reads as distinct
        # from the longer, unrefined dense-skip arcs.
        shortest = span == 1
        rad = -(0.22 + 0.11 * span)
        color = _DIAG_CBAM_BADGE if shortest else _DIAG_SKIP_COLOR
        lw = 1.9 if shortest else 1.0
        ax.add_patch(FancyArrowPatch(
            (x1 + node_w / 2, y + node_h / 2 - 0.02),
            (x2 - node_w / 2, y + node_h / 2 - 0.02),
            connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>", mutation_scale=8,
            linewidth=lw, linestyle=(0, (4, 3)), color=color,
            zorder=(2 if shortest else 1), shrinkA=2, shrinkB=2,
        ))

    fig, ax = plt.subplots(figsize=(15, 12))
    ax.set_aspect("equal")
    ax.axis("off")

    positions: dict[tuple[int, int], tuple[float, float]] = {}
    for i in range(depth):
        for j in range(depth - i):
            x, y = _diagram_node_xy(i, j, col_w, row_h)
            positions[(i, j)] = (x, y)
            if j == 0:
                draw_block(ax, x, y, _DIAG_ENCODER_COLOR, _DIAG_ENCODER_SHADOW,
                           label=f"X{i},0")
            else:
                draw_block(ax, x, y, _DIAG_DECODER_COLOR, _DIAG_DECODER_SHADOW,
                           label=f"X{i},{j}")

    # Encoder backbone (diagonal, solid black, downsampling)
    for i in range(depth - 1):
        x1, y1 = positions[(i, 0)]
        x2, y2 = positions[(i + 1, 0)]
        draw_arrow(ax, (x1, y1 - node_h / 2), (x2, y2 + node_h / 2), _DIAG_DOWN_COLOR, lw=1.8, z=2)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.32, my, f"ResNet\nstage {i + 1}", fontsize=6.3, color="#333333",
                ha="left", va="center", style="italic")

    # Upsampling (vertical, solid red), same column, one row up
    for i in range(1, depth):
        for j in range(depth - i):
            below = positions[(i, j)]
            above = positions[(i - 1, j + 1)]
            draw_arrow(ax, (below[0], below[1] + node_h / 2),
                       (above[0], above[1] - node_h / 2), _DIAG_UP_COLOR, lw=1.3, z=2)

    # Dense skip connections (dashed arcs, within each row). Draw the
    # longer (unrefined) spans first, then the shortest/CBAM ones last so
    # they sit on top where arcs overlap.
    for i in range(depth):
        cols = list(range(depth - i))
        for a in range(len(cols)):
            for b in range(a + 1, len(cols)):
                if b - a != 1:
                    x1, y1 = positions[(i, a)]
                    x2, y2 = positions[(i, b)]
                    draw_skip_arc(ax, x1, x2, y1, span=(b - a))
        for a in range(len(cols) - 1):
            x1, y1 = positions[(i, a)]
            x2, y2 = positions[(i, a + 1)]
            draw_skip_arc(ax, x1, x2, y1, span=1)

    # Input / Output framing
    x0, y0 = positions[(0, 0)]
    xN, yN = positions[(0, depth - 1)]
    in_x, in_y = x0 - col_w, y0
    out_x, out_y = xN + col_w, yN

    for cx, cy, txt in [(in_x, in_y, "Input\nImage"), (out_x, out_y, "Output\nImage")]:
        ax.add_patch(FancyBboxPatch((cx - 0.55, cy - 0.4), 1.1, 0.8,
                                     boxstyle="round,pad=0.02,rounding_size=0.06",
                                     facecolor=_DIAG_IO_COLOR, edgecolor="#333333",
                                     linewidth=0.8, zorder=3))
        ax.text(cx, cy, txt, ha="center", va="center", fontsize=8, color="white",
                fontweight="bold", zorder=4)

    draw_arrow(ax, (in_x + 0.55, in_y), (x0 - node_w / 2, y0), "#333333", lw=1.4, z=2)
    draw_arrow(ax, (xN + node_w / 2, yN), (out_x - 0.55, out_y), "#333333", lw=1.4, z=2)

    top_y = y0 + 2.6
    draw_arrow(ax, (in_x, in_y + 0.4), (in_x, top_y), "#333333", lw=1.0, z=2)
    draw_arrow(ax, (in_x, top_y), (out_x, top_y), "#333333", lw=1.0, z=2)
    draw_arrow(ax, (out_x, top_y), (out_x, out_y + 0.4), "#333333", lw=1.0, z=2)

    # Legend
    legend_x = in_x - 0.2
    legend_y = -((depth - 1) * row_h) - 1.3
    items = [
        (_DIAG_ENCODER_COLOR, "Encoder feature (ResNet-34 stage output)"),
        (_DIAG_DECODER_COLOR, "Nested decoder conv block (U-Net++)"),
    ]
    for k, (color, text) in enumerate(items):
        ly = legend_y - k * 0.5
        ax.add_patch(Rectangle((legend_x, ly - 0.15), 0.4, 0.3, facecolor=color,
                                edgecolor="#2b2b2b", linewidth=0.7))
        ax.text(legend_x + 0.55, ly, text, fontsize=8, va="center")

    line_specs = [
        (_DIAG_DOWN_COLOR, "-", "Downsampling (ResNet-34 encoder)"),
        (_DIAG_UP_COLOR, "-", "Upsampling"),
        (_DIAG_CBAM_BADGE, (0, (4, 3)), "Shortest skip connection (CBAM applied)"),
        (_DIAG_SKIP_COLOR, (0, (4, 3)), "Longer dense skip connection (no CBAM)"),
    ]
    for k, (color, style, text) in enumerate(line_specs):
        ly = legend_y - (2 + k) * 0.5
        ax.plot([legend_x, legend_x + 0.4], [ly, ly], color=color, linewidth=1.8, linestyle=style)
        ax.text(legend_x + 0.55, ly, text, fontsize=8, va="center")

    min_x, max_x = in_x - 0.8, out_x + 0.8
    min_y = legend_y - 6 * 0.5 - 0.3
    max_y = top_y + 0.5
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved architecture diagram to {output_path}")
