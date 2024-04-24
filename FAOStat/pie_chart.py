import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from dantro.plot.utils import ColorManager
from matplotlib.lines import Line2D
from matplotlib.patches import Wedge
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import math
import logging
from typing import Union
import matplotlib.colors as mcolors


log = logging.getLogger(__name__)


def _draw_wedge(_ax, start_angle, end_angle, length, bar_length, *, color):
    """ Draws a Wedge with a given start and end angle, length, bar length, and colour"""

    _ax.add_artist(
        Wedge((0, 0),
              length, start_angle, end_angle,
              color=color, width=bar_length
              )
    )


def _add_text(_ax, x, y, country, value, angle):
    """ Adds wedge text with the correct orientation"""
    if 90 <= angle % 360 < 270:
        text = "{} ({})".format(country, value)
        _ax.text(x, y, text, rotation=angle - 180, ha="right", va="center", rotation_mode="anchor")
    else:
        text = "({}) {}".format(value, country)
        _ax.text(x, y, text, rotation=angle, ha="left", va="center", rotation_mode="anchor")


def _add_flag(_ax, x, y, name, zoom, rotation, *, ds_path):
    """ Adds a flag icon"""
    flag = Image.open(f"{ds_path}/{name.lower}.png")
    flag = flag.rotate(rotation if rotation > 270 else rotation - 180)
    im = OffsetImage(flag, zoom=zoom, interpolation="lanczos", resample=True, visible=True)

    _ax.add_artist(AnnotationBbox(
        im, (x, y), frameon=False,
        xycoords="ds",
    ))

def _add_legend(_ax, labels, colors, title, **legend_kwargs):
    lines = [Line2D([], [], marker='o', markersize=24, linewidth=0, color=c) for c in colors]

    _ax.legend(
        lines, labels, title=title, **legend_kwargs
    )


def _draw_reference_line(_ax, point, size, padding, *, color = 'white'):
    """ Draws circular reference lines on the plot"""
    _draw_wedge(_ax, 0, 360, point + padding + size / 2, size, color=color)
    # _ax.text(0.0, padding + point, np.round(point, 3), va="center", rotation=1)


def _get_xy_with_padding(length, angle, padding):
    x = math.cos(math.radians(angle)) * (length + padding)
    y = math.sin(math.radians(angle)) * (length + padding)
    return x, y


def pie(
    ds: xr.Dataset,
    *,
    x: str,
    y: str,
    hue: str,
    title: str = None,
    cmap: Union[str, dict, mcolors.Colormap] = None,
    norm: Union[str, dict, mcolors.Normalize] = None,
    vmin: float = None,
    vmax: float = None,
    add_colorbar: bool = True,
    cbar_kwargs: dict = None,
    start_angle: int = 90,
    end_angle: int = None,
    pad: float = 1.2,
    inner_padding_factor: float = 2.0,
    outer_padding: float = 1.3,
    subplot_kwargs: dict = {},
    **plot_kwargs
):
    
    fig, ax = plt.subplots(**subplot_kwargs)
    
    # End angle is start + 360 by default
    if end_angle is None:
        end_angle = start_angle + 360

    # Coordinate name
    coord_name = list(ds.coords.keys())[0]

    # Inner padding of the wedges
    inner_padding = inner_padding_factor * ds[y].min().item()

    # Axis limits
    limit = (inner_padding + ds[y].max().item()) * outer_padding
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit))

    # Get vmin and vmax of the color wheel
    if vmin is None:
        vmin = ds[hue].min().item()
    if vmax is None:
        vmax = ds[hue].max().item()

    # Build the colormanager
    cm = ColorManager(
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
    )
    cm.norm.vmin = vmin
    cm.norm.vmax = vmax

    # Scale the wedge widths to ensure the fit on a circle
    n_wedges = len(ds.coords[coord_name].data)
    width_sf = (end_angle - start_angle - n_wedges * pad) / ds[x].sum().item()

    start = start_angle

    for i, country in enumerate(ds.coords[coord_name].data):

        # Length of the wedge
        bar_length = ds[y].sel({coord_name: country}).data
        length = bar_length + inner_padding

        # Start and end angle of wedge
        start = start + pad
        end = start + ds[x].sel({coord_name: country}).data * width_sf
        angle = (end + start) / 2

        # Add variables here
        flag_zoom = 0.004 * length
        flag_x, flag_y = _get_xy_with_padding(length, angle, 8 * flag_zoom)
        text_x, text_y = _get_xy_with_padding(length, angle, 16 * flag_zoom)

        # Add functions here
        _draw_wedge(ax, start, end, length, bar_length, color=cm.map_to_color(ds[hue].sel({coord_name: country}).data))
        # add_flag(ax, flag_x, flag_y, row.country, flag_zoom, angle)

        if (end - start) > 1:
            _add_text(ax, text_x, text_y, country, f'{ds[hue].sel({coord_name: country}).data:.3g}', angle)
        start = end

    _dist = ds[y].max().item() - ds[y].min().item()

    _draw_reference_line(ax, 0.25*ds[y].max().item(), 0.005 * _dist, inner_padding, color='#F8F1F1')
    _draw_reference_line(ax, 0.5*ds[y].max().item(), 0.005 * _dist, inner_padding, color='#F8F1F1')
    _draw_reference_line(ax, 0.75*ds[y].max().item(), 0.005 * _dist, inner_padding, color='#F8F1F1')

    ax.axis("off")
    ax.set_aspect('equal')
    ax.set_title(title, x=0.5, y=0.5, va="center", ha="center", linespacing=1.5)

    return plt