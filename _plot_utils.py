import cartopy
import cartopy.crs as ccrs
import geopandas
import matplotlib.pyplot as plt
import matplotlib.colorbar as colorbar
import networkx as nx
import numpy as np
import pandas as pd
import xarray as xr

from cartopy.feature import BORDERS
from dantro.plot.utils import ColorManager
from pyproj import Transformer


# Convert the country names to an ISO3 code for easier labelling and selection
def get_iso3(lookup_table, country):
    """
    Retrieves the ISO 3166-1 Alpha-3 code (ISO3) for a given country name from a lookup table.
    If the country is not found or its ISO3 code is missing, the function returns the original
    country name.

    Parameters:
    -----------
    lookup_table : pandas.DataFrame
        The lookup table containing country information. It should have the country names
        as the index and a column named 'Alpha-3 code' that stores the corresponding ISO3 codes.

    country : str
        The name of the country for which the ISO3 code is to be retrieved.

    Returns:
    --------
    str
        The ISO3 code corresponding to the country if available.
        If the country is not found in the lookup table or if the ISO3 code is missing,
        the original country name is returned.

    Example:
    --------
    iso3_code = get_iso3('United States of America', lookup_table)
    print(iso3_code)  # Output: 'USA' or the country name if not found.
    """

    # Check if the country is in the lookup table's index
    if country not in lookup_table.index:
        return country

    # Retrieve the ISO3 code for the country
    _iso = lookup_table.loc[country, 'Alpha-3 code']

    # Return the ISO3 code if it exists, otherwise return the country name
    return country if pd.isna(_iso) else _iso


def property_from_iso3(lookup_table, iso3, item, *, correct: bool = True):
    """
    Retrieves a specific property for a country, identified by its ISO3 (Alpha-3) code,
    from a given lookup table. For country names, an optional correction is applied to
    reformat common or longer names.

    Parameters:
    -----------
    lookup_table : pandas.DataFrame
        The DataFrame containing country data. Must have a column named 'Alpha-3 code' for ISO3 codes.

    iso3 : str
        The ISO 3166-1 Alpha-3 code (3-letter country code) identifying the country.

    item : str
        The column name in the lookup_table whose value is to be returned for the specified ISO3 country.
        If 'Name', the country's name may be corrected based on common reformats.

    correct : bool, optional
        If True and item is 'Name', the function will return a corrected/shortened version of the
        country name (default is True).

    Returns:
    --------
    value : Any
        The value corresponding to the given item (property) for the country specified by iso3.
        If the ISO3 code is not found, returns np.nan.
        If item is 'Name', returns the corrected/shortened country name if `correct` is True.

    Corrections applied to country names when `correct=True`:
    ---------------------------------------------------------
    - 'Viet Nam' -> 'Vietnam'
    - 'China, mainland' -> 'China'
    - 'United States of America' -> 'US'
    - 'Russian Federation' -> 'Russia'
    - 'Democratic Republic of the Congo' -> 'DR Congo'
    - 'Iran (Islamic Republic of)' -> 'Iran'
    - 'United Republic of Tanzania' -> 'Tanzania'
    - 'United Kingdom of Great Britain and Northern Ireland' -> 'UK'
    - 'China, Taiwan Province of' -> 'Taiwan'
    - 'Syrian Arab Republic' -> 'Syria'
    - 'Netherlands (Kingdom of the)' -> 'Netherlands'
    - 'Republic of Korea' -> 'South Korea'

    Example:
    --------
    result = property_from_iso3(df, 'USA', 'GDP')
    name = property_from_iso3(df, 'VNM', 'Name', correct=True)
    """

    # Lookup the row based on the ISO3 code
    row = lookup_table[lookup_table['Alpha-3 code'] == iso3]

    # If no row is found, return np.nan
    if row.empty:
        return np.nan

    # Handle case where the requested item is "Name"
    if item == "Name":
        country_name = row.index.values
        if len(country_name) > 0:
            country_name = country_name[0]
        else:
            return iso3

        # Only apply corrections if the flag is set to True
        if correct:
            name_corrections = {
                "Viet Nam": "Vietnam",
                "China, mainland": "China",
                "United States of America": "US",
                "Russian Federation": "Russia",
                "Democratic Republic of the Congo": "DR Congo",
                "Iran (Islamic Republic of)": "Iran",
                "United Republic of Tanzania": "Tanzania",
                "United Kingdom of Great Britain and Northern Ireland": "UK",
                "China, Taiwan Province of": "Taiwan",
                "Syrian Arab Republic": "Syria",
                "Netherlands (Kingdom of the)": "Netherlands",
                "Republic of Korea": "South Korea"
            }
            return name_corrections.get(country_name, country_name)
        return country_name

    # Handle all other cases (non-"Name" columns)
    value = row[item].values
    return value[0] if len(value) > 0 else np.nan


def plot_background_map(_f, _ax, *,
                        extent=None,
                        borders_lw: float = 0.1,
                        add_sea: bool = True,
                        landcolor: str = 'grey',
                        seacolor: str = 'white'
                        ) -> None:
    """
    Plots a simple background map with land, ocean (optional), and country borders.

    This function adds a landmass and, optionally, a sea to the specified axis using Cartopy's NaturalEarth features.
    The colors of both can be specified. It also includes country borders with adjustable line width.
    The map extent can be customized or will default to global coverage.

    Parameters:
    -----------
    _f : matplotlib.figure.Figure
        The figure object on which the map is drawn. Used to set the background color.

    _ax : cartopy.mpl.geoaxes.GeoAxesSubplot
        The GeoAxesSubplot on which the map features (land, ocean, borders) are plotted.

    extent : list of float, optional
        Specifies the geographical extent of the map as [west, east, south, north].
        If None, the default global extent of [-180, 180, -60, 73] is used.

    borders_lw : float, optional
        The line width for the country borders. Default is 0.1.

    add_sea : bool, optional
        Whether to add the ocean feature to the map. Default is True.

    landcolor : str, optional
        Facecolor of the land part. Default is grey.

    seacolor : str, optional
        Facecolor of the sea. Default is white.

    Features added:
    ---------------
    - Light grey landmass with slight transparency.
    - Optional light blue ocean background with slight transparency.
    - Country borders with customizable line width.

    Notes:
    ------
    - The axis is set to 'off' to remove default axis lines and labels.
    - If no extent is provided, the map will cover most of the Earth, from 60°S to 73°N.

    Example:
    --------
    fig, ax = plt.subplots(subplot_kw={'projection': cartopy.crs.PlateCarree()})
    plot_background_map(fig, ax, extent=[-100, 100, -50, 50], borders_lw=0.2, add_sea=True)
    plt.show()
    """

    # Add land
    _land = cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                                facecolor=landcolor, alpha=0.6)
    _ax.axis('off')
    _ax.add_feature(_land, zorder=-2, lw=0)

    # Add sea, if specified
    if add_sea:
        _sea = cartopy.feature.NaturalEarthFeature('physical', 'ocean', '50m',
                                                   facecolor=seacolor,
                                                   alpha=0.2)
        _ax.add_feature(_sea, zorder=-3, lw=0)

    # Add borders
    _ax.add_feature(BORDERS, lw=borders_lw, zorder=-1)

    # Set extent
    _ax.set_extent(extent if extent is not None else [-180, 180, -60, 73])
    _f.patch.set(fc='white', lw=0)


def plot_world_network(data: xr.DataArray, lookup_table: pd.DataFrame,
                       *,
                       fig=None,
                       ax=None,
                       figsize: tuple = (10, 8),
                       special_countries: list = None,
                       extent: list = None,
                       width_factor: float = 1.0,
                       special_edge_width_factor: float = 5e-5,
                       size_factor: float = 1.0,
                       arrow_dict: dict = None,
                       min_edge_weight: float = 0.0,
                       borders_lw: float = 0.1,
                       add_sea: bool = True,
                       landcolor: str = 'grey',
                       seacolor: str = 'white',
                       exporter_color: str = 'darkblue',
                       importer_color: str = 'red'):
    """
    Plots a network of trade flows or interactions between countries on a world map, using exporting and importing nodes
    connected by edges representing flows. Node size and edge width can be scaled by trade volume or other metrics.

    Parameters:
    -----------
    data : xarray.DataArray
        The data array containing trade or interaction data between countries. It should have dimensions 'Source' and
        'Destination' to represent trade between countries.

    lookup_table : pandas.DataFrame
        A DataFrame with country metadata. It must contain columns 'Alpha-3 code', 'Latitude', and 'Longitude'.

    fig : matplotlib.figure.Figure, optional
        The figure to plot on. If None, a new figure is created.

    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, a new set of axes with a map projection is created.

    figsize : tuple, optional
        The size of the figure. Default is (10, 8).

    special_countries : list of tuple, optional
        A list of (Source, Destination) pairs to highlight with special edges. These can represent key trade routes or
        interactions.

    extent : list, optional
        The geographical extent of the map as [west, east, south, north]. Default is global coverage.

    width_factor : float, optional
        A scaling factor for the width of all edges. Default is 1.0.

    special_edge_width_factor : float, optional
        A scaling factor for the width of special edges. Default is 5e-5.

    size_factor : float, optional
        A scaling factor for the size of nodes. Default is 1.0.

    arrow_dict : dict, optional
        Dictionary to customize the appearance of arrows for edges. Default arrow settings can be overridden here.

    min_edge_weight : float, optional
        The minimum threshold for edge weights. Edges with weights below this threshold will not be plotted. Default is 0.0.

    borders_lw : float, optional
        The line width for drawing country borders. Default is 0.1.

    add_sea : bool, optional
        Whether to add an ocean background to the map. Default is True.

    landcolor : str, optional
        Facecolor of the land part. Default is grey.

    seacolor : str, optional
        Facecolor of the sea. Default is white.

    exporter_color : str, optional
        Color of exporting nodes. Default is 'darkblue'

    importer_color : str, optional
        Color of importing nodes. Default is 'red'

    Returns:
    --------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        The figure and axes objects containing the plotted map and network.

    Example:
    --------
    fig, ax = plot_world_network(data, lookup_table, figsize=(12, 10), width_factor=2.0, special_countries=[('USA', 'CHN')])
    plt.show()
    """

    # Initialize the special countries list and arrow_dict if they are not provided
    special_countries = special_countries or []
    arrow_dict = arrow_dict or {}

    # Set up a coordinate transformer for map plotting
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857")

    # Create an empty directed network graph
    nw = nx.DiGraph()

    # Add exporters to the network
    export_nodes = data.sum("Destination", skipna=True)
    for country in export_nodes.coords['Source'].data:
        lat, lon = property_from_iso3(lookup_table, country, 'Latitude'), property_from_iso3(lookup_table, country,
                                                                                             'Longitude')
        if country == 'Other' or np.isnan(lat):
            print(f"Warning: Missing coordinates for country '{country}'")
            continue
        export_value = 2e-3 * export_nodes.sel({"Source": country}).data.item()
        nw.add_node(country, pos=(lat, lon), type='Exporter', export_value=export_value, import_value=0)

    # Add importers to the network
    import_nodes = data.sum("Source", skipna=True)
    for country in import_nodes.coords['Destination'].data:
        lat, lon = property_from_iso3(lookup_table, country, 'Latitude'), property_from_iso3(lookup_table, country,
                                                                                             'Longitude')
        if country == 'Other' or np.isnan(lat):
            continue
        import_value = 2e-3 * import_nodes.sel({"Destination": country}).data.item()
        if country in nw.nodes():
            nw.nodes[country]["import_value"] = import_value
        else:
            nw.add_node(country, pos=(lat, lon), type='Importer', import_value=import_value, export_value=0)

    # Add edges between countries
    for country_A in data.coords["Source"].data:
        if country_A == "Other":
            continue
        for country_B in data.coords["Destination"].data:
            if country_B == "Other":
                continue

            if (country_A, country_B) in special_countries:
                continue

            T = data.sel({"Source": country_A, "Destination": country_B}).data.item()
            if T > 0:
                lat_A, lon_A = property_from_iso3(lookup_table, country_A, 'Latitude'), property_from_iso3(lookup_table,
                                                                                                           country_A,
                                                                                                           'Longitude')
                lat_B, lon_B = property_from_iso3(lookup_table, country_B, 'Latitude'), property_from_iso3(lookup_table,
                                                                                                           country_B,
                                                                                                           'Longitude')
                if np.isnan(lat_A) or np.isnan(lat_B):
                    print(f"Warning: Missing edge data for {country_A} -> {country_B}")
                    continue
                nw.add_edge(country_A, country_B, weight=5e-5 * T)

    # Initialize the figure and axis if not provided
    if fig is None:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': ccrs.Mercator()})

    # Plot the map background
    plot_background_map(fig, ax, extent=extent, borders_lw=borders_lw, add_sea=add_sea, landcolor=landcolor,
                        seacolor=seacolor)

    # Transform coordinates from lat/lon to the Mercator projection
    pos = {k: transformer.transform(*v) for k, v in nx.get_node_attributes(nw, 'pos').items()}

    # Plot exporters and importers with different colors
    exporter_sizes = [size_factor * nw.nodes[n]['export_value'] for n in nw.nodes]
    importer_sizes = [size_factor * nw.nodes[n]['import_value'] for n in nw.nodes]

    nx.draw_networkx_nodes(nw, pos, node_size=exporter_sizes, ax=ax, node_color=exporter_color, alpha=0.5,
                           linewidths=0)
    nx.draw_networkx_nodes(nw, pos, node_size=importer_sizes, ax=ax, node_color=importer_color, alpha=0.5,
                           linewidths=0)

    # Plot special country edges
    arrow_kwargs = {'alpha': 0.7, 'connectionstyle': 'arc3,rad=0.2', 'arrowsize': 10}
    arrow_kwargs.update(arrow_dict)

    for country_1, country_2 in special_countries:
        edge_weight = special_edge_width_factor * data.sel({"Source": country_1, "Destination": country_2}).data.item()
        nx.draw_networkx_edges(nw, pos, edgelist=[(country_1, country_2)], width=edge_weight, ax=ax, **arrow_kwargs)

    # Plot regular edges
    regular_edges = [(u, v) for u, v in nw.edges if (u, v) not in special_countries]
    edge_weights = [max(min_edge_weight, width_factor * w) for w in nx.get_edge_attributes(nw, 'weight').values()]
    nx.draw_networkx_edges(nw, pos, edgelist=regular_edges, width=edge_weights, ax=ax, **arrow_kwargs)

    return fig, ax


def get_relative_diff(predictions: xr.DataArray, year1: int, year2: int, source: str) -> xr.DataArray:
    """
    Calculates the percentage change of a trade dataset for a specified source country between two years.

    Parameters:
    -----------
    predictions : xr.DataArray
        The predictions

    year1, year2 : int
        Start and end years of comparison

    source : str
        The ISO 3166-1 Alpha-3 code of the source country for which to calculate the change

    Returns:
    --------
    xr.DataArray
        An xarray DataArray containing the normalized differences for each destination,
        sorted in ascending order of the differences.
    """
    # Calculate the percentage change in trade
    _diff = predictions.sel({"Source": source, "Year": [year1, year2]}).diff(
        "Year").squeeze() / predictions.sel({"Source": source, "Year": year1}).data

    # Drop NaN values and sort by destination
    _diff = _diff.dropna(dim="Destination")
    _diff = _diff.isel({"Destination": _diff.argsort().data})

    return _diff


def get_diff(predictions: xr.DataArray, year1: int, year2: int, source: str) -> xr.DataArray:
    """
    Calculates the absolute change of a datapoint a specified source country between two years

    Parameters:
    -----------
    predictions : xr.DataArray
        The predictions

    year1, year2 : int
        Start and end years of comparison

    country_code : str
        The ISO 3166-1 Alpha-3 code of the country for which to calculate utility differences.

    Returns:
    --------
    xr.DataArray
        An xarray DataArray containing the change for the specified country
    """

    return predictions.sel({"Source": source, "Year": [year1, year2]}).diff("Year")


def plot_trade_utility_comp_map(data, *, year1: int, year2: int, source: str, world: geopandas.GeoDataFrame,
                           colors: dict, figsize: tuple = None):
    """
    Plots a comparative map of trade and utility changes for a specified source country between two years.
    The function creates two subplots showing changes in trade and utility with color gradients
    representing the magnitude of change. The specified country is highlighted.

    Parameters:
    -----------
    data : xr.Dataset
        dataset containing the trade value and utility predictions

    year1 : int
        Start year of comparison

    year2 : int
        End year of comparison

    source : str
        The ISO 3166-1 Alpha-3 code of the source country to highlight on the map.

    world : GeoDataFrame
        world object to use for the background

    colors : dict
        dictionary of colors to use for plotting

    figsize : tuple, optional
        figure size.

    Returns:
    --------
    fig, axs : matplotlib.figure.Figure, matplotlib.axes.Axes
        The figure and axes objects containing the plotted maps for trade and utility changes.

    Example:
    --------
    fig, axs = trade_utility_comp_map('UKR')
    plt.show()
    """

    # Default green-yellow-red colormap
    cm_GrYeRe = ColorManager(
        cmap={'continuous': True,
              'from_values': {0: colors['c_lightgreen'], 0.5: colors['c_yellow'], 1: colors['c_red']}},
        vmin=0,
        vmax=1,
    )

    # Set up figure and axes
    fig, axs = plt.subplots(figsize=figsize, subplot_kw=dict(projection=ccrs.Mercator()), ncols=2)

    # Fetch trade and utility differences
    _trade_diff = get_relative_diff(data.sel({"variable": "T_pred"})["mean"], year1, year2, source)
    _util_diff = get_diff(data.sel({"variable": "C"})["mean"], year1, year2, source)

    # Define color managers for trade and utility changes
    vmin_trade, vmax_trade = _trade_diff.min().item(), _trade_diff.max().item()
    _cm_pos = ColorManager(
        cmap={'continuous': True, 'from_values': {0: colors['c_yellow'], 1: colors['c_lightgreen']}},
        vmin=0, vmax=vmax_trade
    )
    _cm_neg = ColorManager(
        cmap={'continuous': True, 'from_values': {0: colors['c_red'], 1: colors['c_yellow']}},
        vmin=vmin_trade, vmax=0
    )

    # Plot the background and source country
    for ax in axs:
        plot_background_map(fig, ax, add_sea=False, borders_lw=0.1)
        world[world['ISO_A3_EH'] == source].plot(
            color=colors['c_darkgrey'], ax=ax, zorder=0, lw=0.1, edgecolor=colors['c_lightgrey'], hatch='//////'
        )

    # Plot trade differences
    for country in _trade_diff.coords["Destination"].data:
        if country == source:
            continue
        if country in world['ISO_A3_EH'].values:
            _data = _trade_diff.sel({"Destination": country}).data
            _color = _cm_neg.map_to_color(_data) if _data < 0 else _cm_pos.map_to_color(_data)
            world[world['ISO_A3_EH'] == country].plot(
                color=_color, ax=axs[0], zorder=0, lw=0.1, edgecolor=_color
            )
        else:
            print(f"Warning: Country '{country}' is missing from the world dataset.")

    # Plot utility differences
    vmin_util, vmax_util = _util_diff.min().item(), _util_diff.max().item()
    _cm_util = ColorManager(
        cmap={'continuous': True, 'from_values': {0: colors['c_lightgreen'],
                                                  (0 - vmin_util) / (vmax_util - vmin_util): colors['c_yellow'],
                                                  1: colors['c_red']}},
        vmin=vmin_util, vmax=vmax_util
    )
    for country in _util_diff.coords["Destination"].data:
        if country == source:
            continue
        if country in world['ISO_A3_EH'].values:
            _color = _cm_util.map_to_color(_util_diff.sel({"Destination": country}).data)
            world[world['ISO_A3_EH'] == country].plot(
                color=_color, ax=axs[1], zorder=0, lw=0.1, edgecolor=_color
            )
        else:
            print(f"Warning: Country '{country}' is missing from the world dataset.")

    # Add colorbars
    for idx, label in enumerate(['Trade Change', 'Utility Change']):
        _x0 = 0.02 if idx == 0 else 0.52
        _cax = fig.add_axes([_x0, 0.28, 0.01, 0.2])
        _cbar = colorbar.Colorbar(
            _cax, cmap=cm_GrYeRe.cmap.reversed() if idx == 0 else cm_GrYeRe.cmap, orientation='vertical',
            location='right',
            drawedges=False
        )
        if idx == 0:
            _cbar.set_ticks(ticks=np.linspace(0, 1, 5),
                            labels=["-100%", "-50%", "0", "+1{n:.0f}%".format(n=100 * vmax_trade / 2),
                                    "+1{n:.0f}%".format(n=100 * vmax_trade)])
        else:
            _cbar.set_ticks(ticks=[0, 0.5, 1],
                            labels=[-1, 0, "+1"])
        _cbar.outline.set_linewidth(0.2)
        _cax.tick_params(width=0.2)

    return fig, axs


# Compare the drop in trade volume and utility
def plot_trade_utility_comp(data, lookup_table, *, year1: int, year2: int, source: str, destinations: list,
                        colors: dict, figsize: tuple = None):
    """
        Compare the percentage change in trade volume and the change in utility between two years for multiple destination countries.

        This function generates a comparative bar plot for each destination country. For each country, the function displays
        the relative change in trade volume between the specified years (year1 and year2) and the change in utility. The bars
        are color-coded to represent the magnitude and direction of the changes, and labels are added to display exact values.

        Parameters:
        -----------
        data : xr.DataArray
            The xarray DataArray containing trade and utility predictions.
        lookup_table : pd.DataFrame
            A pandas DataFrame containing country lookup information, used to get country names.
        year1 : int
            The starting year for the comparison.
        year2 : int
            The ending year for the comparison.
        source : str
            The ISO 3166-1 Alpha-3 code of the source country for trade and utility data.
        destinations : list
            A list of destination country ISO 3166-1 Alpha-3 codes to compare.
        colors : dict
            A dictionary of colors for color-coding trade and utility differences.
        figsize : tuple, optional
            A tuple specifying the size of the figure. If not provided, defaults will be used.

        Returns:
        --------
        fig : matplotlib.figure.Figure
            The generated figure containing subplots for each destination country.
        axs : np.ndarray
            Array of axes for the subplots, one per destination country.

        Example:
        --------
        fig, axs = trade_utility_comp(data, lookup_table, year1=2021, year2=2022, source='USA',
                                      destinations=['CHN', 'IND'], colors=color_scheme)
    """

    # Setup figure
    fig, axs = plt.subplots(ncols=len(destinations), figsize=figsize, sharey=True)

    # Fetch trade and utility differences
    _trade_diff = get_relative_diff(data.sel({"variable": "T_pred"})["mean"], year1, year2, source)
    _util_diff = get_diff(data.sel({"variable": "C"})["mean"], year1, year2, source)

    # Set up colormap
    _cm = ColorManager(
        cmap={'continuous': True,
              'from_values': {0: colors['c_red'], 0.5: colors['c_yellow'], 1: colors['c_lightgreen']}},
        vmin=-1,
        vmax=+1,
    )

    # Plot two bars for each country: trade drop and utility drop
    for idx, country in enumerate(destinations):

        # Plot change in trade
        _T = _trade_diff.sel({"Destination": country}).data
        _rect = axs[idx].bar([0], [np.sign(_T) * max(0.01, abs(_T))], color=_cm.map_to_color(_T), lw=0)

        # Add text to bar
        _patch = _rect.get_children()[0]
        _x, _y = 0, _patch._height
        axs[idx].text(_x, np.sign(_y) * 1.1 * max(0.01, abs(_y) + 0.01), "{t:.0f}%".format(t=100 * _T), ha='center',
                      va='top' if _y < 0 else 'bottom')

        # Plot change in utility
        _C = _util_diff.sel({"Destination": country}).squeeze().data

        # Add text
        _rect = axs[idx].bar([1], [3 * np.sign(_C) * max(0.01, abs(_C))], color=_cm.map_to_color(-_C), lw=0)
        _patch = _rect.get_children()[0]
        _x, _y = 1, _patch._height
        _text = axs[idx].text(_x, np.sign(_y) * 1.1 * max(0.01, abs(_y) + 0.05), "{t:.2f}".format(t=_C), ha='center',
                              va='top' if _y < 0 else 'bottom')

        # Add the name of the country
        axs[idx].text(0.75, min(-0.2 + _text._y, -0.5) if _y < 0 else -0.1, property_from_iso3(lookup_table, country,
                                                                                               item='Name', correct=True),
                      rotation=90, va='top')

        axs[idx].axis('off')

    return fig, axs