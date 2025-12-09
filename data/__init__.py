import pandas as pd
import geopandas as gpd
import os

# We make relevant datasets contained in this folder available across the project, for easier integration and plotting.
# The base directory points to the folder containing this file:
base_dir = os.path.dirname(__file__)

# Make the ISO code lookup table available across the project.
lookup_table = pd.read_csv(os.path.join(base_dir, "lookup_table.csv"), index_col=0)

# Load the world shapefile
world = gpd.read_file("data/world_shapefile/ne_50m_admin_0_countries.shp")
world.to_crs("EPSG:3857", inplace=True)