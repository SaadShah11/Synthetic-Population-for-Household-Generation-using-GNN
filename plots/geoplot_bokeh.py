import os
import pandas as pd
import geopandas as gpd
from bokeh.io import show, output_notebook, output_file
from bokeh.models import GeoJSONDataSource, LinearColorMapper, ColorBar, LabelSet, ColumnDataSource, HoverTool
from bokeh.palettes import Viridis256
from bokeh.plotting import figure
from bokeh.transform import linear_cmap
import json

# ── 1.  Paths & data ───────────────────────────────────────────────────
BASE = "../data/"
PERSONS_DIR = "../data/preprocessed-data/individuals"

# Load age data
age = pd.read_csv(os.path.join(PERSONS_DIR, "Age_Perfect_5yrs.csv"))
age = age.rename(columns={"geography code": "MSOA21CD"})

# Load shapefiles
msoa_fp = os.path.join(BASE, "geodata", "MSOA_2021_EW_BGC_V3.shp")
red_fp  = os.path.join(BASE, "geodata", "boundary.geojson")

gdf_msoa = gpd.read_file(msoa_fp).to_crs(3857)
red_bnd = gpd.read_file(red_fp).to_crs(3857)

# Merge population totals
gdf_msoa = gdf_msoa.merge(age[["MSOA21CD", "total"]], on="MSOA21CD", how="left")
gdf_msoa["total"] = gdf_msoa["total"].fillna(0)

# Clip to red boundary
red_union = red_bnd.unary_union
gdf_clip = gdf_msoa[gdf_msoa.intersects(red_union)].copy()

# ── 2.  Reproject to Web Mercator for Bokeh (EPSG:3857) ────────────────
gdf_clip_3857 = gdf_clip.to_crs(epsg=3857)

# Calculate centroids for label placement
gdf_clip_3857["x"] = gdf_clip_3857.geometry.centroid.x
gdf_clip_3857["y"] = gdf_clip_3857.geometry.centroid.y

# ── 3.  GeoJSONDataSource for Bokeh ────────────────────────────────────
geo_source = GeoJSONDataSource(geojson=gdf_clip_3857.to_json())

# ── 4.  Setup color mapper ─────────────────────────────────────────────
color_mapper = LinearColorMapper(palette=Viridis256, low=gdf_clip_3857["total"].min(), high=gdf_clip_3857["total"].max())

# ── 5.  Plotting ───────────────────────────────────────────────────────
p = figure(
    title="MSOAs within Red Boundary (Bokeh, Correct Projection)",
    x_axis_type="mercator",
    y_axis_type="mercator",
    toolbar_location="above",
    tools="pan,wheel_zoom,reset,save",
    width=1000,
    height=800
)
p.grid.grid_line_color = None

# ── 6.  Patches from clipped MSOAs ─────────────────────────────────────
p.patches('xs', 'ys', source=geo_source,
          fill_color={'field': 'total', 'transform': color_mapper},
          fill_alpha=0.8, line_color="white", line_width=0.5)

# ── 7.  Add static MSOA code labels ────────────────────────────────────
label_source = ColumnDataSource(data=dict(
    x=gdf_clip_3857["x"],
    y=gdf_clip_3857["y"],
    name=gdf_clip_3857["MSOA21CD"]
))

labels = LabelSet(x='x', y='y', text='name', source=label_source,
                  text_font_size="8pt", text_color="black", x_offset=0, y_offset=0)
p.add_layout(labels)

# ── 8.  Color bar and hover ────────────────────────────────────────────
color_bar = ColorBar(color_mapper=color_mapper, label_standoff=12, location=(0, 0))
p.add_layout(color_bar, 'right')

hover = HoverTool(tooltips=[("MSOA", "@MSOA21CD"), ("Total", "@total")])
p.add_tools(hover)

# ── 9.  Show plot ──────────────────────────────────────────────────────
output_file("output_areas.html")
show(p)
