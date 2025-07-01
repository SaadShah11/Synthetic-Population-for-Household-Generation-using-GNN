import os, json
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go

# ── 1.  Paths & data ───────────────────────────────────────────────────
BASE = "../data/"
PERSONS_DIR = "../data/preprocessed-data/individuals"

# Load age data and rename the geography code
age = pd.read_csv(os.path.join(PERSONS_DIR, "Age_Perfect_5yrs.csv"))
age = age.rename(columns={"geography code": "MSOA21CD"})

# Load shapefiles
msoa_fp = os.path.join(BASE, "geodata", "MSOA_2021_EW_BGC_V3.shp")
red_fp  = os.path.join(BASE, "geodata", "boundary.geojson")

# ── 2.  Read in spatial data ───────────────────────────────────────────
gdf_msoa = gpd.read_file(msoa_fp).to_crs(4326)
red_bnd = gpd.read_file(red_fp).to_crs(4326)

# ── 3.  Merge population totals ────────────────────────────────────────
gdf_msoa = gdf_msoa.merge(age[["MSOA21CD", "total"]], on="MSOA21CD", how="left")
gdf_msoa["total"] = gdf_msoa["total"].fillna(0)

# ── 3.5. Manually remove unwanted MSOAs ────────────────────────────────
exclude_codes = [
    "E02005939", "E02005979", "E02005963", "E02005959"
]
gdf_msoa = gdf_msoa[~gdf_msoa["MSOA21CD"].isin(exclude_codes)]

# ── 4.  Clip to red boundary ───────────────────────────────────────────
red_union = red_bnd.unary_union
gdf_clip = gdf_msoa[gdf_msoa.intersects(red_union)].copy()

# ── 5.  Compute accurate centroids for labels ──────────────────────────
proj_crs = 27700
gdf_proj = gdf_clip.to_crs(proj_crs)
centroids = gdf_proj.geometry.centroid.to_crs(4326)
gdf_clip["lon"] = centroids.x
gdf_clip["lat"] = centroids.y

# ── 6.  Create choropleth plot ─────────────────────────────────────────
fig = px.choropleth(
    gdf_clip,
    geojson=json.loads(gdf_clip.to_json()),
    locations="MSOA21CD",
    featureidkey="properties.MSOA21CD",
    color="total",
    color_continuous_scale="tealrose",
    projection="mercator",
    title="MSOA Choropleth (Inside Red Boundary Only) with Static Labels"
)

# ── 7.  Add static MSOA21CD labels ─────────────────────────────────────
fig.add_trace(
    go.Scattergeo(
        lon=gdf_clip["lon"],
        lat=gdf_clip["lat"],
        mode="text",
        text=gdf_clip["MSOA21CD"],
        textfont=dict(size=15, color="black"),
        hoverinfo="none",
        showlegend=False
    )
)

# ── 8.  Add red boundary outline ───────────────────────────────────────
for poly in red_bnd.geometry.explode(index_parts=False):
    fig.add_trace(
        go.Scattergeo(
            lon=list(poly.exterior.coords.xy[0]),
            lat=list(poly.exterior.coords.xy[1]),
            mode='lines',
            line=dict(color='black', width=3),
            showlegend=False
        )
    )

# ── 9.  Manually set wide view area (landscape, no fitbounds) ──────────

# Get bounds
lat_min, lat_max = gdf_clip["lat"].min(), gdf_clip["lat"].max()
lon_min, lon_max = gdf_clip["lon"].min(), gdf_clip["lon"].max()

# Define padding
lat_padding = (lat_max - lat_min) * 0.2   # 20% vertical padding
lon_padding = (lon_max - lon_min) * 0.4   # 40% horizontal padding

# Apply padded view ranges
fig.update_geos(
    visible=False,
    lataxis_range=[lat_min - lat_padding, lat_max + lat_padding],
    lonaxis_range=[lon_min - lon_padding, lon_max + lon_padding]
)

# Set layout size (wide aspect)
fig.update_layout(
    width=1600,
    height=900,
    margin=dict(l=0, r=0, t=60, b=0),
    legend_title_text="Population Range",
    title=dict(
        text="MSOA Choropleth (Inside Red Boundary Only) with Discrete Population Bins",
        x=0.5,
        xanchor='center'
    )
)

# ── 10.  Show the plot ─────────────────────────────────────────────────

fig.show()
