import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.affinity import translate, scale
import matplotlib.pyplot as plt

st.set_page_config(page_title="Covid Cases by County — 2020", layout="wide")

DATA_URL = "https://raw.githubusercontent.com/nytimes/covid-19-data/refs/heads/master/rolling-averages/us-counties-2020.csv"
COUNTIES_URL = "https://gist.githubusercontent.com/sdwfrost/d1c73f91dd9d175998ed166eb216994a/raw/e89c35f308cee7e2e5a784e1d3afc5d449e9e4bb/counties.geojson"


@st.cache_data
def load_case_data():
    df = pd.read_csv(DATA_URL)
    df["fips"] = df["geoid"].str.replace("USA-", "", regex=False).str.zfill(5)
    totals = (
        df.groupby(["fips", "county", "state"], as_index=False)["cases"]
          .sum()
          .rename(columns={"cases": "total_cases"})
    )
    return df, totals[totals["total_cases"] > 0]


@st.cache_data
def load_geo_data():
    gdf = gpd.read_file(COUNTIES_URL)
    gdf["centroid"] = gdf.geometry.centroid
    gdf["lon"] = gdf.centroid.x
    gdf["lat"] = gdf.centroid.y
    gdf["fips"] = gdf["GEOID"].astype(str).str.zfill(5)
    return gdf


def make_inset(geom_row, xfact, yfact, target_xy):
    geom = geom_row.geometry.iloc[0]
    origin = geom.centroid.coords[0]
    geom_scaled = scale(geom, xfact=xfact, yfact=yfact, origin=origin)
    minx, miny, _, _ = geom_scaled.bounds
    dx, dy = target_xy[0] - minx, target_xy[1] - miny
    geom_final = translate(geom_scaled, xoff=dx, yoff=dy)

    def transform_xy(lons, lats):
        pts = gpd.GeoSeries.from_xy(lons, lats)
        pts = pts.scale(xfact=xfact, yfact=yfact, origin=origin)
        pts = pts.translate(xoff=dx, yoff=dy)
        return pts.x.values, pts.y.values

    return geom_final, transform_xy


def format_total(x):
    return f"{int(x // 100 * 100):,}+"


@st.cache_resource
def build_figure(label_threshold):
    df1, county_totals = load_case_data()
    gdf = load_geo_data()
    merged = county_totals.merge(gdf[["fips", "lon", "lat"]], on="fips", how="inner")

    fig, ax = plt.subplots(figsize=(15, 10))

    states_bg = gdf.dissolve(by="STATEFP")
    ak_bg = states_bg.loc[["02"]]
    hi_bg = states_bg.loc[["15"]]
    conus_bg = states_bg.drop(index=["02", "15"], errors="ignore")
    conus_bg.plot(ax=ax, color="#ececec", edgecolor="white", linewidth=0.6, zorder=0)

    ak_geom_final, ak_transform = make_inset(ak_bg, 0.35, 0.35, (-127, 24))
    hi_geom_final, hi_transform = make_inset(hi_bg, 1.0, 1.0, (-107, 23))
    gpd.GeoSeries([ak_geom_final]).plot(ax=ax, color="#ececec", edgecolor="white", linewidth=0.6, zorder=0)
    gpd.GeoSeries([hi_geom_final]).plot(ax=ax, color="#ececec", edgecolor="white", linewidth=0.6, zorder=0)

    conus_pts = merged[~merged["state"].isin(["Alaska", "Hawaii"])]
    ak_pts = merged[merged["state"] == "Alaska"]
    hi_pts = merged[merged["state"] == "Hawaii"]
    ak_lon, ak_lat = ak_transform(ak_pts["lon"], ak_pts["lat"])
    hi_lon, hi_lat = hi_transform(hi_pts["lon"], hi_pts["lat"])

    scale_factor = 3000 / merged["total_cases"].max()
    for lons, lats, cases in [
        (conus_pts["lon"], conus_pts["lat"], conus_pts["total_cases"]),
        (ak_lon, ak_lat, ak_pts["total_cases"]),
        (hi_lon, hi_lat, hi_pts["total_cases"]),
    ]:
        ax.scatter(lons, lats, s=cases * scale_factor, color="red", alpha=0.45,
                   edgecolor="darkred", linewidth=0.3, zorder=1)

    for val in [100, 10000]:
        ax.scatter([], [], s=val * scale_factor, color="red", alpha=0.45,
                   edgecolor="darkred", linewidth=0.3, label=f"{val:,}")
    ax.legend(scatterpoints=1, frameon=False, labelspacing=2.5,
              title="Total cases", loc="upper right", fontsize=9, title_fontsize=10)

    state_totals = (
        df1.groupby("state", as_index=False)["cases"].sum()
           .rename(columns={"cases": "state_total"})
    )
    state_centers = merged.groupby("state", as_index=False)[["lon", "lat"]].mean()
    state_totals = state_totals.merge(state_centers, on="state", how="inner")
    state_totals["label"] = state_totals["state_total"].apply(format_total)
    sparse_states = state_totals[state_totals["state_total"] < label_threshold]

    for _, row in sparse_states.iterrows():
        lon, lat = row["lon"], row["lat"]
        if row["state"] == "Alaska":
            lon, lat = ak_transform([lon], [lat])
            lon, lat = lon[0], lat[0]
        elif row["state"] == "Hawaii":
            lon, lat = hi_transform([lon], [lat])
            lon, lat = lon[0], lat[0]
        ax.annotate(f"{row['state']}\n{row['label']}", xy=(lon, lat),
                    fontsize=6.5, ha="center", va="center", zorder=2, color="#444444")

    ax.set_xlim(-130, -65)
    ax.set_ylim(18, 50)
    ax.set_aspect(1.3)
    ax.axis("off")
    ax.set_title("Total Number of Covid Cases by County in 2020",
                 fontsize=16, fontweight="bold", pad=15)
    return fig


st.title("Total Covid Cases by County in 2020")
st.caption("Recreated from the New York Times county-level case tracker. "
           "Data: NYT covid-19-data rolling averages.")

threshold = st.slider("State label threshold (states below this total get labeled)",
                       min_value=0, max_value=800000, value=200000, step=25000)

with st.spinner("Building map..."):
    fig = build_figure(threshold)
    st.pyplot(fig)
