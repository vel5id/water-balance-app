from __future__ import annotations
import os
import numpy as np
import pandas as pd
import streamlit as st
import rasterio
from typing import Callable, Optional
try:
    from wbm.i18n import Translator, DEFAULT_LANG
except Exception:  # minimal fallback
    class Translator:  # type: ignore
        def __init__(self, lang: str = 'ru'): self.lang = lang
        def __call__(self, key: str, **fmt):
            return key if not fmt else key
    DEFAULT_LANG = 'ru'

__all__ = ["render_map"]

def render_map(scenario_df: pd.DataFrame, vol_to_elev, DEM_PATH: str, NDWI_MASK_PATH: str, areas, alpha_default: float = 0.5, tr: Optional[Callable[[str], str]] = None):
    if tr is None:
        # reconstruct translator from session if possible
        lang = getattr(st.session_state, 'lang', DEFAULT_LANG)
        try:
            tr = Translator(lang)
        except Exception:
            tr = lambda k, **_: k  # type: ignore
    st.subheader(tr("map_title"))
    if not os.path.exists(DEM_PATH):
        st.info(tr("bathymetry_missing"))
        return
    if vol_to_elev is None:
        st.info(tr("vol_elev_missing"))
        return
    if "_dem_cache" not in st.session_state:
        with rasterio.open(DEM_PATH) as ds:
            dem = ds.read(1).astype("float32"); nodata = ds.nodata; bounds = ds.bounds; crs = str(ds.crs)
        st.session_state._dem_cache = {"dem": dem, "nodata": nodata, "bounds": bounds, "crs": crs}
    dem = st.session_state._dem_cache["dem"]; nodata = st.session_state._dem_cache["nodata"]
    vis_col1, vis_col2 = st.columns([2,1])
    with vis_col2:
        alpha = st.slider(tr("water_overlay_opacity"), 0.1, 0.9, alpha_default, 0.05, key="map_alpha")
        idx = st.slider(tr("visualization_day"), 0, len(scenario_df)-1, value=len(scenario_df)-1, key="map_day")
        dem_view = st.selectbox(tr("base_background"), ["Hillshade","Grayscale (low-dark)","Grayscale (low-bright)","Terrain colors","Bathymetric","Flat","None"], index=0, key="map_view")
        mask_mode = st.selectbox(tr("water_mask_mode"), [tr("simulated_level"),tr("depth_ndwi")], index=0, key="map_mask")
        if mask_mode == tr("simulated_level"):
            st.caption(tr("default_sim_level_mode"))
    v_sel = float(scenario_df["volume_mcm"].iloc[idx]) if not scenario_df.empty else float("nan")
    try:
        z_level = float(vol_to_elev(v_sel)) if vol_to_elev is not None else None
    except Exception:
        z_level = None
    dem_disp = dem.copy()
    if nodata is not None:
        dem_disp = np.where(dem_disp == nodata, np.nan, dem_disp)
    vmin = np.nanpercentile(dem_disp, 2); vmax = np.nanpercentile(dem_disp, 98)
    norm = (dem_disp - vmin)/max(1e-6,(vmax - vmin)); norm = np.clip(norm,0,1)
    color_vis = None; vis = None
    if dem_view == "Hillshade":
        z = np.nan_to_num(dem_disp, nan=float(np.nanmedian(dem_disp)))
        gx, gy = np.gradient(z); slope = np.pi/2 - np.arctan(np.hypot(gx, gy)); aspect = np.arctan2(-gx, gy)
        az = np.radians(315.0); alt = np.radians(45.0)
        hs = np.sin(alt)*np.sin(slope) + np.cos(alt)*np.cos(slope)*np.cos(az - aspect)
        hs = (hs - hs.min())/(hs.max()-hs.min()+1e-6); vis = hs
    elif dem_view == "Grayscale (low-bright)":
        vis = 1.0 - norm
    elif dem_view == "Grayscale (low-dark)":
        vis = norm
    elif dem_view == "Terrain colors":
        g = norm; ctrl=[(0.0,(20,70,20)),(0.25,(50,120,50)),(0.5,(150,110,60)),(0.75,(200,170,120)),(1.0,(245,245,240))]
        cp_vals = np.array([c[0] for c in ctrl]); cp_cols = np.array([c[1] for c in ctrl], dtype=float)
        def _interp_cols(x):
            x = np.clip(x,0,1); idx = np.searchsorted(cp_vals,x,side='right')-1; idx = np.clip(idx,0,len(cp_vals)-2)
            left_v = cp_vals[idx]; right_v = cp_vals[idx+1]; w = np.where((right_v-left_v)>0,(x-left_v)/(right_v-left_v),0)
            left_c = cp_cols[idx]; right_c = cp_cols[idx+1]; return (left_c*(1-w)[...,None] + right_c*w[...,None]) / 255.0
        color_vis = _interp_cols(g)
    elif dem_view == "Bathymetric":
        z = dem_disp.copy(); land = z.copy(); land_min = np.nanmin(land); land_max = np.nanmax(land)
        land_norm = np.zeros_like(land) if land_max - land_min < 1e-6 else (land - land_min)/(land_max - land_min)
        depth_mask = (z < 0) & ~np.isnan(z); land_mask = (z >= 0) & ~np.isnan(z)
        color_vis = np.zeros(z.shape + (3,), dtype=float)
        if depth_mask.any():
            depths = z[depth_mask]; dmin = depths.min(); dmax = depths.max(); span = max(1e-6,dmax-dmin); dnorm = (depths - dmin)/span
            def blend(c1,c2,w): return c1*(1-w)+c2*w
            mid_color=np.array([0,90,160]); deep_color=np.array([0,25,90]); shallow_color=np.array([120,200,255])
            w_mid = np.clip(dnorm*1.4,0,1); col_mid = blend(deep_color, mid_color, w_mid[:,None]); w_shal = np.clip((dnorm-0.5)*2,0,1)
            col_depth = blend(col_mid, shallow_color, w_shal[:,None]) / 255.0; color_vis[depth_mask] = col_depth
        if land_mask.any():
            ln = land_norm[land_mask]; c1=np.array([40,110,40]); c2=np.array([160,120,60]); c3=np.array([240,235,225])
            mid = np.clip(ln*1.6,0,1); col_land = c1*(1-mid)[:,None] + c2*mid[:,None]; w2 = np.clip((ln-0.5)*2,0,1)
            col_land = col_land*(1-w2)[:,None] + c3*w2[:,None]; color_vis[land_mask] = col_land/255.0
    elif dem_view == "Flat":
        color_vis = np.ones(dem_disp.shape + (3,), dtype=float) * 0.92
    elif dem_view == "None":
        color_vis = np.zeros(dem_disp.shape + (3,), dtype=float)
    else:
        vis = norm
    if vis is not None:
        base_rgb = (vis[...,None]*255).astype(np.uint8); base_rgb = np.repeat(base_rgb,3,axis=2)
    else:
        base_rgb = (np.clip(color_vis,0,1)*255).astype(np.uint8)
    # Determine water mask mode
    if mask_mode == tr("depth_ndwi"):
        if "_ndwi_cache" not in st.session_state:
            if os.path.exists(NDWI_MASK_PATH):
                with rasterio.open(NDWI_MASK_PATH) as ms: ndwi = ms.read(1)
                st.session_state._ndwi_cache = (ndwi > 0)
            else:
                st.session_state._ndwi_cache = None
        ndwi_mask = st.session_state._ndwi_cache
        if ndwi_mask is None or ndwi_mask.shape != dem_disp.shape:
            st.warning(tr("ndwi_mask_missing"))
            water_mask = (dem_disp < 0)
            if nodata is not None: water_mask = water_mask & ~np.isnan(dem_disp)
        else:
            water_mask = (dem_disp < 0) & ndwi_mask
            if nodata is not None: water_mask = water_mask & ~np.isnan(dem_disp)
    else:
        dem_min = float(np.nanmin(dem_disp)); dem_max = float(np.nanmax(dem_disp))
        is_elevation = (dem_min >= 0) or (dem_max > 5)
        if z_level is not None and is_elevation:
            # Direct simulated level fill (elevation DEM)
            water_mask = (dem_disp <= z_level)
            if nodata is not None:
                water_mask = water_mask & ~np.isnan(dem_disp)
        else:
            # Dynamic depth fraction mode for bathymetric (negative) DEM
            try:
                target_area_km2 = float(scenario_df["area_km2"].iloc[idx])
                max_area_curve = float(areas.max()) if hasattr(areas,'max') else target_area_km2
                frac = 0.0 if max_area_curve <= 0 else float(np.clip(target_area_km2 / max_area_curve,0,1))
            except Exception:
                frac = 1.0
            if '_depth_values' not in st.session_state or '_depth_shape' not in st.session_state or st.session_state['_depth_shape'] != dem_disp.shape:
                depth_mask_template = (dem_disp < 0) & ~np.isnan(dem_disp)
                depth_values = dem_disp[depth_mask_template]
                st.session_state._depth_values = depth_values
                st.session_state._depth_mask_template = depth_mask_template
                st.session_state._depth_shape = dem_disp.shape
            depth_values = st.session_state._depth_values; depth_mask_template = st.session_state._depth_mask_template
            if depth_values.size == 0:
                water_mask = (dem_disp < 0)
                if nodata is not None:
                    water_mask = water_mask & ~np.isnan(dem_disp)
            else:
                f = float(np.clip(frac,0,1))
                if f <= 0:
                    water_mask = np.zeros_like(dem_disp,dtype=bool)
                elif f >= 0.9999:
                    water_mask = depth_mask_template
                else:
                    try:
                        thresh = float(np.quantile(depth_values, f))
                    except Exception:
                        thresh = float(depth_values.max())
                    water_mask = (dem_disp <= thresh) & depth_mask_template
                if nodata is not None:
                    water_mask = water_mask & ~np.isnan(dem_disp)
            st.caption(tr("dynamic_depth_caption", frac=frac))
    water_color = np.array([30,144,255], dtype=np.uint8)
    over = base_rgb.copy(); over_float = over.astype(np.float32)
    over_float[water_mask] = (1 - alpha) * over_float[water_mask] + alpha * water_color
    over_img = over_float.astype(np.uint8)
    with vis_col1:
        dem_src_note = os.path.basename(DEM_PATH)
        date_str = scenario_df['date'].iloc[idx].date()
        caption = f"{date_str} | {tr('volume_word')} {v_sel:.1f} {tr('mcm_unit')} | {tr('dem_label')}: {dem_src_note} | {tr('mask_label')}: {mask_mode}"
        if mask_mode == tr("simulated_level") and z_level is not None:
            caption += f" | {tr('level_word')} {z_level:.2f} {tr('m_unit')}"
        st.image(over_img, caption=caption, use_container_width=True)
        dem_stats = dem_disp.copy(); dmin=float(np.nanmin(dem_stats)); dmax=float(np.nanmax(dem_stats)); dmean=float(np.nanmean(dem_stats))
    st.caption(tr("dem_stats", dmin=dmin, dmax=dmax, dmean=dmean))
