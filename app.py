import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import MarkerCluster, HeatMap, Fullscreen
from streamlit_folium import st_folium
from datetime import datetime
import warnings
from io import BytesIO

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="AQUASTAT - National Water Command Center",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp{background:linear-gradient(135deg,#0a0f1e 0%,#0f1425 100%);}
[data-testid="metric-container"]{background:rgba(17,25,40,0.95);border:1px solid #1f2937;border-radius:15px;padding:20px;box-shadow:0 4px 20px rgba(0,0,0,0.5);transition:transform 0.3s ease;}
[data-testid="metric-container"]:hover{transform:translateY(-5px);border-color:#00e5ff;}
h1,h2,h3{color:#00e5ff;font-weight:700;}
h1{font-size:2.5rem;border-bottom:2px solid rgba(0,229,255,0.3);padding-bottom:10px;}
.stTabs [data-baseweb="tab-list"]{background:rgba(17,25,40,0.95);padding:10px;border-radius:10px;gap:10px;}
.stTabs [data-baseweb="tab"]{border-radius:8px;padding:8px 16px;background:#1f2937;color:white;font-weight:600;}
.stTabs [aria-selected="true"]{background:#00e5ff;color:black;}
[data-testid="stSidebar"]{background:rgba(10,15,30,0.95);border-right:1px solid #1f2937;}
.stDataFrame{background:rgba(17,25,40,0.95);border-radius:10px;border:1px solid #1f2937;}
@keyframes pulse{0%{opacity:1;}50%{opacity:0.5;}100%{opacity:1;}}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300)
def load_all_data():
    try:
        sources     = pd.read_csv("water_sources.csv")
        stations    = pd.read_csv("water_monitoring_stations.csv")
        groundwater = pd.read_csv("groundwater_levels.csv")
        rainfall    = pd.read_csv("rainfall_history.csv")
        alerts      = pd.read_csv("active_alerts.csv")
        usage       = pd.read_csv("water_usage_history.csv")
        regional    = pd.read_csv("regional_stats.csv")
        for df in [sources, stations, groundwater, rainfall, alerts, usage, regional]:
            df.columns = df.columns.str.strip()
        return sources, stations, groundwater, rainfall, alerts, usage, regional
    except Exception as e:
        st.error(f"Error loading CSV data: {e}")
        return [pd.DataFrame()] * 7


with st.spinner("Loading AQUASTAT Command Center..."):
    sources, stations, groundwater, rainfall, alerts, usage, regional = load_all_data()

current_year = datetime.now().year

# ── PROCESS SOURCES ──────────────────────────────────────────────────────────
if not sources.empty:
    for col in ['capacity_percent', 'build_year']:
        if col in sources.columns:
            sources[col] = pd.to_numeric(sources[col], errors='coerce')
    sources['age'] = (current_year - sources['build_year']).clip(0, 200)
    sources['health_score'] = (
        sources['capacity_percent'].fillna(50) * 0.4
        + (100 - sources['age'].clip(0, 100).fillna(50)) * 0.3
        + 30
    ).clip(0, 100)
    sources['risk_level'] = pd.cut(
        sources['capacity_percent'],
        bins=[0, 30, 60, 100],
        labels=['Critical', 'Moderate', 'Good'],
        include_lowest=True
    )

# ── PROCESS ALERTS ────────────────────────────────────────────────────────────
if not alerts.empty:
    if 'alert_time' in alerts.columns and alerts['alert_time'].dtype == 'object':
        alerts['alert_time'] = pd.to_datetime(alerts['alert_time'], errors='coerce')

    def get_alert_reason(row):
        reasons = []
        if pd.notna(row.get('capacity_percent')):
            if row['capacity_percent'] < 30:
                reasons.append(f"Critical capacity: {row['capacity_percent']:.1f}%")
            elif row['capacity_percent'] < 60:
                reasons.append(f"Low capacity: {row['capacity_percent']:.1f}%")
        if pd.notna(row.get('ph_level')):
            if row['ph_level'] < 6.5:
                reasons.append(f"pH too low: {row['ph_level']}")
            elif row['ph_level'] > 8.5:
                reasons.append(f"pH too high: {row['ph_level']}")
        if pd.notna(row.get('dissolved_oxygen_mg_l')) and row['dissolved_oxygen_mg_l'] < 4:
            reasons.append(f"Low dissolved oxygen: {row['dissolved_oxygen_mg_l']} mg/L")
        if pd.notna(row.get('turbidity_ntu')) and row['turbidity_ntu'] > 5:
            reasons.append(f"High turbidity: {row['turbidity_ntu']} NTU")
        return " | ".join(reasons) if reasons else "Monitoring alert - Routine check"

    def determine_alert_status(row):
        if pd.notna(row.get('capacity_percent')) and row['capacity_percent'] < 30:
            return 'CRITICAL'
        if pd.notna(row.get('ph_level')) and (row['ph_level'] < 6 or row['ph_level'] > 9):
            return 'CRITICAL'
        if pd.notna(row.get('capacity_percent')) and row['capacity_percent'] < 60:
            return 'WARNING'
        if pd.notna(row.get('ph_level')) and (row['ph_level'] < 6.5 or row['ph_level'] > 8.5):
            return 'WARNING'
        return 'STABLE'

    alerts['alert_reason'] = alerts.apply(get_alert_reason, axis=1)
    alerts['alert_status'] = alerts.apply(determine_alert_status, axis=1)


def add_coordinates_to_sources(sources_df, stations_df):
    if sources_df.empty or stations_df.empty:
        return sources_df
    df = sources_df.copy()
    if 'district_name' in stations_df.columns:
        s = stations_df.copy()
        s['district_clean'] = s['district_name'].str.strip().str.lower()
        dcoords = s.groupby('district_clean').agg({'latitude': 'first', 'longitude': 'first'}).reset_index()
        if 'district' in df.columns:
            df['district_clean'] = df['district'].str.strip().str.lower()
            df = df.merge(dcoords, on='district_clean', how='left').drop('district_clean', axis=1)
    return df


sources = add_coordinates_to_sources(sources, stations)

if not groundwater.empty and 'avg_depth_meters' in groundwater.columns:
    groundwater['stress_level'] = pd.cut(
        groundwater['avg_depth_meters'], bins=[0, 20, 40, 100], labels=['Low', 'Moderate', 'High']
    )

if not rainfall.empty and 'rainfall_cm' in rainfall.columns:
    rainfall['rainfall_category'] = pd.cut(
        rainfall['rainfall_cm'],
        bins=[0, 50, 150, 300, float('inf')],
        labels=['Low', 'Moderate', 'High', 'Extreme']
    )

# ═══════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════
st.sidebar.title("AQUASTAT")
st.sidebar.caption("Command Interface v2.0")

if st.sidebar.button("Reset All Filters", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### Time Filters")

year_range = (1900, 2025)
if not sources.empty and 'build_year' in sources.columns:
    ayrs = sorted(sources['build_year'].dropna().unique())
    if ayrs:
        year_range = st.sidebar.slider(
            "Build Year Range", min_value=int(min(ayrs)), max_value=int(max(ayrs)),
            value=(int(min(ayrs)), int(max(ayrs)))
        )

st.sidebar.markdown("---")
st.sidebar.markdown("### Geographic Filters")

selected_state = "All States"
if not sources.empty and 'state' in sources.columns:
    states = ['All States'] + sorted(sources['state'].dropna().unique().tolist())
    selected_state = st.sidebar.selectbox("State", states, index=0)

selected_district = "All Districts"
if not sources.empty and 'district' in sources.columns:
    dbase = sources[sources['state'] == selected_state] if selected_state != "All States" else sources
    dists = sorted(dbase['district'].dropna().unique().tolist())
    if dists:
        selected_district = st.sidebar.selectbox("District", ['All Districts'] + dists, index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("### Source Filters")

selected_type = "All Types"
if not sources.empty and 'source_type' in sources.columns:
    stypes = ['All Types'] + sorted(sources['source_type'].dropna().unique().tolist())
    selected_type = st.sidebar.selectbox("Source Type", stypes, index=0)

capacity_range = (0.0, 100.0)
if not sources.empty and 'capacity_percent' in sources.columns:
    mn, mx = float(sources['capacity_percent'].min()), float(sources['capacity_percent'].max())
    capacity_range = st.sidebar.slider("Capacity %", min_value=mn, max_value=mx, value=(mn, mx))

selected_risk = "All Risk Levels"
if not sources.empty and 'risk_level' in sources.columns:
    risk_opts = ['All Risk Levels'] + list(sources['risk_level'].dropna().unique())
    selected_risk = st.sidebar.selectbox("Risk Level", risk_opts, index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("### Map Settings")
map_style     = st.sidebar.selectbox("Map Style", ["Esri Satellite (Official)", "Dark Matter", "Light Matter"], index=0)
show_heatmap  = st.sidebar.checkbox("Show Heatmap", True)
show_clusters = st.sidebar.checkbox("Show Clusters", True)
show_stations = st.sidebar.checkbox("Show Monitoring Stations", True)
marker_size   = st.sidebar.slider("Marker Size", 5, 20, 10)
st.sidebar.markdown("---")
st.sidebar.caption(f"Total Sources in DB: {len(sources)}")

# ═══════════════════════════════════════════
# APPLY FILTERS
# ═══════════════════════════════════════════
def apply_filters():
    f = sources.copy()
    if selected_state != "All States" and 'state' in f.columns:
        f = f[f['state'] == selected_state]
    if selected_district != "All Districts" and 'district' in f.columns:
        f = f[f['district'] == selected_district]
    if selected_type != "All Types" and 'source_type' in f.columns:
        f = f[f['source_type'] == selected_type]
    if 'build_year' in f.columns:
        f = f[(f['build_year'] >= year_range[0]) & (f['build_year'] <= year_range[1])]
    if 'capacity_percent' in f.columns:
        f = f[(f['capacity_percent'] >= capacity_range[0]) & (f['capacity_percent'] <= capacity_range[1])]
    if selected_risk != "All Risk Levels" and 'risk_level' in f.columns:
        f = f[f['risk_level'] == selected_risk]
    return f


def filter_stations():
    f = stations.copy()
    if selected_state != "All States" and 'state_name' in f.columns:
        f = f[f['state_name'] == selected_state]
    if selected_district != "All Districts" and 'district_name' in f.columns:
        f = f[f['district_name'] == selected_district]
    return f


filtered_sources  = apply_filters()
filtered_stations = filter_stations()

# ═══════════════════════════════════════════
# HEADER KPIs
# ═══════════════════════════════════════════
st.title("AQUASTAT National Water Command Center")
st.caption(f"**Live Intelligence** - Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Total Sources", f"{len(sources):,}", f"{len(sources)-len(filtered_sources)} filtered")
with c2:
    avg_cap = filtered_sources['capacity_percent'].mean() if not filtered_sources.empty and 'capacity_percent' in filtered_sources.columns else 0
    st.metric("Avg Capacity", f"{avg_cap:.1f}%")
with c3:
    crit_src = len(filtered_sources[filtered_sources['capacity_percent'] < 30]) if not filtered_sources.empty and 'capacity_percent' in filtered_sources.columns else 0
    st.metric("Critical Sources", crit_src, delta_color="inverse")
with c4:
    src_map = len(filtered_sources[filtered_sources['latitude'].notna()]) if not filtered_sources.empty and 'latitude' in filtered_sources.columns else 0
    st.metric("Sources on Map", src_map)
with c5:
    crit_alr = len(alerts[alerts['alert_status'] == 'CRITICAL']) if not alerts.empty and 'alert_status' in alerts.columns else 0
    st.metric("Critical Alerts", crit_alr, delta_color="inverse")

st.markdown("---")

# ═══════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs(["DASHBOARD", "MAP VIEW", "ANALYTICS", "ALERTS", "DATA TABLES"])

# ────────────────────────────────────────────
# TAB 1  DASHBOARD
# ────────────────────────────────────────────
with tab1:
    if filtered_sources.empty:
        st.warning("No water sources match the current filters.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Capacity Distribution")
            if 'capacity_percent' in filtered_sources.columns:
                fig = px.histogram(filtered_sources, x='capacity_percent', nbins=20,
                    title=f"Storage Capacity Distribution ({len(filtered_sources)} sources)",
                    template="plotly_dark", color_discrete_sequence=['#00e5ff'])
                fig.update_layout(xaxis_title="Capacity (%)", yaxis_title="Number of Sources")
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.subheader("Source Types")
            if 'source_type' in filtered_sources.columns:
                tc = filtered_sources['source_type'].value_counts().reset_index()
                tc.columns = ['Source Type', 'Count']
                fig = px.pie(tc, values='Count', names='Source Type', title="Water Sources by Type",
                    template="plotly_dark", color_discrete_sequence=px.colors.sequential.Tealgrn)
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Groundwater Stress Levels")
            if not groundwater.empty and 'stress_level' in groundwater.columns:
                fgw = groundwater.copy()
                if selected_district != "All Districts" and 'district_name' in fgw.columns:
                    fgw = fgw[fgw['district_name'] == selected_district]
                sl = fgw['stress_level'].value_counts().reset_index()
                sl.columns = ['Stress Level', 'Count']
                fig = px.bar(sl, x='Stress Level', y='Count', title="Groundwater Stress Distribution",
                    template="plotly_dark", color='Stress Level',
                    color_discrete_map={'Low':'#00ff9d','Moderate':'#ffd700','High':'#ff4444'}, text_auto=True)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No groundwater data available")
        with c2:
            st.subheader("Rainfall by Season")
            if not rainfall.empty and 'season' in rainfall.columns and 'rainfall_cm' in rainfall.columns:
                fr = rainfall.copy()
                if selected_district != "All Districts" and 'district_name' in fr.columns:
                    fr = fr[fr['district_name'] == selected_district]
                sr = fr.groupby('season')['rainfall_cm'].mean().reset_index()
                fig = px.bar(sr, x='season', y='rainfall_cm', title="Average Rainfall by Season",
                    template="plotly_dark", color='rainfall_cm', color_continuous_scale='Blues', text_auto='.1f')
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No rainfall data available")

        if 'risk_level' in filtered_sources.columns:
            st.subheader("Risk Distribution")
            rc = filtered_sources['risk_level'].value_counts().reset_index()
            rc.columns = ['Risk Level', 'Count']
            fig = px.bar(rc, x='Risk Level', y='Count', color='Risk Level',
                color_discrete_map={'Good':'#00ff9d','Moderate':'#ffd700','Critical':'#ff4444'},
                title="Infrastructure Risk Assessment", template="plotly_dark", text_auto=True)
            st.plotly_chart(fig, use_container_width=True)

# ────────────────────────────────────────────
# TAB 2  MAP VIEW
# ────────────────────────────────────────────
with tab2:
    st.subheader("National Interactive Water Resources Map")

    style_map = {
        "Esri Satellite (Official)": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "Dark Matter": "https://cartodb-basemaps-a.global.ssl.fastly.net/dark_all/{z}/{x}/{y}.png",
        "Light Matter": "https://cartodb-basemaps-a.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png"
    }

    center_lat, center_lon, zoom = 20.5937, 78.9629, 5
    if not filtered_sources.empty and 'latitude' in filtered_sources.columns:
        swc = filtered_sources[filtered_sources['latitude'].notna() & filtered_sources['longitude'].notna()]
        if not swc.empty:
            zoom = 9 if selected_district != "All Districts" else (7 if selected_state != "All States" else 5)
            center_lat, center_lon = swc['latitude'].mean(), swc['longitude'].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom,
                   tiles=style_map[map_style], attr='AQUASTAT | Data: Esri, OSM')
    Fullscreen().add_to(m)
    mc = MarkerCluster().add_to(m) if (show_clusters and len(filtered_sources) > 10) else m

    heat_data = []
    sources_on_map = 0

    if not filtered_sources.empty and 'latitude' in filtered_sources.columns:
        swc = filtered_sources[filtered_sources['latitude'].notna() & filtered_sources['longitude'].notna()]
        for _, row in swc.iterrows():
            cap = row.get('capacity_percent', 50)
            if pd.notna(cap) and cap < 30:
                col, rtxt, rico = '#ff4444', 'CRITICAL', '🔴'
            elif pd.notna(cap) and cap < 60:
                col, rtxt, rico = '#ffd700', 'MODERATE', '🟡'
            else:
                col, rtxt, rico = '#00ff9d', 'GOOD', '🟢'
            heat_data.append([row['latitude'], row['longitude']])
            sources_on_map += 1
            cv = float(cap) if pd.notna(cap) else 0.0
            ph = f"<tr><td colspan='2'><div style='background:rgba(255,0,0,0.1);padding:5px;border-radius:4px;'><b>Alert:</b> {alerts[alerts['source_name']==row['source_name']].iloc[-1].get('alert_reason','') if not alerts.empty and 'source_name' in alerts.columns and len(alerts[alerts['source_name']==row['source_name']])>0 else ''}</div></td></tr>"
            popup_html = f"<div style='font-family:Arial;min-width:280px;background:#0a0f1e;color:white;padding:14px;border-radius:10px;border-left:5px solid {col};'><h4 style='color:{col};margin:0 0 8px 0;'>{rico} {row.get('source_name','')}</h4><hr style='border-color:#1f2937;margin:8px 0;'><table style='width:100%;'><tr><td><b>Type:</b></td><td>{row.get('source_type','')}</td></tr><tr><td><b>District:</b></td><td>{row.get('district','')}</td></tr><tr><td><b>State:</b></td><td>{row.get('state','')}</td></tr><tr><td><b>Capacity:</b></td><td>{cv:.1f}%</td></tr><tr><td><b>Age:</b></td><td>{row.get('age',0):.0f} yrs</td></tr><tr><td><b>Health:</b></td><td>{row.get('health_score',0):.1f}%</td></tr></table></div>"
            marker = folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=marker_size + (3 if pd.notna(cap) and cap < 30 else 0),
                color=col, fill=True, fillOpacity=0.7,
                popup=folium.Popup(popup_html, max_width=320),
                tooltip=f"{row.get('source_name','')} | {cv:.0f}% | {rtxt}"
            )
            if show_clusters and len(filtered_sources) > 10:
                marker.add_to(mc)
            else:
                marker.add_to(m)

        if show_heatmap and heat_data:
            HeatMap(heat_data, radius=15, blur=10,
                    gradient={0.2:'blue',0.4:'cyan',0.6:'lime',0.8:'yellow',1:'red'}).add_to(m)

    if show_stations and not filtered_stations.empty and 'latitude' in filtered_stations.columns:
        for _, st_row in filtered_stations[filtered_stations['latitude'].notna() & filtered_stations['longitude'].notna()].iterrows():
            sts = st_row.get('status', 'Unknown')
            sc = 'green' if sts == 'Active' else ('orange' if sts == 'Maintenance' else 'red')
            si = '✅' if sts == 'Active' else ('🔄' if sts == 'Maintenance' else '⚠️')
            sph = st_row.get('ph_level', 7); sdo = st_row.get('dissolved_oxygen_mg_l', 5)
            wq = "Poor" if (pd.notna(sph) and (sph < 6.5 or sph > 8.5)) else ("Fair" if (pd.notna(sdo) and sdo < 4) else "Good")
            qc = "#ff4444" if wq == "Poor" else ("#ffd700" if wq == "Fair" else "#00ff9d")
            sp = f"<div style='font-family:Arial;min-width:260px;background:#0a0f1e;color:white;padding:14px;border-radius:10px;border-left:5px solid {sc};'><h4 style='margin:0 0 8px 0;'>{si} {st_row.get('station_name','')}</h4><hr style='border-color:#1f2937;margin:8px 0;'><table><tr><td><b>District:</b></td><td>{st_row.get('district_name','')}</td></tr><tr><td><b>Status:</b></td><td><span style='color:{sc};'>{sts}</span></td></tr><tr><td><b>pH:</b></td><td>{st_row.get('ph_level','N/A')}</td></tr><tr><td><b>DO:</b></td><td>{st_row.get('dissolved_oxygen_mg_l','N/A')} mg/L</td></tr><tr><td><b>Turbidity:</b></td><td>{st_row.get('turbidity_ntu','N/A')} NTU</td></tr><tr><td><b>Quality:</b></td><td><span style='color:{qc};'>{wq}</span></td></tr></table></div>"
            folium.Marker(
                location=[st_row['latitude'], st_row['longitude']],
                icon=folium.Icon(color=sc, icon='info-sign', prefix='glyphicon'),
                popup=folium.Popup(sp, max_width=300),
                tooltip=f"Station: {st_row.get('station_name','')} | {sts}"
            ).add_to(m)

    st_folium(m, width=1300, height=600)

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Sources on Map", sources_on_map)
    with c2: st.metric("Total Filtered Sources", len(filtered_sources))
    with c3:
        cov = (sources_on_map / len(filtered_sources) * 100) if len(filtered_sources) > 0 else 0
        st.metric("Coordinate Coverage", f"{cov:.1f}%")
    st.markdown("---")
    cols = st.columns(5)
    with cols[0]: st.markdown("🟢 **Good** (>=60%)")
    with cols[1]: st.markdown("🟡 **Moderate** (30-60%)")
    with cols[2]: st.markdown("🔴 **Critical** (<30%)")
    with cols[3]: st.markdown("🔵 **Monitoring Station**")
    with cols[4]: st.markdown("🔥 **Heatmap Area**")

# ────────────────────────────────────────────
# TAB 3  ANALYTICS
# ────────────────────────────────────────────
with tab3:
    st.subheader("Advanced Analytics")
    atab1, atab2, atab3 = st.tabs(["Trends", "Comparisons", "Statistics"])

    with atab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Rainfall Trend")
            if not rainfall.empty and 'record_year' in rainfall.columns and 'rainfall_cm' in rainfall.columns:
                fr = rainfall.copy()
                if selected_district != "All Districts" and 'district_name' in fr.columns:
                    fr = fr[fr['district_name'] == selected_district]
                rt = fr.groupby('record_year')['rainfall_cm'].mean().reset_index()
                fig = px.line(rt, x='record_year', y='rainfall_cm', title="Average Rainfall Over Years",
                    template="plotly_dark", markers=True)
                fig.update_traces(line_color='#00e5ff', line_width=3)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No rainfall data available")
        with c2:
            st.subheader("Groundwater Trend")
            if not groundwater.empty and 'assessment_year' in groundwater.columns and 'avg_depth_meters' in groundwater.columns:
                fgw = groundwater.copy()
                if selected_district != "All Districts" and 'district_name' in fgw.columns:
                    fgw = fgw[fgw['district_name'] == selected_district]
                gt = fgw.groupby('assessment_year')['avg_depth_meters'].mean().reset_index()
                fig = px.line(gt, x='assessment_year', y='avg_depth_meters', title="Average Groundwater Depth",
                    template="plotly_dark", markers=True)
                fig.update_traces(line_color='#ffd700', line_width=3)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No groundwater data available")

    with atab2:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Capacity by State")
            if not filtered_sources.empty and 'state' in filtered_sources.columns and 'capacity_percent' in filtered_sources.columns:
                sc2 = filtered_sources.groupby('state')['capacity_percent'].mean().sort_values(ascending=False).head(10)
                fig = px.bar(x=sc2.values, y=sc2.index, orientation='h',
                    title=f"Average Capacity by State ({len(sc2)} states)",
                    template="plotly_dark", color=sc2.values, color_continuous_scale='Tealgrn',
                    labels={'x':'Avg Capacity (%)','y':'State'})
                fig.update_traces(texttemplate='%{x:.1f}%', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No state data available")
        with c2:
            st.subheader("Extraction vs Recharge")
            if not groundwater.empty and 'recharge_rate_mcm' in groundwater.columns and 'extraction_pct' in groundwater.columns:
                fgw = groundwater.copy()
                if selected_district != "All Districts" and 'district_name' in fgw.columns:
                    fgw = fgw[fgw['district_name'] == selected_district]
                fig = px.scatter(fgw, x='recharge_rate_mcm', y='extraction_pct',
                    size='avg_depth_meters' if 'avg_depth_meters' in fgw.columns else None,
                    color='district_name' if 'district_name' in fgw.columns else None,
                    title="Groundwater Extraction vs Recharge Rate", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No groundwater data available")

    with atab3:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Statistical Summary")
            if not filtered_sources.empty:
                scols = [c for c in ['capacity_percent','age','health_score'] if c in filtered_sources.columns]
                if scols:
                    st.dataframe(filtered_sources[scols].describe().style.format("{:.2f}"), use_container_width=True)
                else:
                    st.info("No numerical data available")
            else:
                st.info("No source data available")
        with c2:
            st.subheader("Correlation Matrix")
            if not filtered_sources.empty and not groundwater.empty and 'district' in filtered_sources.columns and 'district_name' in groundwater.columns:
                merged = filtered_sources.merge(groundwater, left_on='district', right_on='district_name', how='inner')
                if not merged.empty:
                    ncols = [c for c in ['capacity_percent','age','avg_depth_meters','extraction_pct','recharge_rate_mcm'] if c in merged.columns]
                    if len(ncols) >= 2:
                        cd = merged[ncols].dropna()
                        if not cd.empty:
                            fig = px.imshow(cd.corr(), text_auto=True, aspect="auto",
                                title="Feature Correlation Matrix", template="plotly_dark",
                                color_continuous_scale='RdBu_r')
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Insufficient data for correlation")
                    else:
                        st.info("Not enough numerical columns")
                else:
                    st.info("No matching data for correlation")
            else:
                st.info("Insufficient data for correlation")

# ────────────────────────────────────────────
# TAB 4  ALERTS
# KEY FIX: build each card as a single-line string via "".join()
# so Streamlit never sees a leading-whitespace / indented HTML block
# which it would render as a <pre><code> block instead of HTML.
# ────────────────────────────────────────────
with tab4:
    st.subheader("Active Alerts and Warnings")

    if not alerts.empty and 'alert_status' in alerts.columns:
        aws = alerts.copy()
        if not sources.empty and 'source_name' in sources.columns and 'source_name' in aws.columns:
            src_cols = [c for c in ['source_name','source_type','district','state'] if c in sources.columns]
            if src_cols:
                aws = aws.merge(sources[src_cols], on='source_name', how='left')
                for col in ['source_type','district','state']:
                    if col in aws.columns:
                        aws[col] = aws[col].fillna('Unknown')

        acounts = aws['alert_status'].value_counts()
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            cc = int(acounts.get('CRITICAL', 0))
            st.metric("CRITICAL", cc, delta="Immediate action required" if cc > 0 else None, delta_color="inverse")
        with c2:
            wc = int(acounts.get('WARNING', 0))
            st.metric("WARNING", wc, delta="Monitor closely" if wc > 0 else None)
        with c3:
            st.metric("STABLE", int(acounts.get('STABLE', 0)))
        with c4:
            st.metric("TOTAL", len(aws))

        st.markdown("---")

        fa = aws.copy()
        if selected_state != "All States" and 'state' in fa.columns:
            fa = fa[fa['state'] == selected_state]
        if selected_district != "All Districts" and 'district' in fa.columns:
            fa = fa[fa['district'] == selected_district]
        if selected_type != "All Types" and 'source_type' in fa.columns:
            fa = fa[fa['source_type'] == selected_type]

        atf = st.selectbox("Filter by Alert Status", ["All Alerts","CRITICAL","WARNING","STABLE"], index=0)
        if atf != "All Alerts":
            fa = fa[fa['alert_status'] == atf]

        if fa.empty:
            st.info(f"No {atf.lower()} alerts match the current filters")
        else:
            fa = fa.copy()
            fa['_sev'] = fa['alert_status'].map({'CRITICAL':0,'WARNING':1,'STABLE':2})
            fa = fa.sort_values('_sev').drop('_sev', axis=1)

            def _s(v, d='Unknown'):
                if v is None: return d
                s = str(v)
                return d if s.lower() in ('nan','none','') else s

            def _m(v, suf=''):
                if v is None: return "N/A"
                if isinstance(v, float) and np.isnan(v): return "N/A"
                return f"{v}{suf}"

            for _, alert in fa.iterrows():
                st_val = alert['alert_status']
                if st_val == 'CRITICAL':
                    bc, icon, sevtxt = "#ff4444", "🔴", "IMMEDIATE ACTION REQUIRED"
                elif st_val == 'WARNING':
                    bc, icon, sevtxt = "#ffd700", "🟡", "MONITOR CLOSELY"
                else:
                    bc, icon, sevtxt = "#00ff9d", "🟢", "NORMAL OPERATIONS"

                sname  = _s(alert.get('source_name'))
                stype  = _s(alert.get('source_type'))
                dist   = _s(alert.get('district'))
                state  = _s(alert.get('state'))
                reason = _s(alert.get('alert_reason'), 'No reason provided')

                loc = f"{dist}, {state}" if dist != 'Unknown' and state != 'Unknown' else (dist if dist != 'Unknown' else (state if state != 'Unknown' else "Location unknown"))

                at = alert.get('alert_time', '')
                tstr = at.strftime('%Y-%m-%d %H:%M:%S') if isinstance(at, pd.Timestamp) else str(at)

                raw_cap = alert.get('capacity_percent', None)
                if raw_cap is None or (isinstance(raw_cap, float) and np.isnan(raw_cap)):
                    cap_disp, cv = "N/A", 0.0
                else:
                    cv = float(raw_cap)
                    cap_disp = f"{cv:.1f}%"

                ph_d   = _m(alert.get('ph_level'))
                do_d   = _m(alert.get('dissolved_oxygen_mg_l'), ' mg/L')
                turb_d = _m(alert.get('turbidity_ntu'), ' NTU')
                temp_d = _m(alert.get('temperature_c'), ' C')

                # Build card as list of parts then join – guarantees zero leading whitespace
                parts = [
                    f'<div style="background:rgba(17,25,40,0.95);border:2px solid {bc};border-left:6px solid {bc};border-radius:15px;padding:20px;margin:12px 0;">',
                    f'<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px;">',
                    f'<div><span style="font-size:1.25rem;font-weight:700;color:white;">{icon} {sname}</span><br>',
                    f'<span style="color:#8892b0;font-size:0.9rem;">{stype} | {loc}</span></div>',
                    f'<div style="text-align:right;"><span style="color:{bc};font-size:1.1rem;font-weight:700;">{st_val}</span><br>',
                    f'<span style="color:#8892b0;font-size:0.78rem;">{sevtxt}</span></div></div>',
                    f'<hr style="border:none;border-top:1px solid {bc};margin:10px 0;">',
                    f'<div style="background:rgba(0,0,0,0.25);border-left:4px solid {bc};padding:10px 14px;border-radius:6px;margin-bottom:14px;">',
                    f'<span style="color:{bc};font-weight:700;">Alert Reason: </span>',
                    f'<span style="color:white;">{reason}</span></div>',
                    f'<div style="display:flex;flex-wrap:wrap;gap:24px;margin-bottom:14px;">',
                    f'<div style="flex:1;min-width:180px;">',
                    f'<div style="color:#aaa;font-size:0.82rem;">Current Capacity</div>',
                    f'<div style="font-size:1.05rem;font-weight:700;color:white;margin:4px 0;">{cap_disp}</div>',
                    f'<div style="background:#1f2937;height:8px;border-radius:4px;">',
                    f'<div style="background:{bc};width:{cv}%;height:8px;border-radius:4px;"></div></div></div>',
                    f'<div><div style="color:#aaa;font-size:0.82rem;">pH Level</div><div style="color:white;font-weight:600;margin-top:4px;">{ph_d}</div></div>',
                    f'<div><div style="color:#aaa;font-size:0.82rem;">Alert Time</div><div style="color:white;margin-top:4px;font-size:0.9rem;">{tstr}</div></div>',
                    f'</div>',
                    f'<div style="display:flex;flex-wrap:wrap;gap:24px;padding-top:12px;border-top:1px solid #1f2937;">',
                    f'<div><div style="color:#8892b0;font-size:0.8rem;">Dissolved Oxygen</div><strong style="color:white;">{do_d}</strong></div>',
                    f'<div><div style="color:#8892b0;font-size:0.8rem;">Turbidity</div><strong style="color:white;">{turb_d}</strong></div>',
                    f'<div><div style="color:#8892b0;font-size:0.8rem;">Temperature</div><strong style="color:white;">{temp_d}</strong></div>',
                    f'</div></div>',
                ]
                st.markdown("".join(parts), unsafe_allow_html=True)

    else:
        st.success("No active alerts - All systems normal")
        st.balloons()

# ────────────────────────────────────────────
# TAB 5  DATA TABLES
# ────────────────────────────────────────────
with tab5:
    st.subheader("Data Explorer")

    table_choice = st.selectbox("Select Table to View",
        ["Water Sources","Monitoring Stations","Groundwater Levels",
         "Rainfall History","Water Usage","Active Alerts","Regional Statistics"])

    def pick(df, cols):
        return [c for c in cols if c in df.columns]

    def dl(df, fname):
        if not df.empty:
            st.download_button("Download CSV", df.to_csv(index=False).encode('utf-8'), fname, "text/csv", use_container_width=True)

    if table_choice == "Water Sources":
        cols = pick(filtered_sources, ['source_name','source_type','capacity_percent','max_capacity_mcm','build_year','age','state','district','risk_level'])
        st.dataframe(filtered_sources[cols] if cols else filtered_sources, use_container_width=True, hide_index=True)
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Total Sources", len(filtered_sources))
        with c2:
            avg = filtered_sources['capacity_percent'].mean() if 'capacity_percent' in filtered_sources.columns and not filtered_sources.empty else None
            st.metric("Avg Capacity", f"{avg:.1f}%" if avg is not None else "N/A")
        with c3:
            tb = len(filtered_sources[filtered_sources['is_transboundary']==1]) if 'is_transboundary' in filtered_sources.columns else "N/A"
            st.metric("Transboundary", tb)
        dl(filtered_sources, f"water_sources_{selected_state}_{selected_district}.csv")

    elif table_choice == "Monitoring Stations":
        ds = filter_stations()
        cols = pick(ds, ['station_name','state_name','district_name','latitude','longitude','ph_level','dissolved_oxygen_mg_l','turbidity_ntu','status'])
        st.dataframe(ds[cols] if cols else ds, use_container_width=True, hide_index=True)
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Total Stations", len(ds))
        with c2: st.metric("Active", len(ds[ds['status']=='Active']) if 'status' in ds.columns else "N/A")
        with c3: st.metric("Maintenance", len(ds[ds['status']=='Maintenance']) if 'status' in ds.columns else "N/A")
        dl(ds, f"monitoring_stations_{selected_state}_{selected_district}.csv")

    elif table_choice == "Groundwater Levels":
        dg = groundwater.copy()
        if selected_district != "All Districts" and 'district_name' in dg.columns:
            dg = dg[dg['district_name'] == selected_district]
        cols = pick(dg, ['district_name','avg_depth_meters','extraction_pct','recharge_rate_mcm','assessment_year','stress_level'])
        st.dataframe(dg[cols] if cols else dg, use_container_width=True, hide_index=True)
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Districts", len(dg))
        with c2: st.metric("Avg Depth", f"{dg['avg_depth_meters'].mean():.1f} m" if 'avg_depth_meters' in dg.columns and not dg.empty else "N/A")
        with c3: st.metric("High Stress", len(dg[dg['stress_level']=='High']) if 'stress_level' in dg.columns else "N/A")
        dl(dg, f"groundwater_{selected_district}.csv")

    elif table_choice == "Rainfall History":
        dr = rainfall.copy()
        if selected_district != "All Districts" and 'district_name' in dr.columns:
            dr = dr[dr['district_name'] == selected_district]
        cols = pick(dr, ['district_name','rainfall_cm','record_year','season','rainfall_category'])
        st.dataframe(dr[cols] if cols else dr, use_container_width=True, hide_index=True)
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Total Records", len(dr))
        with c2: st.metric("Avg Rainfall", f"{dr['rainfall_cm'].mean():.1f} cm" if 'rainfall_cm' in dr.columns and not dr.empty else "N/A")
        with c3: st.metric("Years of Data", dr['record_year'].nunique() if 'record_year' in dr.columns else "N/A")
        dl(dr, f"rainfall_{selected_district}.csv")

    elif table_choice == "Water Usage":
        du = usage.copy()
        if selected_state != "All States" and 'state' in du.columns: du = du[du['state']==selected_state]
        if selected_district != "All Districts" and 'district' in du.columns: du = du[du['district']==selected_district]
        if selected_type != "All Types" and 'source_type' in du.columns: du = du[du['source_type']==selected_type]
        cols = pick(du, ['source_name','source_type','sector','sub_sector','consumer_name','consumption_mcm','record_year','season','state','district'])
        st.dataframe(du[cols] if cols else du, use_container_width=True, hide_index=True)
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Total Records", len(du))
        with c2: st.metric("Total Consumption", f"{du['consumption_mcm'].sum():.1f} MCM" if 'consumption_mcm' in du.columns and not du.empty else "N/A")
        with c3: st.metric("Avg Consumption", f"{du['consumption_mcm'].mean():.1f} MCM" if 'consumption_mcm' in du.columns and not du.empty else "N/A")
        dl(du, f"water_usage_{selected_state}_{selected_district}.csv")

    elif table_choice == "Active Alerts":
        da = alerts.copy()
        if not sources.empty and 'source_name' in sources.columns and 'source_name' in da.columns:
            sc2 = pick(sources, ['source_name','source_type','district','state'])
            if sc2: da = da.merge(sources[sc2], on='source_name', how='left')
        if selected_state != "All States" and 'state' in da.columns: da = da[da['state']==selected_state]
        if selected_district != "All Districts" and 'district' in da.columns: da = da[da['district']==selected_district]
        if selected_type != "All Types" and 'source_type' in da.columns: da = da[da['source_type']==selected_type]
        cols = pick(da, ['source_name','source_type','district','state','capacity_percent','ph_level','alert_status','alert_time','alert_reason'])
        disp = da[cols].copy() if cols else da.copy()
        if 'alert_time' in disp.columns and pd.api.types.is_datetime64_any_dtype(disp['alert_time']):
            disp['alert_time'] = disp['alert_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        st.dataframe(disp, use_container_width=True, hide_index=True)
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Total Alerts", len(da))
        with c2: st.metric("Critical", len(da[da['alert_status']=='CRITICAL']) if 'alert_status' in da.columns else "N/A")
        with c3: st.metric("Warning", len(da[da['alert_status']=='WARNING']) if 'alert_status' in da.columns else "N/A")
        dl(da, f"active_alerts_{selected_state}_{selected_district}.csv")

    elif table_choice == "Regional Statistics":
        cols = pick(regional, ['region_name','population_count','annual_rainfall_avg_cm'])
        st.dataframe(regional[cols] if cols else regional, use_container_width=True, hide_index=True)
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Total Regions", len(regional))
        with c2: st.metric("Total Population", f"{regional['population_count'].sum():,}" if 'population_count' in regional.columns and not regional.empty else "N/A")
        with c3: st.metric("Avg Rainfall", f"{regional['annual_rainfall_avg_cm'].mean():.1f} cm" if 'annual_rainfall_avg_cm' in regional.columns and not regional.empty else "N/A")
        dl(regional, "regional_statistics.csv")

# ── SIDEBAR SUMMARY ───────────────────────────────────────────────────────────
with st.sidebar.expander("Current Filter Summary", expanded=False):
    smc = len(filtered_sources[filtered_sources['latitude'].notna()]) if not filtered_sources.empty and 'latitude' in filtered_sources.columns else 0
    st.markdown(
        f"**Time:** {year_range[0]}-{year_range[1]}  \n"
        f"**State:** {selected_state}  \n"
        f"**District:** {selected_district}  \n"
        f"**Type:** {selected_type}  \n"
        f"**Capacity:** {capacity_range[0]:.0f}%-{capacity_range[1]:.0f}%  \n"
        f"**Risk:** {selected_risk}  \n"
        f"**Sources:** {len(filtered_sources)} of {len(sources)}  \n"
        f"**On Map:** {smc}"
    )

# ── EXPORT ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
if st.sidebar.button("Export All Filtered Data", use_container_width=True):
    export_data = {'water_sources': filtered_sources, 'monitoring_stations': filtered_stations}
    if not filtered_sources.empty and 'district' in filtered_sources.columns and 'district_name' in groundwater.columns:
        export_data['groundwater'] = groundwater[groundwater['district_name'].isin(filtered_sources['district'].unique())]
    if not filtered_sources.empty and 'district' in filtered_sources.columns and 'district_name' in rainfall.columns:
        export_data['rainfall'] = rainfall[rainfall['district_name'].isin(filtered_sources['district'].unique())]
    if not filtered_sources.empty and 'source_name' in filtered_sources.columns and 'source_name' in usage.columns:
        export_data['usage'] = usage[usage['source_name'].isin(filtered_sources['source_name'])]
    if not filtered_sources.empty and 'source_name' in filtered_sources.columns and 'source_name' in alerts.columns:
        export_data['alerts'] = alerts[alerts['source_name'].isin(filtered_sources['source_name'])]
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, df in export_data.items():
            if not df.empty:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
    st.sidebar.download_button("Download Excel Report", output.getvalue(),
        f"aquastat_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div style="text-align:center;"><p style="color:#00e5ff;font-size:1.2rem;">AQUASTAT</p><p style="color:#8892b0;">National Water Command Center</p></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div style="text-align:center;"><p style="color:#8892b0;">Data Source: Ministry of Jal Shakti</p><p style="color:#8892b0;">Last Updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div style="text-align:center;"><p style="color:#8892b0;">2025 All Rights Reserved</p><p style="color:#8892b0;">Version 3.0 | For Official Use</p></div>', unsafe_allow_html=True)

st.markdown('<div style="position:fixed;bottom:10px;right:10px;background:rgba(0,229,255,0.1);padding:5px 10px;border-radius:5px;font-size:0.8rem;">Data refreshes every 5 minutes</div>', unsafe_allow_html=True)
