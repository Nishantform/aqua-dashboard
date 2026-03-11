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
import json
import warnings
import os
from io import BytesIO

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# -------------------------
# PAGE CONFIG
# -------------------------

st.set_page_config(
    page_title="AQUASTAT - National Water Command Center",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# CUSTOM CSS
# -------------------------

st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0a0f1e 0%, #0f1425 100%);
    }
    
    /* Metric containers */
    [data-testid="metric-container"] {
        background: rgba(17, 25, 40, 0.95);
        border: 1px solid #1f2937;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        transition: transform 0.3s ease;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        border-color: #00e5ff;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #00e5ff;
        font-weight: 700;
    }
    
    h1 {
        font-size: 2.5rem;
        border-bottom: 2px solid rgba(0, 229, 255, 0.3);
        padding-bottom: 10px;
    }
    
    /* Cards */
    .info-card {
        background: rgba(17, 25, 40, 0.95);
        border: 1px solid #1f2937;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    
    .info-card:hover {
        border-color: #00e5ff;
        box-shadow: 0 8px 30px rgba(0, 229, 255, 0.2);
    }
    
    /* Status indicators */
    .status-critical {
        color: #ff4444;
        font-weight: 600;
        animation: pulse 2s infinite;
    }
    
    .status-warning {
        color: #ffd700;
        font-weight: 600;
    }
    
    .status-good {
        color: #00ff9d;
        font-weight: 600;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(17, 25, 40, 0.95);
        padding: 10px;
        border-radius: 10px;
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        background: #1f2937;
        color: white;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: #00e5ff;
        color: black;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(10, 15, 30, 0.95);
        border-right: 1px solid #1f2937;
    }
    
    /* Dataframes */
    .stDataFrame {
        background: rgba(17, 25, 40, 0.95);
        border-radius: 10px;
        border: 1px solid #1f2937;
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background: linear-gradient(90deg, #00e5ff, #00b8ff);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 20px;
        color: #8892b0;
        border-top: 1px solid #1f2937;
        margin-top: 30px;
    }
    
    /* Alert reasons */
    .alert-reason {
        background: rgba(255,255,255,0.05);
        padding: 8px;
        border-radius: 5px;
        margin-top: 10px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------
# DATABASE CONNECTION
# -------------------------

@st.cache_data(ttl=300)
def load_all_data():
    """Load all data from local CSV files"""
    try:
        sources = pd.read_csv("water_sources.csv")
        stations = pd.read_csv("water_monitoring_stations.csv")
        groundwater = pd.read_csv("groundwater_levels.csv")
        rainfall = pd.read_csv("rainfall_history.csv")
        alerts = pd.read_csv("active_alerts.csv")
        usage = pd.read_csv("water_usage_history.csv")
        regional = pd.read_csv("regional_stats.csv")
        
        # Clean column names by stripping whitespace
        sources.columns = sources.columns.str.strip()
        stations.columns = stations.columns.str.strip()
        groundwater.columns = groundwater.columns.str.strip()
        rainfall.columns = rainfall.columns.str.strip()
        alerts.columns = alerts.columns.str.strip()
        usage.columns = usage.columns.str.strip()
        regional.columns = regional.columns.str.strip()
        
        return sources, stations, groundwater, rainfall, alerts, usage, regional
    except Exception as e:
        st.error(f"Error loading CSV data: {e}")
        return [pd.DataFrame()] * 7

# -------------------------
# LOAD DATA
# -------------------------

with st.spinner("🚀 Loading AQUASTAT Command Center..."):
    sources, stations, groundwater, rainfall, alerts, usage, regional = load_all_data()

# -------------------------
# DATA PROCESSING
# -------------------------

current_year = datetime.now().year

# Process Sources
if not sources.empty:
    # Convert numeric columns
    for col in ['capacity_percent', 'build_year']:
        if col in sources.columns:
            sources[col] = pd.to_numeric(sources[col], errors='coerce')
    
    # Calculate age
    sources['age'] = current_year - sources['build_year']
    sources['age'] = sources['age'].clip(0, 200)
    
    # Calculate health score
    sources['health_score'] = (
        sources['capacity_percent'].fillna(50) * 0.4 + 
        (100 - sources['age'].clip(0, 100).fillna(50)) * 0.3 + 
        30
    ).clip(0, 100)
    
    # Risk classification
    sources['risk_level'] = pd.cut(
        sources['capacity_percent'],
        bins=[0, 30, 60, 100],
        labels=['Critical', 'Moderate', 'Good'],
        include_lowest=True
    )

# Process Alerts - Add reason for alert
if not alerts.empty:
    # Convert alert_time to datetime if it's string
    if 'alert_time' in alerts.columns and alerts['alert_time'].dtype == 'object':
        alerts['alert_time'] = pd.to_datetime(alerts['alert_time'], errors='coerce')
    
    # Add alert reason based on conditions
    def get_alert_reason(row):
        reasons = []
        if row.get('capacity_percent', 100) < 30:
            reasons.append(f"Critical capacity: {row['capacity_percent']:.1f}%")
        elif row.get('capacity_percent', 100) < 60:
            reasons.append(f"Low capacity: {row['capacity_percent']:.1f}%")
        
        if row.get('ph_level', 7) < 6.5:
            reasons.append(f"pH too low: {row['ph_level']}")
        elif row.get('ph_level', 7) > 8.5:
            reasons.append(f"pH too high: {row['ph_level']}")
        
        if 'dissolved_oxygen_mg_l' in row and row['dissolved_oxygen_mg_l'] < 4:
            reasons.append(f"Low dissolved oxygen: {row['dissolved_oxygen_mg_l']} mg/L")
        
        if 'turbidity_ntu' in row and row['turbidity_ntu'] > 5:
            reasons.append(f"High turbidity: {row['turbidity_ntu']} NTU")
        
        if not reasons:
            reasons.append("Monitoring alert - Routine check")
        
        return " | ".join(reasons)
    
    alerts['alert_reason'] = alerts.apply(get_alert_reason, axis=1)
    
    # Determine alert status based on conditions
    def determine_alert_status(row):
        if row.get('capacity_percent', 100) < 30 or row.get('ph_level', 7) < 6 or row.get('ph_level', 7) > 9:
            return 'CRITICAL'
        elif row.get('capacity_percent', 100) < 60 or (row.get('ph_level', 7) < 6.5 or row.get('ph_level', 7) > 8.5):
            return 'WARNING'
        else:
            return 'STABLE'
    
    alerts['alert_status'] = alerts.apply(determine_alert_status, axis=1)

# Add coordinates from monitoring stations to sources
def add_coordinates_to_sources(sources_df, stations_df):
    """Add coordinates from monitoring stations to water sources"""
    
    if sources_df.empty or stations_df.empty:
        return sources_df
    
    df = sources_df.copy()
    
    # Create a mapping of district to first available coordinates
    stations_df['district_clean'] = stations_df['district_name'].str.strip().str.lower()
    
    # Get first coordinates per district
    district_coords = stations_df.groupby('district_clean').agg({
        'latitude': 'first',
        'longitude': 'first'
    }).reset_index()
    
    # Clean source districts for matching
    df['district_clean'] = df['district'].str.strip().str.lower()
    
    # Merge
    df = df.merge(district_coords, on='district_clean', how='left')
    
    # Drop temporary column
    df = df.drop('district_clean', axis=1)
    
    return df

# Apply coordinate mapping
sources = add_coordinates_to_sources(sources, stations)

# Process Groundwater
if not groundwater.empty:
    groundwater['stress_level'] = pd.cut(
        groundwater['avg_depth_meters'],
        bins=[0, 20, 40, 100],
        labels=['Low', 'Moderate', 'High']
    )

# Process Rainfall
if not rainfall.empty:
    rainfall['rainfall_category'] = pd.cut(
        rainfall['rainfall_cm'],
        bins=[0, 50, 150, 300, float('inf')],
        labels=['Low', 'Moderate', 'High', 'Extreme']
    )

# -------------------------
# SIDEBAR FILTERS
# -------------------------

st.sidebar.title("🎮 AQUASTAT")
st.sidebar.caption("Command Interface v2.0")

# Reset button
if st.sidebar.button("🔄 Reset All Filters", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")

# ========== TIME FILTERS ==========
st.sidebar.markdown("### 📅 Time Filters")

# Year filter
if not sources.empty and 'build_year' in sources.columns:
    available_years = sorted(sources['build_year'].dropna().unique())
    if available_years:
        min_year = int(min(available_years))
        max_year = int(max(available_years))
        year_range = st.sidebar.slider(
            "Build Year Range",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year)
        )
    else:
        year_range = (1900, 2025)
else:
    year_range = (1900, 2025)

st.sidebar.markdown("---")

# ========== GEOGRAPHIC FILTERS ==========
st.sidebar.markdown("### 🌍 Geographic Filters")

# State filter - default to "All States"
if not sources.empty and 'state' in sources.columns:
    states = ['All States'] + sorted(sources['state'].dropna().unique().tolist())
    selected_state = st.sidebar.selectbox("State", states, index=0)
else:
    selected_state = "All States"

# District filter based on state
if not sources.empty and 'district' in sources.columns:
    if selected_state != "All States":
        districts = sources[sources['state'] == selected_state]['district'].dropna().unique()
    else:
        districts = sources['district'].dropna().unique()
    
    districts = ['All Districts'] + sorted(districts.tolist())
    selected_district = st.sidebar.selectbox("District", districts, index=0)
else:
    selected_district = "All Districts"

st.sidebar.markdown("---")

# ========== SOURCE FILTERS ==========
st.sidebar.markdown("### 💧 Source Filters")

# Source type - default to "All Types"
if not sources.empty and 'source_type' in sources.columns:
    source_types = ['All Types'] + sorted(sources['source_type'].dropna().unique().tolist())
    selected_type = st.sidebar.selectbox("Source Type", source_types, index=0)
else:
    selected_type = "All Types"

# Capacity range
if not sources.empty and 'capacity_percent' in sources.columns:
    min_cap = float(sources['capacity_percent'].min())
    max_cap = float(sources['capacity_percent'].max())
    capacity_range = st.sidebar.slider(
        "Capacity %",
        min_value=min_cap,
        max_value=max_cap,
        value=(min_cap, max_cap)
    )
else:
    capacity_range = (0, 100)

# Risk level filter
if not sources.empty and 'risk_level' in sources.columns:
    risk_options = ['All Risk Levels'] + list(sources['risk_level'].unique())
    selected_risk = st.sidebar.selectbox("Risk Level", risk_options, index=0)
else:
    selected_risk = "All Risk Levels"

st.sidebar.markdown("---")

# ========== MAP SETTINGS ==========
st.sidebar.markdown("### 🗺️ Map Settings")

map_style = st.sidebar.selectbox(
    "Map Style",
    ["Esri Satellite (Official)", "Dark Matter", "Light Matter"],
    index=0
)

show_heatmap = st.sidebar.checkbox("Show Heatmap", True)
show_clusters = st.sidebar.checkbox("Show Clusters", True)
show_stations = st.sidebar.checkbox("Show Monitoring Stations", True)
marker_size = st.sidebar.slider("Marker Size", 5, 20, 10)

# Show filter stats
st.sidebar.markdown("---")
st.sidebar.caption(f"Total Sources in DB: {len(sources)}")

# ========== APPLY FILTERS ==========

def apply_filters():
    """Apply all filters to data"""
    
    filtered_sources = sources.copy()
    
    # Apply state filter
    if selected_state != "All States":
        filtered_sources = filtered_sources[filtered_sources['state'] == selected_state]
    
    # Apply district filter
    if selected_district != "All Districts":
        filtered_sources = filtered_sources[filtered_sources['district'] == selected_district]
    
    # Apply source type filter
    if selected_type != "All Types":
        filtered_sources = filtered_sources[filtered_sources['source_type'] == selected_type]
    
    # Apply year filter
    if 'build_year' in filtered_sources.columns:
        filtered_sources = filtered_sources[
            (filtered_sources['build_year'] >= year_range[0]) &
            (filtered_sources['build_year'] <= year_range[1])
        ]
    
    # Apply capacity filter
    if 'capacity_percent' in filtered_sources.columns:
        filtered_sources = filtered_sources[
            (filtered_sources['capacity_percent'] >= capacity_range[0]) &
            (filtered_sources['capacity_percent'] <= capacity_range[1])
        ]
    
    # Apply risk filter
    if selected_risk != "All Risk Levels" and 'risk_level' in filtered_sources.columns:
        filtered_sources = filtered_sources[filtered_sources['risk_level'] == selected_risk]
    
    return filtered_sources

filtered_sources = apply_filters()

# Also filter stations based on geographic filters for consistency
def filter_stations():
    """Filter stations based on selected state and district"""
    filtered_stations = stations.copy()
    
    if selected_state != "All States":
        filtered_stations = filtered_stations[filtered_stations['state_name'] == selected_state]
    
    if selected_district != "All Districts":
        filtered_stations = filtered_stations[filtered_stations['district_name'] == selected_district]
    
    return filtered_stations

filtered_stations = filter_stations()

# -------------------------
# MAIN DASHBOARD
# -------------------------

# Header
st.title("💧 AQUASTAT National Water Command Center")
st.caption(f"**Live Intelligence** • Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# KPI Row
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Total Sources",
        f"{len(sources):,}",
        f"{len(sources) - len(filtered_sources)} filtered"
    )

with col2:
    avg_cap = filtered_sources['capacity_percent'].mean() if not filtered_sources.empty else 0
    st.metric("Avg Capacity", f"{avg_cap:.1f}%")

with col3:
    critical = len(filtered_sources[filtered_sources['capacity_percent'] < 30]) if not filtered_sources.empty else 0
    st.metric("Critical Sources", f"{critical}", delta_color="inverse")

with col4:
    sources_with_coords = len(filtered_sources[filtered_sources['latitude'].notna()]) if not filtered_sources.empty else 0
    st.metric("Sources on Map", f"{sources_with_coords}")

with col5:
    if not alerts.empty:
        critical_alerts = len(alerts[alerts['alert_status'] == 'CRITICAL']) if 'alert_status' in alerts.columns else 0
        st.metric("Critical Alerts", f"{critical_alerts}", delta_color="inverse")
    else:
        st.metric("Active Alerts", "0", delta_color="inverse")

st.markdown("---")

# Main Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 DASHBOARD",
    "🗺️ MAP VIEW",
    "📈 ANALYTICS",
    "⚠️ ALERTS",
    "📋 DATA TABLES"
])

# =====================
# TAB 1: DASHBOARD
# =====================

with tab1:
    if filtered_sources.empty:
        st.warning("⚠️ No water sources match the current filters. Try clearing some filters or selecting 'All States'.")
        
        # Show sample of all data
        with st.expander("📋 Show all sources sample"):
            st.dataframe(sources[['source_name', 'source_type', 'state', 'district', 'capacity_percent']].head(20))
    
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Capacity Distribution")
            fig = px.histogram(
                filtered_sources,
                x='capacity_percent',
                nbins=20,
                title=f"Storage Capacity Distribution ({len(filtered_sources)} sources)",
                template="plotly_dark",
                color_discrete_sequence=['#00e5ff']
            )
            fig.update_layout(
                xaxis_title="Capacity (%)",
                yaxis_title="Number of Sources"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🏭 Source Types")
            type_counts = filtered_sources['source_type'].value_counts().reset_index()
            type_counts.columns = ['Source Type', 'Count']
            fig = px.pie(
                type_counts,
                values='Count',
                names='Source Type',
                title=f"Water Sources by Type",
                template="plotly_dark",
                color_discrete_sequence=px.colors.sequential.Tealgrn
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        # Second row
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Groundwater Stress Levels")
            if not groundwater.empty:
                # Filter groundwater based on selected district if applicable
                filtered_gw = groundwater.copy()
                if selected_district != "All Districts":
                    filtered_gw = filtered_gw[filtered_gw['district_name'] == selected_district]
                
                stress_counts = filtered_gw['stress_level'].value_counts().reset_index()
                stress_counts.columns = ['Stress Level', 'Count']
                fig = px.bar(
                    stress_counts,
                    x='Stress Level',
                    y='Count',
                    title="Groundwater Stress Distribution",
                    template="plotly_dark",
                    color='Stress Level',
                    color_discrete_map={
                        'Low': '#00ff9d',
                        'Moderate': '#ffd700',
                        'High': '#ff4444'
                    },
                    text_auto=True
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("ℹ️ No groundwater data available")
        
        with col2:
            st.subheader("☔ Rainfall by Season")
            if not rainfall.empty:
                # Filter rainfall based on selected district if applicable
                filtered_rain = rainfall.copy()
                if selected_district != "All Districts":
                    filtered_rain = filtered_rain[filtered_rain['district_name'] == selected_district]
                
                season_rain = filtered_rain.groupby('season')['rainfall_cm'].mean().reset_index()
                fig = px.bar(
                    season_rain,
                    x='season',
                    y='rainfall_cm',
                    title="Average Rainfall by Season",
                    template="plotly_dark",
                    color='rainfall_cm',
                    color_continuous_scale='Blues',
                    text_auto='.1f'
                )
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("ℹ️ No rainfall data available")
        
        # Third row - Risk distribution
        st.subheader("⚠️ Risk Distribution")
        risk_counts = filtered_sources['risk_level'].value_counts().reset_index()
        risk_counts.columns = ['Risk Level', 'Count']
        
        fig = px.bar(
            risk_counts,
            x='Risk Level',
            y='Count',
            color='Risk Level',
            color_discrete_map={
                'Good': '#00ff9d',
                'Moderate': '#ffd700',
                'Critical': '#ff4444'
            },
            title="Infrastructure Risk Assessment",
            template="plotly_dark",
            text_auto=True
        )
        st.plotly_chart(fig, use_container_width=True)

# =====================
# TAB 2: MAP VIEW (FILTERED SOURCES ONLY)
# =====================

with tab2:
    st.subheader("🗺️ National Interactive Water Resources Map")
    
    # Display current filter info
    filter_info = []
    if selected_state != "All States":
        filter_info.append(f"State: {selected_state}")
    if selected_district != "All Districts":
        filter_info.append(f"District: {selected_district}")
    if selected_type != "All Types":
        filter_info.append(f"Type: {selected_type}")
    
    if filter_info:
        st.info(f"**Showing:** {', '.join(filter_info)} | **Total Sources:** {len(filtered_sources)}")
    
    # 1. Map style mapping
    style_map = {
        "Esri Satellite (Official)": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "Dark Matter": "https://cartodb-basemaps-a.global.ssl.fastly.net/dark_all/{z}/{x}/{y}.png",
        "Light Matter": "https://cartodb-basemaps-a.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png"
    }
    
    # 2. Create map centered on India or selected region
    if not filtered_sources.empty and selected_district != "All Districts":
        # Center on selected district if possible
        district_sources = filtered_sources[filtered_sources['latitude'].notna()]
        if not district_sources.empty:
            center_lat = district_sources['latitude'].mean()
            center_lon = district_sources['longitude'].mean()
            zoom = 9
        else:
            center_lat, center_lon, zoom = 20.5937, 78.9629, 5
    elif not filtered_sources.empty and selected_state != "All States":
        # Center on selected state
        state_sources = filtered_sources[filtered_sources['latitude'].notna()]
        if not state_sources.empty:
            center_lat = state_sources['latitude'].mean()
            center_lon = state_sources['longitude'].mean()
            zoom = 7
        else:
            center_lat, center_lon, zoom = 20.5937, 78.9629, 5
    else:
        center_lat, center_lon, zoom = 20.5937, 78.9629, 5
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles=style_map[map_style],
        attr='AQUASTAT | Data: Esri, OSM, Survey of India'
    )
    
    # Add fullscreen control
    Fullscreen().add_to(m)
    
    # 3. Initialize Marker Layer
    if show_clusters and len(filtered_sources) > 10:
        marker_cluster = MarkerCluster().add_to(m)
    else:
        marker_cluster = m
    
    heat_data = []
    sources_on_map = 0
    
    # 4. Process ONLY filtered water sources with coordinates
    sources_with_coords = filtered_sources[
        filtered_sources['latitude'].notna() & 
        filtered_sources['longitude'].notna()
    ]
    
    if not sources_with_coords.empty:
        for _, source in sources_with_coords.iterrows():
            # Determine color and risk text based on capacity
            if source['capacity_percent'] < 30:
                color = '#ff4444'  # Critical
                risk_text = "CRITICAL"
                status_icon = "🔴"
            elif source['capacity_percent'] < 60:
                color = '#ffd700'  # Moderate
                risk_text = "MODERATE"
                status_icon = "🟡"
            else:
                color = '#00ff9d'  # Good
                risk_text = "GOOD"
                status_icon = "🟢"
            
            heat_data.append([source['latitude'], source['longitude']])
            sources_on_map += 1
            
            # Get alert info for this source if any
            source_alerts = alerts[alerts['source_name'] == source['source_name']] if not alerts.empty else pd.DataFrame()
            alert_info = ""
            if not source_alerts.empty:
                latest_alert = source_alerts.iloc[-1]
                alert_info = f"""
                <tr>
                    <td colspan="2" style="padding-top: 10px;">
                        <div style="background: rgba(255,0,0,0.1); padding: 5px; border-radius: 5px;">
                            <b>⚠️ Active Alert:</b> {latest_alert.get('alert_reason', 'Unknown')}<br>
                            <small>Status: {latest_alert.get('alert_status', 'Unknown')}</small>
                        </div>
                    </td>
                </tr>
                """
            
            popup_html = f"""
            <div style="font-family: Arial; min-width: 300px; background: #0a0f1e; color: white; padding: 15px; border-radius: 10px; border-left: 5px solid {color};">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <h4 style="color: {color}; margin:0;">{status_icon} {source['source_name']}</h4>
                    <span style="background: {color}; color: black; padding: 3px 8px; border-radius: 12px; font-size: 0.8rem; font-weight: bold;">{risk_text}</span>
                </div>
                <hr style="margin:10px 0; border-color: #1f2937;">
                <table style="width:100%; border-collapse: collapse;">
                    <tr><td style="padding: 5px 0;"><b>📍 Type:</b></td><td>{source['source_type']}</td></tr>
                    <tr><td style="padding: 5px 0;"><b>🏛️ District:</b></td><td>{source['district']}</td></tr>
                    <tr><td style="padding: 5px 0;"><b>🗺️ State:</b></td><td>{source['state']}</td></tr>
                    <tr><td style="padding: 5px 0;"><b>📊 Capacity:</b></td>
                        <td>
                            {source['capacity_percent']:.1f}%
                            <div style="background: #1f2937; height: 6px; width: 100px; border-radius: 3px; margin-top: 3px;">
                                <div style="background: {color}; width: {source['capacity_percent']}%; height: 6px; border-radius: 3px;"></div>
                            </div>
                        </td>
                    </tr>
                    <tr><td style="padding: 5px 0;"><b>📅 Age:</b></td><td>{source['age']:.0f} years</td></tr>
                    <tr><td style="padding: 5px 0;"><b>💯 Health Score:</b></td><td>{source['health_score']:.1f}%</td></tr>
                    <tr><td style="padding: 5px 0;"><b>🏭 Build Year:</b></td><td>{source['build_year']:.0f}</td></tr>
                    {alert_info}
                </table>
                <div style="margin-top: 10px; font-size: 0.8rem; color: #8892b0; text-align: center;">
                    Click to view details | Last updated: {datetime.now().strftime('%H:%M:%S')}
                </div>
            </div>
            """
            
            # Enhanced tooltip with more info
            tooltip_text = f"{source['source_name']} | {source['capacity_percent']:.0f}% capacity | {risk_text}"
            
            # Create circle marker
            marker = folium.CircleMarker(
                location=[source['latitude'], source['longitude']],
                radius=marker_size + (3 if source['capacity_percent'] < 30 else 0),
                color=color,
                fill=True,
                fillOpacity=0.7,
                popup=folium.Popup(popup_html, max_width=350),
                tooltip=tooltip_text
            )
            
            if show_clusters and len(filtered_sources) > 10:
                marker.add_to(marker_cluster)
            else:
                marker.add_to(m)
        
        # 5. Add heatmap if enabled (only for filtered sources)
        if show_heatmap and heat_data:
            HeatMap(
                heat_data,
                radius=15,
                blur=10,
                gradient={0.2: 'blue', 0.4: 'cyan', 0.6: 'lime', 0.8: 'yellow', 1: 'red'}
            ).add_to(m)
    
    # 6. Add monitoring stations if enabled (filtered by geographic selections)
    if show_stations and not filtered_stations.empty:
        stations_with_coords = filtered_stations[
            filtered_stations['latitude'].notna() & 
            filtered_stations['longitude'].notna()
        ]
        for _, station in stations_with_coords.iterrows():
            # Color based on status
            if station['status'] == 'Active':
                station_color = 'green'
                status_icon = "✅"
            elif station['status'] == 'Maintenance':
                station_color = 'orange'
                status_icon = "🔄"
            else:
                station_color = 'red'
                status_icon = "⚠️"
            
            # Get water quality status
            water_quality = "Good"
            quality_color = "#00ff9d"
            if station['ph_level'] < 6.5 or station['ph_level'] > 8.5:
                water_quality = "Poor"
                quality_color = "#ff4444"
            elif station['dissolved_oxygen_mg_l'] < 4:
                water_quality = "Fair"
                quality_color = "#ffd700"
            
            station_popup = f"""
            <div style="font-family: Arial; min-width: 280px; background: #0a0f1e; color: white; padding: 15px; border-radius: 10px; border-left: 5px solid {station_color};">
                <h4 style="margin:0 0 10px 0; color: white;">{status_icon} {station['station_name']}</h4>
                <hr style="margin:10px 0; border-color: #1f2937;">
                <table style="width:100%; border-collapse: collapse;">
                    <tr><td><b>📍 District:</b></td><td>{station['district_name']}</td></tr>
                    <tr><td><b>📊 Status:</b></td><td><span style="color: {station_color};">{station['status']}</span></td></tr>
                    <tr><td><b>🧪 pH Level:</b></td><td>{station['ph_level']}</td></tr>
                    <tr><td><b>💨 DO (mg/L):</b></td><td>{station['dissolved_oxygen_mg_l']}</td></tr>
                    <tr><td><b>🌫️ Turbidity (NTU):</b></td><td>{station['turbidity_ntu']}</td></tr>
                    <tr><td><b>💧 Quality:</b></td><td><span style="color: {quality_color};">{water_quality}</span></td></tr>
                </table>
            </div>
            """
            
            folium.Marker(
                location=[station['latitude'], station['longitude']],
                icon=folium.Icon(color=station_color, icon='info-sign', prefix='glyphicon'),
                popup=folium.Popup(station_popup, max_width=300),
                tooltip=f"Station: {station['station_name']} | {station['status']}"
            ).add_to(m)
    
    # 7. Display map
    st_folium(m, width=1300, height=600)
    
    # 8. Map statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Sources on Map", sources_on_map)
    with col2:
        st.metric("Total Filtered Sources", len(filtered_sources))
        with col3:
        coverage = (sources_on_map/len(filtered_sources)*100) if len(filtered_sources) > 0 else 0
        st.metric("Coordinate Coverage", f"{coverage:.1f}%")
    
    # 9. Legend
    st.markdown("---")
    cols = st.columns(5)
    with cols[0]:
        st.markdown("🟢 **Good** (≥60%)")
    with cols[1]:
        st.markdown("🟡 **Moderate** (30-60%)")
    with cols[2]:
        st.markdown("🔴 **Critical** (<30%)")
    with cols[3]:
        st.markdown("🔵 **Monitoring Station**")
    with cols[4]:
        st.markdown("🔥 **Heatmap Area**")

# =====================
# TAB 3: ANALYTICS
# =====================

with tab3:
    st.subheader("📈 Advanced Analytics")
    
    # Create sub-tabs
    atab1, atab2, atab3 = st.tabs(["📊 Trends", "📉 Comparisons", "📐 Statistics"])
    
    with atab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Rainfall Trend")
            if not rainfall.empty:
                # Filter rainfall based on selected district if applicable
                filtered_rain = rainfall.copy()
                if selected_district != "All Districts":
                    filtered_rain = filtered_rain[filtered_rain['district_name'] == selected_district]
                
                rain_trend = filtered_rain.groupby('record_year')['rainfall_cm'].mean().reset_index()
                fig = px.line(
                    rain_trend,
                    x='record_year',
                    y='rainfall_cm',
                    title="Average Rainfall Over Years",
                    template="plotly_dark",
                    markers=True
                )
                fig.update_traces(line_color='#00e5ff', line_width=3)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No rainfall data available")
        
        with col2:
            st.subheader("Groundwater Trend")
            if not groundwater.empty and 'assessment_year' in groundwater.columns:
                # Filter groundwater based on selected district if applicable
                filtered_gw = groundwater.copy()
                if selected_district != "All Districts":
                    filtered_gw = filtered_gw[filtered_gw['district_name'] == selected_district]
                
                gw_trend = filtered_gw.groupby('assessment_year')['avg_depth_meters'].mean().reset_index()
                fig = px.line(
                    gw_trend,
                    x='assessment_year',
                    y='avg_depth_meters',
                    title="Average Groundwater Depth",
                    template="plotly_dark",
                    markers=True
                )
                fig.update_traces(line_color='#ffd700', line_width=3)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No groundwater data available")
    
    with atab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Capacity by State")
            if not filtered_sources.empty and 'state' in filtered_sources.columns:
                state_cap = filtered_sources.groupby('state')['capacity_percent'].mean().sort_values(ascending=False)
                if len(state_cap) > 10:
                    state_cap = state_cap.head(10)
                
                fig = px.bar(
                    x=state_cap.values,
                    y=state_cap.index,
                    orientation='h',
                    title=f"Average Capacity by State ({len(state_cap)} states)",
                    template="plotly_dark",
                    color=state_cap.values,
                    color_continuous_scale='Tealgrn',
                    labels={'x': 'Avg Capacity (%)', 'y': 'State'}
                )
                fig.update_traces(texttemplate='%{x:.1f}%', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No state data available")
        
        with col2:
            st.subheader("Extraction vs Recharge")
            if not groundwater.empty:
                # Filter groundwater based on selected district if applicable
                filtered_gw = groundwater.copy()
                if selected_district != "All Districts":
                    filtered_gw = filtered_gw[filtered_gw['district_name'] == selected_district]
                
                fig = px.scatter(
                    filtered_gw,
                    x='recharge_rate_mcm',
                    y='extraction_pct',
                    size='avg_depth_meters',
                    color='district_name',
                    title="Groundwater Extraction vs Recharge Rate",
                    template="plotly_dark",
                    labels={
                        'recharge_rate_mcm': 'Recharge Rate (MCM)',
                        'extraction_pct': 'Extraction %',
                        'avg_depth_meters': 'Depth (m)'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No groundwater data available")
    
    with atab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Statistical Summary")
            if not filtered_sources.empty:
                stats_df = filtered_sources[['capacity_percent', 'age', 'health_score']].describe()
                st.dataframe(stats_df.style.format("{:.2f}"), use_container_width=True)
            else:
                st.info("No source data available")
        
        with col2:
            st.subheader("Correlation Matrix")
            if not filtered_sources.empty and not groundwater.empty:
                # Merge for correlation
                merged = filtered_sources.merge(
                    groundwater,
                    left_on='district',
                    right_on='district_name',
                    how='inner'
                )
                
                if not merged.empty:
                    numeric_cols = ['capacity_percent', 'age', 'avg_depth_meters', 'extraction_pct', 'recharge_rate_mcm']
                    corr_data = merged[numeric_cols].dropna()
                    
                    if not corr_data.empty:
                        corr_matrix = corr_data.corr()
                        
                        fig = px.imshow(
                            corr_matrix,
                            text_auto=True,
                            aspect="auto",
                            title="Feature Correlation Matrix",
                            template="plotly_dark",
                            color_continuous_scale='RdBu_r',
                            labels=dict(color="Correlation")
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Insufficient data for correlation")
                else:
                    st.info("No matching data for correlation")
            else:
                st.info("Insufficient data for correlation")

# =====================
# TAB 4: ALERTS (FIXED VERSION)
# =====================

with tab4:
    st.subheader("🚨 Active Alerts and Warnings")
    
    if not alerts.empty:
        # FIRST, MERGE ALERTS WITH SOURCES TO GET PROPER LOCATION INFO
        alerts_with_source = alerts.copy()
        
        if not sources.empty and 'source_name' in sources.columns:
            # Get only needed columns from sources
            source_cols = ['source_name', 'source_type', 'district', 'state']
            available_source_cols = [col for col in source_cols if col in sources.columns]
            
            if available_source_cols:
                # Merge alerts with source information
                alerts_with_source = alerts.merge(
                    sources[available_source_cols],
                    on='source_name',
                    how='left'
                )
        
        # Count alerts by status
        alert_counts = alerts_with_source['alert_status'].value_counts()
        
        # Create metrics for each alert type
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            critical_count = alert_counts.get('CRITICAL', 0)
            st.metric("🔴 CRITICAL", critical_count, 
                     delta="Immediate action required" if critical_count > 0 else None,
                     delta_color="inverse")
        with col2:
            warning_count = alert_counts.get('WARNING', 0)
            st.metric("🟡 WARNING", warning_count,
                     delta="Monitor closely" if warning_count > 0 else None)
        with col3:
            stable_count = alert_counts.get('STABLE', 0)
            st.metric("🟢 STABLE", stable_count)
        with col4:
            st.metric("📊 TOTAL", len(alerts_with_source))
        
        st.markdown("---")
        
        # Filter alerts based on selected filters
        filtered_alerts = alerts_with_source.copy()
        
        if selected_state != "All States" and 'state' in filtered_alerts.columns:
            filtered_alerts = filtered_alerts[filtered_alerts['state'] == selected_state]
        
        if selected_district != "All Districts" and 'district' in filtered_alerts.columns:
            filtered_alerts = filtered_alerts[filtered_alerts['district'] == selected_district]
        
        if selected_type != "All Types" and 'source_type' in filtered_alerts.columns:
            filtered_alerts = filtered_alerts[filtered_alerts['source_type'] == selected_type]
        
        # Add alert type selector
        alert_type_filter = st.selectbox(
            "Filter by Alert Status",
            ["All Alerts", "CRITICAL", "WARNING", "STABLE"],
            index=0
        )
        
        if alert_type_filter != "All Alerts":
            filtered_alerts = filtered_alerts[filtered_alerts['alert_status'] == alert_type_filter]
        
        if filtered_alerts.empty:
            st.info(f"ℹ️ No {alert_type_filter.lower()} alerts match the current filters")
        else:
            # Sort alerts by severity
            severity_order = {'CRITICAL': 0, 'WARNING': 1, 'STABLE': 2}
            filtered_alerts['severity_rank'] = filtered_alerts['alert_status'].map(severity_order)
            filtered_alerts = filtered_alerts.sort_values('severity_rank').drop('severity_rank', axis=1)
            
            for _, alert in filtered_alerts.iterrows():
                if alert['alert_status'] == 'CRITICAL':
                    status_class = "status-critical"
                    border_color = "#ff4444"
                    icon = "🔴"
                    severity_text = "IMMEDIATE ACTION REQUIRED"
                elif alert['alert_status'] == 'WARNING':
                    status_class = "status-warning"
                    border_color = "#ffd700"
                    icon = "🟡"
                    severity_text = "MONITOR CLOSELY"
                else:
                    status_class = "status-good"
                    border_color = "#00ff9d"
                    icon = "🟢"
                    severity_text = "NORMAL OPERATIONS"
                
                # Get source information (now available from merge)
                source_type = alert.get('source_type', 'Unknown')
                district = alert.get('district', 'Unknown')
                state = alert.get('state', 'Unknown')
                
                # Format location string properly
                if district != 'Unknown' and state != 'Unknown':
                    location_str = f"{district}, {state}"
                elif district != 'Unknown':
                    location_str = f"{district}"
                elif state != 'Unknown':
                    location_str = f"{state}"
                else:
                    location_str = "Location unknown"
                
                # Format alert time
                alert_time = alert['alert_time']
                if isinstance(alert_time, pd.Timestamp):
                    time_str = alert_time.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    time_str = str(alert_time)
                
                st.markdown(f"""
                <div class="info-card" style="border-left: 5px solid {border_color};">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h3 style="margin:0;">{icon} {alert['source_name']}</h3>
                            <p style="color: #8892b0; margin:0;">
                                {source_type} | {location_str}
                            </p>
                        </div>
                        <div style="text-align: right;">
                            <span class="{status_class}" style="font-size: 1.2rem; display: block;">{alert['alert_status']}</span>
                            <small style="color: #8892b0;">{severity_text}</small>
                        </div>
                    </div>
                    <hr style="margin:10px 0; border-color: {border_color};">
                    
                    <!-- Alert Reason Section -->
                    <div class="alert-reason" style="background: rgba({','.join(str(int(border_color.lstrip('#')[i:i+2], 16)) for i in (0, 2, 4))}, 0.1); padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                        <strong style="color: {border_color};">⚠️ Alert Reason:</strong>
                        <p style="margin: 5px 0 0 0; color: white;">{alert['alert_reason']}</p>
                    </div>
                    
                    <div style="display: flex; gap: 30px; flex-wrap: wrap;">
                        <div style="flex: 1; min-width: 200px;">
                            <p><strong>📊 Current Capacity:</strong> {alert['capacity_percent']}%</p>
                            <div style="background: #1f2937; height: 10px; width: 100%; border-radius: 5px;">
                                <div style="background: {border_color}; width: {alert['capacity_percent']}%; height: 10px; border-radius: 5px;"></div>
                            </div>
                        </div>
                        <div>
                            <p><strong>🧪 pH Level:</strong> {alert.get('ph_level', 'N/A')}</p>
                            <p><strong>⏰ Alert Time:</strong> {time_str}</p>
                        </div>
                    </div>
                    
                    <!-- Additional Metrics -->
                    <div style="display: flex; gap: 20px; margin-top: 15px; padding-top: 10px; border-top: 1px solid #1f2937;">
                        <div>
                            <small style="color: #8892b0;">Dissolved Oxygen</small>
                            <p style="margin:0;"><strong>{alert.get('dissolved_oxygen_mg_l', 'N/A')} mg/L</strong></p>
                        </div>
                        <div>
                            <small style="color: #8892b0;">Turbidity</small>
                            <p style="margin:0;"><strong>{alert.get('turbidity_ntu', 'N/A')} NTU</strong></p>
                        </div>
                        <div>
                            <small style="color: #8892b0;">Temperature</small>
                            <p style="margin:0;"><strong>{alert.get('temperature_c', 'N/A')} °C</strong></p>
                        </div>
                    </div>
                    
                    <!-- Action Button -->
                    <div style="margin-top: 15px;">
                        <a href="#" style="background: {border_color}; color: black; padding: 5px 15px; border-radius: 5px; text-decoration: none; font-weight: bold; display: inline-block;" onclick="alert('View details for {alert['source_name']}')">View Details →</a>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.success("✅ No active alerts - All systems normal")
        st.balloons()

# =====================
# TAB 5: DATA TABLES
# =====================

with tab5:
    st.subheader("📋 Data Explorer")
    
    table_choice = st.selectbox(
        "Select Table to View",
        ["Water Sources", "Monitoring Stations", "Groundwater Levels", 
         "Rainfall History", "Water Usage", "Active Alerts", "Regional Statistics"]
    )
    
    if table_choice == "Water Sources":
        # Check which columns actually exist to avoid KeyError
        available_cols = []
        desired_cols = ['source_name', 'source_type', 'capacity_percent', 'max_capacity_mcm', 
                       'build_year', 'age', 'state', 'district', 'origin_state', 'is_transboundary', 'risk_level']
        
        for col in desired_cols:
            if col in filtered_sources.columns:
                available_cols.append(col)
        
        if available_cols:
            st.dataframe(
                filtered_sources[available_cols], 
                use_container_width=True, 
                hide_index=True
            )
        else:
            st.dataframe(filtered_sources, use_container_width=True, hide_index=True)
            st.warning("Expected columns not found. Showing all available data.")
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Sources", len(filtered_sources))
        with col2:
            if not filtered_sources.empty and 'capacity_percent' in filtered_sources.columns:
                st.metric("Avg Capacity", f"{filtered_sources['capacity_percent'].mean():.1f}%")
            else:
                st.metric("Avg Capacity", "N/A")
        with col3:
            if not filtered_sources.empty and 'is_transboundary' in filtered_sources.columns:
                st.metric("Transboundary", len(filtered_sources[filtered_sources['is_transboundary'] == 1]))
            else:
                st.metric("Transboundary", "N/A")
        
        if not filtered_sources.empty:
            csv = filtered_sources.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download CSV",
                csv,
                f"water_sources_{selected_state}_{selected_district}_{selected_type}.csv",
                "text/csv",
                use_container_width=True
            )
    
    elif table_choice == "Monitoring Stations":
        # Filter stations based on selections
        display_stations = stations.copy()
        if selected_state != "All States" and 'state_name' in display_stations.columns:
            display_stations = display_stations[display_stations['state_name'] == selected_state]
        if selected_district != "All Districts" and 'district_name' in display_stations.columns:
            display_stations = display_stations[display_stations['district_name'] == selected_district]
        
        # Check which columns exist
        available_cols = []
        desired_cols = ['station_name', 'state_name', 'district_name', 'latitude', 
                       'longitude', 'ph_level', 'dissolved_oxygen_mg_l', 'turbidity_ntu', 'status']
        
        for col in desired_cols:
            if col in display_stations.columns:
                available_cols.append(col)
        
        if available_cols:
            st.dataframe(
                display_stations[available_cols],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.dataframe(display_stations, use_container_width=True, hide_index=True)
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Stations", len(display_stations))
        with col2:
            if 'status' in display_stations.columns:
                active = len(display_stations[display_stations['status'] == 'Active'])
                st.metric("Active Stations", active)
            else:
                st.metric("Active Stations", "N/A")
        with col3:
            if 'status' in display_stations.columns:
                maintenance = len(display_stations[display_stations['status'] == 'Maintenance'])
                st.metric("Maintenance", maintenance)
            else:
                st.metric("Maintenance", "N/A")
        
        if not display_stations.empty:
            csv = display_stations.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download CSV",
                csv,
                f"monitoring_stations_{selected_state}_{selected_district}.csv",
                "text/csv",
                use_container_width=True
            )
    
    elif table_choice == "Groundwater Levels":
        # Filter groundwater based on selections
        display_gw = groundwater.copy()
        if selected_district != "All Districts" and 'district_name' in display_gw.columns:
            display_gw = display_gw[display_gw['district_name'] == selected_district]
        
        # Check which columns exist
        available_cols = []
        desired_cols = ['district_name', 'avg_depth_meters', 'extraction_pct',
                       'recharge_rate_mcm', 'assessment_year', 'stress_level']
        
        for col in desired_cols:
            if col in display_gw.columns:
                available_cols.append(col)
        
        if available_cols:
            st.dataframe(
                display_gw[available_cols],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.dataframe(display_gw, use_container_width=True, hide_index=True)
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Districts", len(display_gw))
        with col2:
            if not display_gw.empty and 'avg_depth_meters' in display_gw.columns:
                avg_depth = display_gw['avg_depth_meters'].mean()
                st.metric("Avg Depth", f"{avg_depth:.1f} m")
            else:
                st.metric("Avg Depth", "N/A")
        with col3:
            if not display_gw.empty and 'stress_level' in display_gw.columns:
                high_stress = len(display_gw[display_gw['stress_level'] == 'High'])
                st.metric("High Stress", high_stress)
            else:
                st.metric("High Stress", "N/A")
        
        if not display_gw.empty:
            csv = display_gw.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download CSV",
                csv,
                f"groundwater_{selected_district}.csv",
                "text/csv",
                use_container_width=True
            )
    
    elif table_choice == "Rainfall History":
        # Filter rainfall based on selections
        display_rain = rainfall.copy()
        if selected_district != "All Districts" and 'district_name' in display_rain.columns:
            display_rain = display_rain[display_rain['district_name'] == selected_district]
        
        # Check which columns exist
        available_cols = []
        desired_cols = ['district_name', 'rainfall_cm', 'record_year', 'season', 'rainfall_category']
        
        for col in desired_cols:
            if col in display_rain.columns:
                available_cols.append(col)
        
        if available_cols:
            st.dataframe(
                display_rain[available_cols],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.dataframe(display_rain, use_container_width=True, hide_index=True)
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(display_rain))
        with col2:
            if not display_rain.empty and 'rainfall_cm' in display_rain.columns:
                avg_rain = display_rain['rainfall_cm'].mean()
                st.metric("Avg Rainfall", f"{avg_rain:.1f} cm")
            else:
                st.metric("Avg Rainfall", "N/A")
        with col3:
            if not display_rain.empty and 'record_year' in display_rain.columns:
                years = display_rain['record_year'].nunique()
                st.metric("Years of Data", years)
            else:
                st.metric("Years of Data", "N/A")
        
        if not display_rain.empty:
            csv = display_rain.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download CSV",
                csv,
                f"rainfall_{selected_district}.csv",
                "text/csv",
                use_container_width=True
            )
    
    elif table_choice == "Water Usage":
        # Filter usage based on selections
        display_usage = usage.copy()
        if selected_state != "All States" and 'state' in display_usage.columns:
            display_usage = display_usage[display_usage['state'] == selected_state]
        if selected_district != "All Districts" and 'district' in display_usage.columns:
            display_usage = display_usage[display_usage['district'] == selected_district]
        if selected_type != "All Types" and 'source_type' in display_usage.columns:
            display_usage = display_usage[display_usage['source_type'] == selected_type]
        
        # Check which columns exist
        available_cols = []
        desired_cols = ['source_name', 'source_type', 'sector', 'sub_sector',
                       'consumer_name', 'consumption_mcm', 'record_year',
                       'season', 'state', 'district']
        
        for col in desired_cols:
            if col in display_usage.columns:
                available_cols.append(col)
        
        if available_cols:
            st.dataframe(
                display_usage[available_cols],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.dataframe(display_usage, use_container_width=True, hide_index=True)
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(display_usage))
        with col2:
            if not display_usage.empty and 'consumption_mcm' in display_usage.columns:
                total_consumption = display_usage['consumption_mcm'].sum()
                st.metric("Total Consumption", f"{total_consumption:.1f} MCM")
            else:
                st.metric("Total Consumption", "N/A")
        with col3:
            if not display_usage.empty and 'consumption_mcm' in display_usage.columns:
                avg_consumption = display_usage['consumption_mcm'].mean()
                st.metric("Avg Consumption", f"{avg_consumption:.1f} MCM")
            else:
                st.metric("Avg Consumption", "N/A")
        
        if not display_usage.empty:
            csv = display_usage.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download CSV",
                csv,
                f"water_usage_{selected_state}_{selected_district}.csv",
                "text/csv",
                use_container_width=True
            )
    
    elif table_choice == "Active Alerts":
        # Filter alerts based on selections
        display_alerts = alerts.copy()
        
        # Merge with sources to get location info for display
        if not sources.empty and 'source_name' in sources.columns:
            source_cols = ['source_name', 'source_type', 'district', 'state']
            available_source_cols = [col for col in source_cols if col in sources.columns]
            
            if available_source_cols:
                display_alerts = display_alerts.merge(
                    sources[available_source_cols],
                    on='source_name',
                    how='left'
                )
        
        if selected_state != "All States" and 'state' in display_alerts.columns:
            display_alerts = display_alerts[display_alerts['state'] == selected_state]
        if selected_district != "All Districts" and 'district' in display_alerts.columns:
            display_alerts = display_alerts[display_alerts['district'] == selected_district]
        if selected_type != "All Types" and 'source_type' in display_alerts.columns:
            display_alerts = display_alerts[display_alerts['source_type'] == selected_type]
        
        # Check which columns exist
        available_cols = []
        desired_cols = ['source_name', 'source_type', 'district', 'state', 'capacity_percent', 
                       'ph_level', 'alert_status', 'alert_time', 'alert_reason']
        
        for col in desired_cols:
            if col in display_alerts.columns:
                available_cols.append(col)
        
        if available_cols:
            # Format alert_time for display
            display_copy = display_alerts[available_cols].copy()
            if 'alert_time' in display_copy.columns and pd.api.types.is_datetime64_any_dtype(display_copy['alert_time']):
                display_copy['alert_time'] = display_copy['alert_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            st.dataframe(display_copy, use_container_width=True, hide_index=True)
        else:
            st.dataframe(display_alerts, use_container_width=True, hide_index=True)
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Alerts", len(display_alerts))
        with col2:
            if not display_alerts.empty and 'alert_status' in display_alerts.columns:
                critical = len(display_alerts[display_alerts['alert_status'] == 'CRITICAL'])
                st.metric("🔴 Critical", critical)
            else:
                st.metric("🔴 Critical", "N/A")
        with col3:
            if not display_alerts.empty and 'alert_status' in display_alerts.columns:
                warning = len(display_alerts[display_alerts['alert_status'] == 'WARNING'])
                st.metric("🟡 Warning", warning)
            else:
                st.metric("🟡 Warning", "N/A")
        
        # Add stable count as additional metric
        if not display_alerts.empty and 'alert_status' in display_alerts.columns:
            stable = len(display_alerts[display_alerts['alert_status'] == 'STABLE'])
            st.metric("🟢 Stable", stable)
        
        if not display_alerts.empty:
            csv = display_alerts.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download CSV",
                csv,
                f"active_alerts_{selected_state}_{selected_district}.csv",
                "text/csv",
                use_container_width=True
            )
    
    elif table_choice == "Regional Statistics":
        # Check which columns exist
        available_cols = []
        desired_cols = ['region_name', 'population_count', 'annual_rainfall_avg_cm']
        
        for col in desired_cols:
            if col in regional.columns:
                available_cols.append(col)
        
        if available_cols:
            st.dataframe(
                regional[available_cols],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.dataframe(regional, use_container_width=True, hide_index=True)
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Regions", len(regional))
        with col2:
            if not regional.empty and 'population_count' in regional.columns:
                total_pop = regional['population_count'].sum()
                st.metric("Total Population", f"{total_pop:,}")
            else:
                st.metric("Total Population", "N/A")
        with col3:
            if not regional.empty and 'annual_rainfall_avg_cm' in regional.columns:
                avg_rain = regional['annual_rainfall_avg_cm'].mean()
                st.metric("Avg Rainfall", f"{avg_rain:.1f} cm")
            else:
                st.metric("Avg Rainfall", "N/A")
        
        if not regional.empty:
            csv = regional.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download CSV",
                csv,
                "regional_statistics.csv",
                "text/csv",
                use_container_width=True
            )

# =====================
# SIDEBAR FILTER SUMMARY
# =====================

with st.sidebar.expander("📊 Current Filter Summary", expanded=False):
    st.markdown(f"""
    **Time Range:** {year_range[0]} - {year_range[1]}
    
    **Geography:**
    - State: {selected_state}
    - District: {selected_district}
    
    **Source Filters:**
    - Type: {selected_type}
    - Capacity: {capacity_range[0]}% - {capacity_range[1]}%
    - Risk: {selected_risk}
    
    **Results:**
    - Sources: {len(filtered_sources)} of {len(sources)}
    - On Map: {len(filtered_sources[filtered_sources['latitude'].notna()]) if not filtered_sources.empty else 0}
    """)

# =====================
# EXPORT ALL FILTERED DATA
# =====================

st.sidebar.markdown("---")
if st.sidebar.button("📦 Export All Filtered Data", use_container_width=True):
    # Create a dictionary of all filtered datasets
    export_data = {
        'water_sources': filtered_sources,
        'monitoring_stations': filtered_stations,
        'groundwater': groundwater[groundwater['district_name'].isin(filtered_sources['district'].unique())] if not filtered_sources.empty and 'district_name' in groundwater.columns and 'district' in filtered_sources.columns else pd.DataFrame(),
        'rainfall': rainfall[rainfall['district_name'].isin(filtered_sources['district'].unique())] if not filtered_sources.empty and 'district_name' in rainfall.columns and 'district' in filtered_sources.columns else pd.DataFrame(),
        'usage': usage[usage['source_id'].isin(filtered_sources['source_id'])] if not filtered_sources.empty and 'source_id' in usage.columns and 'source_id' in filtered_sources.columns else pd.DataFrame(),
        'alerts': alerts[alerts['source_name'].isin(filtered_sources['source_name'])] if not filtered_sources.empty and 'source_name' in alerts.columns and 'source_name' in filtered_sources.columns else pd.DataFrame()
    }
    
    # Create Excel file with multiple sheets
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, df in export_data.items():
            if not df.empty:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    st.sidebar.download_button(
        "📥 Download Excel Report",
        output.getvalue(),
        f"aquastat_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# =====================
# FOOTER
# =====================

st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style="text-align: center;">
        <p style="color: #00e5ff; font-size: 1.2rem;">💧 AQUASTAT</p>
        <p style="color: #8892b0;">National Water Command Center</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style="text-align: center;">
        <p style="color: #8892b0;">Data Source: Ministry of Jal Shakti</p>
        <p style="color: #8892b0;">Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="text-align: center;">
        <p style="color: #8892b0;">© 2025 All Rights Reserved</p>
        <p style="color: #8892b0;">Version 3.0 | For Official Use</p>
    </div>
    """, unsafe_allow_html=True)

# Add a small note about data freshness
st.markdown("""
<div style="position: fixed; bottom: 10px; right: 10px; background: rgba(0,229,255,0.1); padding: 5px 10px; border-radius: 5px; font-size: 0.8rem;">
    🔄 Data refreshes every 5 minutes
</div>
""", unsafe_allow_html=True)
