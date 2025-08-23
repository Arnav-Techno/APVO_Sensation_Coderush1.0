import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from pathlib import Path

# Try to import HeatMap, use fallback if not available
try:
    from folium.plugins import HeatMap
    HEATMAP_AVAILABLE = True
except ImportError:
    HEATMAP_AVAILABLE = False

# Configure page
st.set_page_config(
    page_title="🌍 Global Environmental Data Explorer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f8f0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2E8B57;
    }
    .info-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .stSelectbox > div > div > div > div {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

def load_sample_data():
    """Load sample satellite data for demonstration"""
    
    # Check if our generated data exists
    data_dir = Path("./global_data")
    if data_dir.exists():
        csv_files = list(data_dir.glob("*.csv"))
        if csv_files:
            # Load and combine all CSV files
            all_data = []
            for csv_file in csv_files:
                df = pd.read_csv(csv_file)
                all_data.append(df)
            
            # Combine all dataframes
            if len(all_data) == 1:
                combined_df = all_data[0]
            else:
                # Merge on common columns
                combined_df = all_data[0]
                for df in all_data[1:]:
                    combined_df = pd.merge(combined_df, df, on=['longitude', 'latitude'], how='outer', suffixes=('', '_y'))
                    # Drop duplicate columns
                    combined_df = combined_df.loc[:,~combined_df.columns.str.endswith('_y')]
            
            return combined_df, "Generated Satellite Data"
    
    # Create sample data if no real data available
    np.random.seed(42)
    
    # Create a grid of points around major cities
    cities = [
        {"name": "New York", "lat": 40.7128, "lon": -74.0060},
        {"name": "London", "lat": 51.5074, "lon": -0.1278},
        {"name": "Tokyo", "lat": 35.6762, "lon": 139.6503},
        {"name": "Delhi", "lat": 28.7041, "lon": 77.1025},
        {"name": "Beijing", "lat": 39.9042, "lon": 116.4074},
        {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777},
        {"name": "São Paulo", "lat": -23.5558, "lon": -46.6396},
        {"name": "Cairo", "lat": 30.0444, "lon": 31.2357},
        {"name": "Sydney", "lat": -33.8688, "lon": 151.2093},
        {"name": "Lagos", "lat": 6.5244, "lon": 3.3792}
    ]
    
    data = []
    for city in cities:
        # Generate points around each city
        for i in range(50):
            lat_offset = np.random.normal(0, 0.5)
            lon_offset = np.random.normal(0, 0.5)
            
            # Create realistic pollution data based on city type
            base_no2 = np.random.uniform(1e-5, 8e-5)  # Higher in cities
            base_pm25 = np.random.uniform(10, 80)      # PM2.5 in μg/m³
            base_co = np.random.uniform(0.1, 2.0)      # CO in ppm
            
            data.append({
                'latitude': city['lat'] + lat_offset,
                'longitude': city['lon'] + lon_offset,
                'city_region': city['name'],
                'NO2_column_number_density': base_no2 * np.random.uniform(0.5, 2.0),
                'pm25_estimate': base_pm25 * np.random.uniform(0.3, 1.8),
                'CO_estimate': base_co * np.random.uniform(0.4, 2.2),
                'year': 2023,
                'data_type': 'sample'
            })
    
    return pd.DataFrame(data), "Sample Environmental Data"

def create_pollution_map(df, parameter, color_scale="Viridis"):
    """Create an interactive pollution map"""
    
    # Parameter mapping - add support for all possible columns
    param_info = {
        'NO2_column_number_density': {
            'name': 'Nitrogen Dioxide (NO2)',
            'unit': 'mol/m²',
            'scale_factor': 1e6,  # Convert to 10^-6 mol/m²
            'unit_display': '×10⁻⁶ mol/m²'
        },
        'pm25_estimate': {
            'name': 'PM2.5 Concentration',
            'unit': 'μg/m³',
            'scale_factor': 1,
            'unit_display': 'μg/m³'
        },
        'CO_estimate': {
            'name': 'Carbon Monoxide (CO)',
            'unit': 'ppm',
            'scale_factor': 1,
            'unit_display': 'ppm'
        },
        # Add support for other possible columns
        'blue': {
            'name': 'Blue Band Reflectance',
            'unit': 'reflectance',
            'scale_factor': 1,
            'unit_display': 'reflectance'
        },
        'green': {
            'name': 'Green Band Reflectance', 
            'unit': 'reflectance',
            'scale_factor': 1,
            'unit_display': 'reflectance'
        },
        'red': {
            'name': 'Red Band Reflectance',
            'unit': 'reflectance', 
            'scale_factor': 1,
            'unit_display': 'reflectance'
        },
        'nir': {
            'name': 'Near Infrared Reflectance',
            'unit': 'reflectance',
            'scale_factor': 1, 
            'unit_display': 'reflectance'
        },
        'ndvi': {
            'name': 'NDVI (Vegetation Index)',
            'unit': 'index',
            'scale_factor': 1,
            'unit_display': 'index'
        },
        'evi': {
            'name': 'EVI (Enhanced Vegetation Index)',
            'unit': 'index', 
            'scale_factor': 1,
            'unit_display': 'index'
        }
    }
    
    # For any parameter not in the mapping, create a default entry
    if parameter not in param_info:
        param_info[parameter] = {
            'name': parameter.replace('_', ' ').title(),
            'unit': 'units',
            'scale_factor': 1,
            'unit_display': 'units'
        }
    
    info = param_info[parameter]
    
    # Filter out invalid data
    valid_data = df.dropna(subset=[parameter])
    if valid_data.empty:
        st.warning(f"No valid data available for {parameter}")
        return None, None
    
    # Scale the data for display
    valid_data = valid_data.copy()
    valid_data['display_value'] = valid_data[parameter] * info['scale_factor']
    
    # Create base map centered on data
    center_lat = valid_data['latitude'].mean()
    center_lon = valid_data['longitude'].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=2,
        tiles='OpenStreetMap'
    )
    
    # Use clean circle markers instead of messy heatmap
    # Normalize values to create better color mapping
    min_val = valid_data['display_value'].quantile(0.1)  # Use 10th percentile to avoid outliers
    max_val = valid_data['display_value'].quantile(0.9)   # Use 90th percentile to avoid outliers
    
    # Create color function
    def get_color(value):
        # Normalize value between 0 and 1
        normalized = (value - min_val) / (max_val - min_val) if max_val > min_val else 0
        normalized = max(0, min(1, normalized))  # Clamp between 0 and 1
        
        # Create color gradient from green (low) to red (high)
        if normalized < 0.33:
            return 'green'
        elif normalized < 0.66:
            return 'orange'
        else:
            return 'red'
    
    # Add clean markers with better spacing
    sample_data = valid_data.sample(n=min(500, len(valid_data)))  # Limit markers for cleaner look
    
    for idx, row in sample_data.iterrows():
        color = get_color(row['display_value'])
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=4,
            popup=folium.Popup(
                f"<b>{info['name']}</b><br>"
                f"Value: {row['display_value']:.4f} {info['unit_display']}<br>"
                f"Location: ({row['latitude']:.3f}, {row['longitude']:.3f})",
                max_width=200
            ),
            color=color,
            weight=1,
            fill=True,
            fillColor=color,
            fillOpacity=0.7
        ).add_to(m)
    
    # Add legend
    legend_html = f'''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 150px; height: 90px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p><b>{info['name']}</b></p>
    <p><i class="fa fa-circle" style="color:green"></i> Low</p>
    <p><i class="fa fa-circle" style="color:orange"></i> Medium</p>
    <p><i class="fa fa-circle" style="color:red"></i> High</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m, valid_data

def create_scatter_plot(df, parameters):
    """Create scatter plots for selected parameters"""
    
    if not parameters:
        return None
    
    # Filter valid data
    valid_df = df.dropna(subset=parameters)
    
    if valid_df.empty:
        return None
    
    # Create subplot for multiple parameters
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set1[:len(parameters)]
    
    for i, param in enumerate(parameters):
        if param in valid_df.columns:
            # Scale data for better visualization
            if 'NO2' in param:
                y_values = valid_df[param] * 1e6
                y_label = 'NO2 (×10⁻⁶ mol/m²)'
            elif 'pm25' in param:
                y_values = valid_df[param]
                y_label = 'PM2.5 (μg/m³)'
            elif 'CO' in param:
                y_values = valid_df[param]
                y_label = 'CO (ppm)'
            else:
                y_values = valid_df[param]
                y_label = param
            
            fig.add_trace(go.Scatter(
                x=valid_df['longitude'],
                y=y_values,
                mode='markers',
                name=param.replace('_', ' ').title(),
                marker=dict(
                    color=colors[i % len(colors)],
                    size=8,
                    opacity=0.6
                ),
                hovertemplate=f"""
                <b>{param}</b><br>
                Longitude: %{{x:.3f}}<br>
                Value: %{{y:.3f}}<br>
                <extra></extra>
                """
            ))
    
    fig.update_layout(
        title="Pollution Parameters vs Longitude",
        xaxis_title="Longitude",
        yaxis_title="Concentration",
        height=400,
        hovermode='closest'
    )
    
    return fig

def display_statistics(df, parameters):
    """Display statistical summary"""
    
    if df.empty or not parameters:
        return
    
    valid_df = df.dropna(subset=parameters)
    
    if valid_df.empty:
        st.warning("No valid data for statistical analysis")
        return
    
    # Create statistics table
    stats_data = []
    
    for param in parameters:
        if param in valid_df.columns:
            values = valid_df[param]
            
            # Scale for display
            if 'NO2' in param:
                values = values * 1e6
                unit = '×10⁻⁶ mol/m²'
            elif 'pm25' in param:
                unit = 'μg/m³'
            elif 'CO' in param:
                unit = 'ppm'
            else:
                unit = 'units'
            
            stats_data.append({
                'Parameter': param.replace('_', ' ').title(),
                'Count': len(values),
                'Mean': f"{values.mean():.4f} {unit}",
                'Std Dev': f"{values.std():.4f} {unit}",
                'Min': f"{values.min():.4f} {unit}",
                'Max': f"{values.max():.4f} {unit}"
            })
    
    stats_df = pd.DataFrame(stats_data)
    st.dataframe(stats_df, use_container_width=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🌍 Global Environmental Data Explorer</h1>', 
                unsafe_allow_html=True)
    st.markdown("*Interactive Satellite Data Visualization for Air Quality Monitoring*")
    
    # Load data
    with st.spinner("Loading environmental data..."):
        df, data_source = load_sample_data()
    
    st.success(f"✅ Loaded {len(df)} data points from {data_source}")
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛 Controls")
        
        # Data info
        st.markdown("### 📊 Dataset Information")
        st.markdown(f"""
        - *Data Points*: {len(df):,}
        - *Coverage*: Global  
        - *Columns*: {', '.join(df.columns.tolist())}
        - *Source*: {data_source}
        """)
        
        # Parameter selection
        st.markdown("### 🌫 Pollution Parameters")
        
        # Find all pollution-related parameters in the actual data
        pollution_keywords = ['NO2', 'pm25', 'CO', 'SO2', 'O3', 'aod', 'AOD']
        available_params = []
        
        for col in df.columns:
            if any(keyword.lower() in col.lower() for keyword in pollution_keywords):
                available_params.append(col)
        
        # If we still don't find anything, look for numeric columns (excluding coordinates)
        if not available_params:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            exclude_cols = ['longitude', 'latitude', 'year', 'extraction_date']
            available_params = [col for col in numeric_cols if col not in exclude_cols]
        
        # Map parameter
        map_param = st.selectbox(
            "Choose parameter for map visualization:",
            available_params,
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        # Chart parameters
        chart_params = st.multiselect(
            "Select parameters for charts:",
            available_params,
            default=available_params[:2] if len(available_params) >= 2 else available_params,
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        # Map style
        st.markdown("### 🗺 Map Settings")
        map_style = st.selectbox(
            "Map Style:",
            ["OpenStreetMap", "CartoDB Positron", "Stamen Terrain"]
        )
        
        # Color scheme
        color_scheme = st.selectbox(
            "Color Scheme:",
            ["Viridis", "Plasma", "Inferno", "Magma", "YlOrRd", "RdYlBu_r"]
        )
        
        # Display options
        st.markdown("### 📈 Display Options")
        show_statistics = st.checkbox("Show Statistics", True)
        show_scatter = st.checkbox("Show Scatter Plot", True)
        show_high_pollution = st.checkbox("Mark High Pollution Areas", True)
    
    # Main content area
    if not df.empty and map_param:
        
        # Create and display map
        st.subheader(f"🗺 {map_param.replace('_', ' ').title()} Global Distribution")
        
        with st.spinner("Creating interactive map..."):
            try:
                result = create_pollution_map(df, map_param, color_scheme)
                
                if result and len(result) == 2:
                    map_obj, map_data = result
                else:
                    map_obj, map_data = None, None
                
                if map_obj and map_data is not None:
                    st_folium(map_obj, width=1200, height=600, returned_objects=[])
                    
                    # Display summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Data Points",
                            f"{len(map_data):,}",
                            help="Number of valid measurements"
                        )
                    
                    with col2:
                        mean_val = map_data['display_value'].mean()
                        st.metric(
                            "Average Value",
                            f"{mean_val:.3f}",
                            help="Mean concentration across all points"
                        )
                    
                    with col3:
                        max_val = map_data['display_value'].max()
                        st.metric(
                            "Maximum Value",
                            f"{max_val:.3f}",
                            help="Highest recorded concentration"
                        )
                    
                    with col4:
                        # Count high pollution areas
                        threshold = map_data['display_value'].quantile(0.8)
                        high_count = len(map_data[map_data['display_value'] >= threshold])
                        st.metric(
                            "High Pollution Areas",
                            high_count,
                            help="Areas above 80th percentile"
                        )
                
            except Exception as e:
                st.error(f"Error creating map: {str(e)}")
        
        # Statistics section
        if show_statistics and chart_params:
            st.subheader("📊 Statistical Summary")
            display_statistics(df, chart_params)
        
        # Scatter plot section
        if show_scatter and chart_params:
            st.subheader("📈 Parameter Analysis")
            
            scatter_fig = create_scatter_plot(df, chart_params)
            if scatter_fig:
                st.plotly_chart(scatter_fig, use_container_width=True)
            else:
                st.warning("No valid data available for scatter plot")
        
        # Regional analysis
        if 'city_region' in df.columns:
            st.subheader("🏙 Regional Analysis")
            
            # Group by region if available
            regional_data = df.groupby('city_region')[chart_params].mean().round(4)
            
            if not regional_data.empty:
                # Create bar chart for regional comparison
                fig_bar = px.bar(
                    regional_data.reset_index(),
                    x='city_region',
                    y=chart_params,
                    title="Average Pollution by Region",
                    barmode='group'
                )
                fig_bar.update_layout(height=400)
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # Regional statistics table
                st.dataframe(regional_data, use_container_width=True)
    
    else:
        st.warning("No data available for visualization")
    
    # Information section
    with st.expander("📖 About This Application", expanded=False):
        st.markdown("""
        ### 🌍 Global Environmental Data Explorer
        
        This application visualizes satellite-derived environmental data for air quality monitoring worldwide.
        
        *Features:*
        - 🗺 *Interactive Maps*: Heatmaps showing pollution distribution
        - 📊 *Statistical Analysis*: Comprehensive data statistics
        - 📈 *Interactive Charts*: Multi-parameter visualization
        - 🏙 *Regional Analysis*: City and region-based comparisons
        
        *Data Sources:*
        - *Sentinel-5P*: Atmospheric composition monitoring
        - *MODIS*: Earth observation satellite data
        - *Landsat*: Land surface and vegetation data
        
        *Parameters:*
        - *NO2*: Nitrogen dioxide concentrations
        - *PM2.5*: Fine particulate matter
        - *CO*: Carbon monoxide levels
        - *AOD*: Aerosol optical depth
        
        *Usage Instructions:*
        1. Select parameters using the sidebar controls
        2. Choose map style and color scheme
        3. Explore the interactive visualizations
        4. View regional analysis and statistics
        
        *Technical Information:*
        - Data is processed from multiple satellite sources
        - Measurements are displayed in standard scientific units
        - High pollution areas are automatically highlighted
        - All visualizations are interactive and responsive
        """)
        
        st.info("💡 *Tip*: Use the sidebar controls to customize the visualization and explore different parameters!")

if _name_ == "_main_":
    main()