import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import pandas as pd
import os

# Set page configuration for Streamlit
st.set_page_config(layout="wide", page_title="test_world_1 Temperature Visualization")


def load_year_data(year):
    """Load data for the selected year"""
    try:
        filepath = f"data/test_world_1_{year}.csv"
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        st.error(f"Error loading data for year {year}: {e}")
        return None

def create_heatmap(year):
    """Create a heatmap for the specified year"""
    # Load data for the selected year
    df = load_year_data(year)
    
    if df is None or df.empty:
        st.error(f"No data available for year {year}")
        return None
    
    # Create a list of [lat, lon, value] for the heatmap
    heat_data = [[row.latitude, row.longitude, row.sensor_value] for index, row in df.iterrows()]
    
    # Create a folium map centered on test_world_1
    m = folium.Map(
        location=[21.1458, 79.0882],  # test_world_1 Coordinates
        tiles="cartodbpositron",
        zoom_start=12,
    )
    
    # Add the heatmap to the map with larger, more visible points
    # Further increased radius and adjusted other parameters
    HeatMap(heat_data, radius=30, blur=5, min_opacity=0.6, max_zoom=13, 
            gradient={"0.4": 'blue', "0.6": 'lime', "0.8": 'yellow', "1.0": 'red'}).add_to(m)
    
    return m

def display_statistics(year):
    """Display statistics for the selected year's data"""
    df = load_year_data(year)
    
    if df is None or df.empty:
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Average Temperature (°C)", f"{df['sensor_value'].mean():.2f}")
    
    with col2:
        st.metric("Maximum Temperature (°C)", f"{df['sensor_value'].max():.2f}")
    
    with col3:
        st.metric("Data Points", f"{len(df)}")

def main():
    """Main function to run the Streamlit app"""
    st.title("test_world_1 Temperature Visualization (2017-2024)")
    
    # Add information about the visualization
    with st.expander("About this visualization"):
        st.write("""
        This application visualizes simulated temperature data for test_world_1 city from 2017 to 2024. 
        The heatmaps show the distribution and intensity of temperature readings across the city.
        
        - **Sensor Type**: Temperature (°C)
        - **Data Points**: 200+ per year
        - **Location**: test_world_1, Maharashtra, India
        
        Use the year slider below to view temperature data for different years. Note that the patterns 
        change over time to reflect realistic variations in urban temperature readings.
        """)
        
        st.markdown("[Data Source: SLSTR Satellite Temperature Data](https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/slstr/)")
    
    # Year selection slider with more interactive styling
    st.markdown("""
    <style>
    div.row-widget.stSlider > div {
        background-color: #f5f9ff;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e0e8f0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    selected_year = st.slider(
        "Select Year to View Temperature Data", 
        min_value=2017, 
        max_value=2024, 
        value=2021,  # Setting default to 2021 as requested
        step=1,
        format="%d"
    )
    
    # Display statistics for the selected year
    display_statistics(selected_year)
    
    # Create and display the heatmap
    with st.spinner(f"Generating heatmap for {selected_year}..."):
        m = create_heatmap(selected_year)
        
        if m:
            # Display the map in Streamlit
            st_folium(m, width="100%", height=600)
            
            # Display information about the data source
            st.caption(f"Temperature heatmap for test_world_1 in {selected_year}. Data points show temperature variations across the city.")
        else:
            st.error("Failed to create heatmap. Please check the data files.")
    
    # Display year-specific information with data source
    st.subheader(f"Year {selected_year} - Data Notes")
    
    year_notes = {
        2017: "Early temperature monitoring with limited coverage, showing cooler overall readings.",
        2018: "Expanded temperature sensors with improved data consistency.",
        2019: "Notable temperature increases in commercial and industrial zones.",
        2020: "Lower temperatures in city center, potentially influenced by reduced urban activity.",
        2021: "Gradual warming trend with more uniform temperature distribution.",
        2022: "Significant heat increase in industrial areas, indicating urban heat island effect.",
        2023: "Higher temperatures concentrated in dense urban centers.",
        2024: "Current year with extreme heat readings in specific areas, showing climate trend concerns."
    }
    
    col1, col2 = st.columns([7, 3])
    
    with col1:
        st.info(year_notes[selected_year])
    
    with col2:
        st.markdown("""
        <div style="background-color:#f0f7ff; padding:15px; border-radius:10px; border:1px solid #4ba3ff;">
        <h4 style="color:#0066cc; margin-top:0;">Data Source</h4>
        <p><a href="https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/slstr/" target="_blank">
        SLSTR Satellite Temperature Data</a></p>
        <small>Sea and Land Surface Temperature Radiometer (SLSTR)</small>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
