import folium
from folium import plugins
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import branca.colormap as cm
from utils import get_color_scale

class MapVisualizer:
    """Handles map visualization and layer management"""
    
    def _init_(self):
        self.color_schemes = {
            'PM2.5': ['#00FF00', '#FFFF00', '#FFA500', '#FF0000', '#8B0000'],
            'NO2': ['#0000FF', '#00FFFF', '#FFFF00', '#FF8C00', '#FF0000'],
            'SO2': ['#FFFFFF', '#FFFF99', '#FFCC99', '#FF9999', '#FF6666'],
            'CO': ['#E6F3FF', '#99D6FF', '#66C2FF', '#3399FF', '#0066CC'],
            'O3': ['#F0F8FF', '#B3E0FF', '#80D0FF', '#4DA6FF', '#1A8CFF'],
            'AOD': ['#F5F5F5', '#D3D3D3', '#A9A9A9', '#696969', '#2F2F2F']
        }
        
        # WHO Air Quality Guidelines (where applicable)
        self.quality_thresholds = {
            'PM2.5': [5, 15, 25, 37.5, 75],  # μg/m³
            'NO2': [1e-5, 4e-5, 8e-5, 1.2e-4, 2e-4],  # mol/m²
            'SO2': [1e-4, 3e-4, 6e-4, 1e-3, 2e-3],  # mol/m²
            'CO': [0.01, 0.03, 0.06, 0.12, 0.25],  # mol/m²
            'O3': [0.06, 0.10, 0.14, 0.18, 0.24],  # mol/m²
            'AOD': [0.1, 0.2, 0.4, 0.6, 1.0]  # unitless
        }
    
    def create_base_map(self, center: List[float], style: str = "OpenStreetMap") -> folium.Map:
        """Create base map with specified style"""
        
        # Map style configurations
        tile_configs = {
            "OpenStreetMap": {
                'tiles': 'OpenStreetMap',
                'attr': '© OpenStreetMap contributors'
            },
            "Satellite": {
                'tiles': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                'attr': 'Esri, DigitalGlobe, GeoEye, Earthstar Geographics, CNES/Airbus DS, USDA, USGS, AeroGRID, IGN, and the GIS User Community'
            },
            "CartoDB Positron": {
                'tiles': 'CartoDB positron',
                'attr': '© CartoDB'
            },
            "CartoDB Dark_Matter": {
                'tiles': 'CartoDB dark_matter',
                'attr': '© CartoDB'
            }
        }
        
        config = tile_configs.get(style, tile_configs["OpenStreetMap"])
        
        # Create map
        m = folium.Map(
            location=center,
            zoom_start=10,
            tiles=config['tiles'],
            attr=config['attr']
        )
        
        # Add fullscreen plugin
        plugins.Fullscreen(
            position='topright',
            title='Expand map',
            title_cancel='Exit fullscreen',
            force_separate_button=True
        ).add_to(m)
        
        # Add measure control
        plugins.MeasureControl(
            position='topleft',
            primary_length_unit='kilometers',
            secondary_length_unit='miles',
            primary_area_unit='sqkilometers',
            secondary_area_unit='sqmiles'
        ).add_to(m)
        
        return m
    
    def add_pollution_layer(self, map_obj: folium.Map, data: np.ndarray, 
                          parameter: str, opacity: float = 0.7) -> folium.Map:
        """Add pollution data as a heatmap layer"""
        
        if data is None or data.size == 0:
            return map_obj
        
        try:
            # Get map bounds
            bounds = self._get_map_bounds(map_obj)
            
            # Convert data to heatmap points
            heat_data = self._data_to_heatmap_points(data, bounds, parameter)
            
            if not heat_data:
                return map_obj
            
            # Create heatmap layer
            heat_layer = plugins.HeatMap(
                heat_data,
                name=f"{parameter} Layer",
                min_opacity=0.1,
                radius=15,
                blur=10,
                gradient=self._get_heatmap_gradient(parameter),
                overlay=True,
                control=True
            )
            
            heat_layer.add_to(map_obj)
            
            # Add contour layer for better visualization
            self._add_contour_layer(map_obj, data, bounds, parameter, opacity)
            
            # Add color legend
            self._add_color_legend(map_obj, parameter)
            
        except Exception as e:
            print(f"Error adding pollution layer: {str(e)}")
        
        return map_obj
    
    def _get_map_bounds(self, map_obj: folium.Map) -> Tuple[float, float, float, float]:
        """Get the bounds of the map"""
        # Default bounds - in a real implementation, you'd extract from map_obj
        center = map_obj.location
        lat, lon = center[0], center[1]
        
        # Approximate bounds (±0.1 degrees)
        return (lon - 0.1, lat - 0.1, lon + 0.1, lat + 0.1)
    
    def _data_to_heatmap_points(self, data: np.ndarray, bounds: Tuple[float, float, float, float], 
                               parameter: str) -> List[List[float]]:
        """Convert 2D data array to heatmap points"""
        
        if data is None or data.size == 0:
            return []
        
        # Create coordinate grids
        min_lon, min_lat, max_lon, max_lat = bounds
        
        if len(data.shape) == 2:
            rows, cols = data.shape
            lats = np.linspace(min_lat, max_lat, rows)
            lons = np.linspace(min_lon, max_lon, cols)
        else:
            # Handle 1D data
            size = int(np.sqrt(len(data))) if len(data.shape) == 1 else 10
            lats = np.linspace(min_lat, max_lat, size)
            lons = np.linspace(min_lon, max_lon, size)
            data = data.reshape((size, -1)) if len(data) >= size else np.array([[np.nanmean(data)]])
        
        heat_points = []
        
        # Normalize data for heatmap intensity
        valid_data = data[~np.isnan(data)]
        if len(valid_data) == 0:
            return []
        
        min_val, max_val = np.min(valid_data), np.max(valid_data)
        if max_val == min_val:
            intensity_scale = lambda x: 0.5
        else:
            intensity_scale = lambda x: (x - min_val) / (max_val - min_val)
        
        # Convert to heat points
        for i in range(len(lats)):
            for j in range(min(len(lons), data.shape[1] if len(data.shape) > 1 else 1)):
                if len(data.shape) == 2:
                    value = data[i, j]
                else:
                    value = data[i] if i < len(data) else np.nan
                
                if not np.isnan(value) and value > 0:
                    lat, lon = lats[i], lons[j] if j < len(lons) else lons[0]
                    intensity = intensity_scale(value)
                    heat_points.append([lat, lon, intensity])
        
        return heat_points
    
    def _get_heatmap_gradient(self, parameter: str) -> Dict[float, str]:
        """Get color gradient for heatmap based on parameter"""
        
        colors = self.color_schemes.get(parameter, self.color_schemes['PM2.5'])
        
        gradient = {}
        for i, color in enumerate(colors):
            gradient[i / (len(colors) - 1)] = color
        
        return gradient
    
    def _add_contour_layer(self, map_obj: folium.Map, data: np.ndarray, 
                          bounds: Tuple[float, float, float, float], 
                          parameter: str, opacity: float):
        """Add contour layer for better data visualization"""
        
        try:
            # Skip contours if data is too small or invalid
            if data is None or data.size < 9:
                return
            
            # Create a simple grid overlay using rectangles
            min_lon, min_lat, max_lon, max_lat = bounds
            
            if len(data.shape) == 2:
                rows, cols = data.shape
            else:
                rows = cols = int(np.sqrt(len(data)))
                data = data.reshape((rows, -1)) if len(data) >= rows else np.array([[np.nanmean(data)]])
            
            # Calculate cell dimensions
            lat_step = (max_lat - min_lat) / rows
            lon_step = (max_lon - min_lon) / cols
            
            # Get color map
            colors = self.color_schemes.get(parameter, self.color_schemes['PM2.5'])
            thresholds = self.quality_thresholds.get(parameter, [0, 0.25, 0.5, 0.75, 1.0])
            
            valid_data = data[~np.isnan(data)]
            if len(valid_data) == 0:
                return
            
            data_min, data_max = np.min(valid_data), np.max(valid_data)
            
            # Create feature group for contours
            contour_group = folium.FeatureGroup(name=f"{parameter} Contours", overlay=True)
            
            for i in range(rows):
                for j in range(min(cols, data.shape[1] if len(data.shape) > 1 else 1)):
                    if len(data.shape) == 2:
                        value = data[i, j]
                    else:
                        value = data[i] if i < len(data) else np.nan
                    
                    if not np.isnan(value):
                        # Calculate cell bounds
                        cell_min_lat = min_lat + i * lat_step
                        cell_max_lat = cell_min_lat + lat_step
                        cell_min_lon = min_lon + j * lon_step
                        cell_max_lon = cell_min_lon + lon_step
                        
                        # Determine color based on value
                        color_idx = self._get_color_index(value, data_min, data_max, len(colors))
                        color = colors[color_idx]
                        
                        # Create rectangle
                        bounds_rect = [
                            [cell_min_lat, cell_min_lon],
                            [cell_max_lat, cell_max_lon]
                        ]
                        
                        folium.Rectangle(
                            bounds=bounds_rect,
                            color=color,
                            fillColor=color,
                            fillOpacity=opacity * 0.6,
                            weight=0,
                            popup=f"{parameter}: {value:.4f}"
                        ).add_to(contour_group)
            
            contour_group.add_to(map_obj)
            
        except Exception as e:
            print(f"Error adding contour layer: {str(e)}")
    
    def _get_color_index(self, value: float, data_min: float, data_max: float, num_colors: int) -> int:
        """Get color index based on value range"""
        
        if data_max == data_min:
            return num_colors // 2
        
        normalized = (value - data_min) / (data_max - data_min)
        color_idx = int(normalized * (num_colors - 1))
        return min(max(0, color_idx), num_colors - 1)
    
    def _add_color_legend(self, map_obj: folium.Map, parameter: str):
        """Add color legend to the map"""
        
        try:
            colors = self.color_schemes.get(parameter, self.color_schemes['PM2.5'])
            thresholds = self.quality_thresholds.get(parameter, [0, 0.25, 0.5, 0.75, 1.0])
            
            # Create colormap
            colormap = cm.LinearColormap(
                colors=colors,
                vmin=thresholds[0],
                vmax=thresholds[-1],
                caption=f'{parameter} Concentration'
            )
            
            # Add units based on parameter
            units = self._get_parameter_units(parameter)
            colormap.caption = f'{parameter} ({units})'
            
            # Add to map
            colormap.add_to(map_obj)
            
        except Exception as e:
            print(f"Error adding color legend: {str(e)}")
    
    def _get_parameter_units(self, parameter: str) -> str:
        """Get units for different parameters"""
        
        units_map = {
            'PM2.5': 'μg/m³',
            'NO2': 'mol/m²',
            'SO2': 'mol/m²',
            'CO': 'mol/m²',
            'O3': 'mol/m²',
            'AOD': 'unitless'
        }
        
        return units_map.get(parameter, 'units')
    
    def add_measurement_stations(self, map_obj: folium.Map, stations: List[Dict]) -> folium.Map:
        """Add air quality measurement stations to the map"""
        
        station_group = folium.FeatureGroup(name="Measurement Stations", overlay=True, control=True)
        
        for station in stations:
            folium.Marker(
                location=[station.get('lat', 0), station.get('lon', 0)],
                popup=folium.Popup(
                    f"<b>{station.get('name', 'Unknown Station')}</b><br>"
                    f"Type: {station.get('type', 'Air Quality')}<br>"
                    f"Status: {station.get('status', 'Active')}",
                    max_width=200
                ),
                tooltip=station.get('name', 'Measurement Station'),
                icon=folium.Icon(color='blue', icon='info-sign')
            ).add_to(station_group)
        
        station_group.add_to(map_obj)
        return map_obj
    
    def add_administrative_boundaries(self, map_obj: folium.Map, boundaries_data: Dict) -> folium.Map:
        """Add administrative boundaries to the map"""
        
        try:
            boundary_group = folium.FeatureGroup(name="Administrative Boundaries", overlay=True, control=False)
            
            # This would typically load GeoJSON data
            # For now, we'll skip this feature
            
            boundary_group.add_to(map_obj)
            
        except Exception as e:
            print(f"Error adding boundaries: {str(e)}")
        
        return map_obj
    
    def create_time_animation(self, map_obj: folium.Map, time_series_data: Dict[str, np.ndarray], 
                            parameter: str) -> folium.Map:
        """Create animated visualization for time series data"""
        
        try:
            # This would require additional plugins for animation
            # For now, we'll add a placeholder for future implementation
            pass
            
        except Exception as e:
            print(f"Error creating time animation: {str(e)}")
        
        return map_obj