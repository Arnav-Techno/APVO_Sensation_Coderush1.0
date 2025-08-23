import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any, Optional
import json

def get_color_scale(parameter: str, values: np.ndarray) -> List[str]:
    """Get appropriate color scale for a parameter"""
    
    color_schemes = {
        'PM2.5': ['#00FF00', '#FFFF00', '#FFA500', '#FF0000', '#8B0000'],  # Green to red
        'NO2': ['#0000FF', '#00FFFF', '#FFFF00', '#FF8C00', '#FF0000'],    # Blue to red
        'SO2': ['#FFFFFF', '#FFFF99', '#FFCC99', '#FF9999', '#FF6666'],    # White to red
        'CO': ['#E6F3FF', '#99D6FF', '#66C2FF', '#3399FF', '#0066CC'],     # Light to dark blue
        'O3': ['#F0F8FF', '#B3E0FF', '#80D0FF', '#4DA6FF', '#1A8CFF'],     # Light to dark blue
        'AOD': ['#F5F5F5', '#D3D3D3', '#A9A9A9', '#696969', '#2F2F2F']    # Light to dark gray
    }
    
    return color_schemes.get(parameter, color_schemes['PM2.5'])

def format_date_range(start_date: datetime, end_date: datetime) -> str:
    """Format date range for display"""
    
    if start_date == end_date:
        return start_date.strftime("%Y-%m-%d")
    else:
        return f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"

def validate_coordinates(lat: float, lon: float) -> bool:
    """Validate latitude and longitude coordinates"""
    
    return -90 <= lat <= 90 and -180 <= lon <= 180

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points using Haversine formula"""
    
    # Convert latitude and longitude from degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat/2)*2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)*2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    # Earth's radius in kilometers
    earth_radius = 6371.0
    
    return earth_radius * c

def get_air_quality_index(parameter: str, value: float) -> Dict[str, Any]:
    """Calculate Air Quality Index and health information"""
    
    # WHO Air Quality Guidelines and health classifications
    aqi_ranges = {
        'PM2.5': {
            'thresholds': [0, 5, 15, 25, 37.5, 75],
            'categories': ['Good', 'Fair', 'Moderate', 'Poor', 'Very Poor', 'Extremely Poor'],
            'colors': ['#00FF00', '#FFFF00', '#FFA500', '#FF0000', '#8B0000', '#654321'],
            'health_messages': [
                'Air quality is considered satisfactory',
                'Air quality is acceptable for most people',
                'Members of sensitive groups may experience health effects',
                'Everyone may begin to experience health effects',
                'Health warnings of emergency conditions',
                'Health alert: everyone may experience serious health effects'
            ]
        },
        'NO2': {
            'thresholds': [0, 1e-5, 4e-5, 8e-5, 1.2e-4, 2e-4],
            'categories': ['Good', 'Fair', 'Moderate', 'Poor', 'Very Poor', 'Extremely Poor'],
            'colors': ['#00FF00', '#FFFF00', '#FFA500', '#FF0000', '#8B0000', '#654321'],
            'health_messages': [
                'Low levels of nitrogen dioxide',
                'Acceptable levels for most people',
                'May affect sensitive individuals',
                'May cause respiratory issues',
                'Significant health concerns',
                'Dangerous levels - avoid outdoor activities'
            ]
        },
        'SO2': {
            'thresholds': [0, 1e-4, 3e-4, 6e-4, 1e-3, 2e-3],
            'categories': ['Good', 'Fair', 'Moderate', 'Poor', 'Very Poor', 'Extremely Poor'],
            'colors': ['#00FF00', '#FFFF00', '#FFA500', '#FF0000', '#8B0000', '#654321'],
            'health_messages': [
                'Low sulfur dioxide levels',
                'Acceptable for most people',
                'May affect people with asthma',
                'Respiratory effects likely',
                'Significant health risks',
                'Dangerous - stay indoors'
            ]
        }
    }
    
    if parameter not in aqi_ranges:
        return {
            'aqi': 0,
            'category': 'Unknown',
            'color': '#CCCCCC',
            'health_message': 'No information available'
        }
    
    param_data = aqi_ranges[parameter]
    thresholds = param_data['thresholds']
    
    # Find appropriate category
    category_index = 0
    for i, threshold in enumerate(thresholds[1:], 1):
        if value <= threshold:
            category_index = i - 1
            break
    else:
        category_index = len(thresholds) - 2  # Maximum category
    
    # Calculate AQI value (simplified)
    if category_index < len(thresholds) - 1:
        # Linear interpolation within category
        low_threshold = thresholds[category_index]
        high_threshold = thresholds[category_index + 1]
        aqi_low = category_index * 50
        aqi_high = (category_index + 1) * 50
        
        aqi = aqi_low + (aqi_high - aqi_low) * (value - low_threshold) / (high_threshold - low_threshold)
    else:
        aqi = category_index * 50
    
    return {
        'aqi': int(aqi),
        'category': param_data['categories'][category_index],
        'color': param_data['colors'][category_index],
        'health_message': param_data['health_messages'][category_index]
    }

def generate_sample_stations(center_lat: float, center_lon: float, radius_km: float = 50) -> List[Dict]:
    """Generate sample air quality monitoring stations"""
    
    stations = []
    station_types = ['Traffic', 'Industrial', 'Urban Background', 'Rural']
    
    # Generate random stations within the specified radius
    np.random.seed(42)  # For reproducibility
    
    for i in range(8):
        # Random position within radius
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(5, radius_km)
        
        # Convert to lat/lon offset
        lat_offset = distance * np.cos(angle) / 111.32  # Approximate km per degree
        lon_offset = distance * np.sin(angle) / (111.32 * np.cos(np.radians(center_lat)))
        
        station = {
            'id': f'STATION_{i+1:03d}',
            'name': f'Air Quality Station {i+1}',
            'lat': center_lat + lat_offset,
            'lon': center_lon + lon_offset,
            'type': np.random.choice(station_types),
            'status': 'Active' if np.random.random() > 0.1 else 'Maintenance',
            'elevation': int(np.random.uniform(10, 500)),  # meters above sea level
            'established': (datetime.now() - timedelta(days=np.random.randint(365, 3650))).strftime('%Y-%m-%d')
        }
        stations.append(station)
    
    return stations

def export_data_to_formats(data: Dict[str, Any], formats: List[str] = ['csv', 'json', 'geojson']) -> Dict[str, str]:
    """Export data to various formats"""
    
    exported_files = {}
    
    try:
        if 'csv' in formats:
            # Convert to pandas DataFrame and export CSV
            df_data = []
            
            for param, param_data in data.get('data', {}).items():
                for date, values in param_data.items():
                    if values is not None:
                        # Flatten 2D array to rows
                        if hasattr(values, 'shape') and len(values.shape) == 2:
                            rows, cols = values.shape
                            for i in range(rows):
                                for j in range(cols):
                                    if not np.isnan(values[i, j]):
                                        df_data.append({
                                            'parameter': param,
                                            'date': date,
                                            'row': i,
                                            'col': j,
                                            'value': values[i, j]
                                        })
            
            if df_data:
                df = pd.DataFrame(df_data)
                csv_content = df.to_csv(index=False)
                exported_files['csv'] = csv_content
        
        if 'json' in formats:
            # Convert numpy arrays to lists for JSON serialization
            json_data = {}
            for param, param_data in data.get('data', {}).items():
                json_data[param] = {}
                for date, values in param_data.items():
                    if values is not None:
                        json_data[param][date] = values.tolist() if hasattr(values, 'tolist') else values
            
            json_content = json.dumps(json_data, indent=2)
            exported_files['json'] = json_content
        
        if 'geojson' in formats:
            # Create GeoJSON format
            geojson_features = []
            bbox = data.get('bbox', [-1, -1, 1, 1])
            
            for param, param_data in data.get('data', {}).items():
                for date, values in param_data.items():
                    if values is not None and hasattr(values, 'shape'):
                        if len(values.shape) == 2:
                            rows, cols = values.shape
                            lat_coords = np.linspace(bbox[1], bbox[3], rows)
                            lon_coords = np.linspace(bbox[0], bbox[2], cols)
                            
                            for i in range(rows):
                                for j in range(cols):
                                    if not np.isnan(values[i, j]):
                                        feature = {
                                            "type": "Feature",
                                            "geometry": {
                                                "type": "Point",
                                                "coordinates": [lon_coords[j], lat_coords[i]]
                                            },
                                            "properties": {
                                                "parameter": param,
                                                "date": date,
                                                "value": float(values[i, j])
                                            }
                                        }
                                        geojson_features.append(feature)
            
            geojson_content = json.dumps({
                "type": "FeatureCollection",
                "features": geojson_features
            }, indent=2)
            exported_files['geojson'] = geojson_content
    
    except Exception as e:
        print(f"Error exporting data: {str(e)}")
    
    return exported_files

def validate_api_response(response_data: Any, expected_structure: Dict) -> bool:
    """Validate API response structure"""
    
    if not isinstance(response_data, dict):
        return False
    
    for key, value_type in expected_structure.items():
        if key not in response_data:
            return False
        
        if not isinstance(response_data[key], value_type):
            return False
    
    return True

def create_data_quality_report(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a data quality assessment report"""
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'parameters': {},
        'overall_quality': 'Unknown',
        'recommendations': []
    }
    
    total_coverage = 0
    total_parameters = 0
    
    for param, param_data in data.get('data', {}).items():
        param_stats = {
            'dates_available': len(param_data),
            'temporal_coverage': 0,
            'spatial_coverage': 0,
            'data_completeness': 0,
            'quality_score': 0
        }
        
        valid_dates = 0
        total_coverage_param = 0
        
        for date, values in param_data.items():
            if values is not None:
                valid_dates += 1
                
                # Calculate spatial coverage
                valid_pixels = np.count_nonzero(~np.isnan(values))
                total_pixels = values.size
                coverage = valid_pixels / total_pixels if total_pixels > 0 else 0
                total_coverage_param += coverage
        
        if valid_dates > 0:
            param_stats['temporal_coverage'] = valid_dates / len(param_data)
            param_stats['spatial_coverage'] = total_coverage_param / valid_dates
            param_stats['data_completeness'] = (param_stats['temporal_coverage'] + param_stats['spatial_coverage']) / 2
            param_stats['quality_score'] = param_stats['data_completeness'] * 100
        
        report['parameters'][param] = param_stats
        
        total_coverage += param_stats['data_completeness']
        total_parameters += 1
    
    # Overall quality assessment
    if total_parameters > 0:
        avg_quality = (total_coverage / total_parameters) * 100
        
        if avg_quality >= 80:
            report['overall_quality'] = 'Excellent'
        elif avg_quality >= 60:
            report['overall_quality'] = 'Good'
        elif avg_quality >= 40:
            report['overall_quality'] = 'Fair'
        elif avg_quality >= 20:
            report['overall_quality'] = 'Poor'
        else:
            report['overall_quality'] = 'Very Poor'
        
        # Generate recommendations
        if avg_quality < 50:
            report['recommendations'].append('Consider expanding the time range to improve data availability')
            report['recommendations'].append('Check API key permissions and rate limits')
        
        if any(stats['spatial_coverage'] < 0.3 for stats in report['parameters'].values()):
            report['recommendations'].append('High cloud coverage detected - consider alternative dates')
        
        if total_parameters < 3:
            report['recommendations'].append('Consider adding more parameters for comprehensive analysis')
    
    return report

def format_value_with_units(parameter: str, value: float) -> str:
    """Format numerical value with appropriate units"""
    
    units_map = {
        'PM2.5': 'μg/m³',
        'NO2': 'mol/m²',
        'SO2': 'mol/m²', 
        'CO': 'mol/m²',
        'O3': 'mol/m²',
        'AOD': ''  # unitless
    }
    
    precision_map = {
        'PM2.5': 1,
        'NO2': 6,
        'SO2': 6,
        'CO': 4,
        'O3': 4,
        'AOD': 3
    }
    
    unit = units_map.get(parameter, '')
    precision = precision_map.get(parameter, 3)
    
    if parameter in ['NO2', 'SO2'] and value < 1e-3:
        # Use scientific notation for very small values
        formatted = f"{value:.2e}"
    else:
        formatted = f"{value:.{precision}f}"
    
    return f"{formatted} {unit}".strip()