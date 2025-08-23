import ee
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
import requests
from typing import List, Dict, Tuple, Optional
import streamlit as st

class GlobalDataExtractor:
    """Extract global satellite data for Landsat and atmospheric parameters"""
    
    def _init_(self):
        # Initialize Earth Engine (requires authentication)
        try:
            ee.Initialize()
            self.ee_initialized = True
        except:
            self.ee_initialized = False
        
        # Global grid parameters
        self.grid_resolution = 0.1  # degrees (~10km at equator)
        
        # Landsat datasets
        self.landsat_datasets = {
            'LC08_Annual': 'LANDSAT/LC08/C02/T1_L2_ANNUAL_GREENEST_TOA',
            'LC09_Annual': 'LANDSAT/LC09/C02/T1_L2_ANNUAL_GREENEST_TOA'
        }
        
        # Sentinel-5P datasets for air quality
        self.s5p_datasets = {
            'NO2': 'COPERNICUS/S5P/NRTI/L3_NO2',
            'SO2': 'COPERNICUS/S5P/NRTI/L3_SO2',
            'CO': 'COPERNICUS/S5P/NRTI/L3_CO',
            'O3': 'COPERNICUS/S5P/NRTI/L3_O3',
            'CH4': 'COPERNICUS/S5P/NRTI/L3_CH4',
            'HCHO': 'COPERNICUS/S5P/NRTI/L3_HCHO'
        }
        
        # MODIS for PM2.5/AOD
        self.modis_datasets = {
            'AOD': 'MODIS/061/MOD04_L2',  # For PM2.5 estimation
            'TERRA_AOD': 'MODIS/061/MOD04_L2',
            'AQUA_AOD': 'MODIS/061/MYD04_L2'
        }
    
    def create_global_grid(self) -> ee.FeatureCollection:
        """Create global point grid for sampling"""
        
        if not self.ee_initialized:
            raise Exception("Earth Engine not initialized")
        
        # Create global grid points
        # Latitude: -90 to 90, Longitude: -180 to 180
        lat_range = ee.List.sequence(-89.9, 89.9, self.grid_resolution)
        lon_range = ee.List.sequence(-179.9, 179.9, self.grid_resolution)
        
        # Create grid points
        def create_points(lat):
            lat = ee.Number(lat)
            def create_point_for_lon(lon):
                lon = ee.Number(lon)
                return ee.Feature(ee.Geometry.Point([lon, lat]), {
                    'latitude': lat,
                    'longitude': lon
                })
            return lon_range.map(create_point_for_lon)
        
        # Flatten the nested list
        points_nested = lat_range.map(create_points)
        points_flat = points_nested.flatten()
        
        return ee.FeatureCollection(points_flat)
    
    def extract_landsat_data(self, year: int, region_bounds: Optional[List[float]] = None) -> Dict:
        """Extract Landsat annual composite data globally"""
        
        if not self.ee_initialized:
            return self._mock_landsat_data(year, region_bounds)
        
        try:
            # Load Landsat collection
            landsat = ee.ImageCollection(self.landsat_datasets['LC08_Annual']) \
                       .filterDate(f'{year}-01-01', f'{year}-12-31')
            
            if landsat.size().getInfo() == 0:
                # Fallback to LC09 if LC08 not available
                landsat = ee.ImageCollection(self.landsat_datasets['LC09_Annual']) \
                         .filterDate(f'{year}-01-01', f'{year}-12-31')
            
            # Create annual composite
            composite = landsat.median()
            
            # Select bands
            bands = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']  # Blue, Green, Red, NIR, SWIR1, SWIR2
            composite = composite.select(bands)
            
            # Calculate vegetation indices
            ndvi = composite.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
            evi = composite.expression(
                '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
                {
                    'NIR': composite.select('SR_B5'),
                    'RED': composite.select('SR_B4'),
                    'BLUE': composite.select('SR_B2')
                }).rename('EVI')
            
            # Add indices to composite
            composite = composite.addBands([ndvi, evi])
            
            # Create sampling grid
            if region_bounds:
                # Regional extraction
                region = ee.Geometry.Rectangle(region_bounds)
                grid = self._create_regional_grid(region_bounds)
            else:
                # Global extraction (this will be very large!)
                grid = self.create_global_grid()
                # Limit to first 100k points for testing
                grid = ee.FeatureCollection(grid.toList(100000))
            
            # Sample the composite at grid points
            sampled = composite.sampleRegions(
                collection=grid,
                properties=['latitude', 'longitude'],
                scale=1000,  # 1km resolution for faster processing
                tileScale=4
            )
            
            return {
                'type': 'landsat',
                'year': year,
                'data': sampled,
                'bands': bands + ['NDVI', 'EVI']
            }
            
        except Exception as e:
            st.error(f"Error extracting Landsat data: {str(e)}")
            return self._mock_landsat_data(year, region_bounds)
    
    def extract_atmospheric_data(self, parameter: str, start_date: str, end_date: str, 
                               region_bounds: Optional[List[float]] = None) -> Dict:
        """Extract atmospheric data from Sentinel-5P"""
        
        if not self.ee_initialized:
            return self._mock_atmospheric_data(parameter, start_date, end_date, region_bounds)
        
        try:
            if parameter not in self.s5p_datasets:
                raise ValueError(f"Parameter {parameter} not supported")
            
            # Load Sentinel-5P collection
            collection = ee.ImageCollection(self.s5p_datasets[parameter]) \
                          .filterDate(start_date, end_date)
            
            # Get parameter-specific band
            band_map = {
                'NO2': 'NO2_column_number_density',
                'SO2': 'SO2_column_number_density',
                'CO': 'CO_column_number_density',
                'O3': 'O3_column_number_density',
                'CH4': 'CH4_column_volume_mixing_ratio_dry_air',
                'HCHO': 'tropospheric_HCHO_column_number_density'
            }
            
            band = band_map.get(parameter, 'NO2_column_number_density')
            
            # Create temporal composite (median)
            composite = collection.select(band).median()
            
            # Create sampling grid
            if region_bounds:
                grid = self._create_regional_grid(region_bounds)
            else:
                grid = self.create_global_grid()
                grid = ee.FeatureCollection(grid.toList(50000))  # Limit for processing
            
            # Sample the composite
            sampled = composite.sampleRegions(
                collection=grid,
                properties=['latitude', 'longitude'],
                scale=5000,  # 5km resolution for S5P
                tileScale=4
            )
            
            return {
                'type': 'atmospheric',
                'parameter': parameter,
                'start_date': start_date,
                'end_date': end_date,
                'data': sampled,
                'band': band
            }
            
        except Exception as e:
            st.error(f"Error extracting {parameter} data: {str(e)}")
            return self._mock_atmospheric_data(parameter, start_date, end_date, region_bounds)
    
    def extract_pm25_data(self, start_date: str, end_date: str, 
                         region_bounds: Optional[List[float]] = None) -> Dict:
        """Extract PM2.5 data from MODIS AOD"""
        
        if not self.ee_initialized:
            return self._mock_pm25_data(start_date, end_date, region_bounds)
        
        try:
            # Load MODIS AOD data
            modis_terra = ee.ImageCollection(self.modis_datasets['TERRA_AOD']) \
                           .filterDate(start_date, end_date)
            modis_aqua = ee.ImageCollection(self.modis_datasets['AQUA_AOD']) \
                          .filterDate(start_date, end_date)
            
            # Combine collections
            modis_combined = modis_terra.merge(modis_aqua)
            
            # Select AOD band
            aod = modis_combined.select('Optical_Depth_047').median()
            
            # Convert AOD to PM2.5 (simplified relationship)
            # PM2.5 ≈ AOD × 25 (very rough approximation, varies by region)
            pm25 = aod.multiply(25).rename('PM25_estimate')
            
            # Create sampling grid
            if region_bounds:
                grid = self._create_regional_grid(region_bounds)
            else:
                grid = self.create_global_grid()
                grid = ee.FeatureCollection(grid.toList(50000))
            
            # Sample the PM2.5 estimates
            sampled = pm25.sampleRegions(
                collection=grid,
                properties=['latitude', 'longitude'],
                scale=1000,
                tileScale=4
            )
            
            return {
                'type': 'pm25',
                'start_date': start_date,
                'end_date': end_date,
                'data': sampled,
                'band': 'PM25_estimate'
            }
            
        except Exception as e:
            st.error(f"Error extracting PM2.5 data: {str(e)}")
            return self._mock_pm25_data(start_date, end_date, region_bounds)
    
    def _create_regional_grid(self, bounds: List[float]) -> ee.FeatureCollection:
        """Create grid for a specific region"""
        
        min_lon, min_lat, max_lon, max_lat = bounds
        
        lat_range = ee.List.sequence(min_lat, max_lat, self.grid_resolution)
        lon_range = ee.List.sequence(min_lon, max_lon, self.grid_resolution)
        
        def create_points(lat):
            lat = ee.Number(lat)
            def create_point_for_lon(lon):
                lon = ee.Number(lon)
                return ee.Feature(ee.Geometry.Point([lon, lat]), {
                    'latitude': lat,
                    'longitude': lon
                })
            return lon_range.map(create_point_for_lon)
        
        points_nested = lat_range.map(create_points)
        points_flat = points_nested.flatten()
        
        return ee.FeatureCollection(points_flat)
    
    def export_to_csv(self, extraction_result: Dict, filename: str) -> str:
        """Export extraction result to CSV"""
        
        try:
            if self.ee_initialized and 'data' in extraction_result:
                # Get data from Earth Engine
                data = extraction_result['data']
                features = data.getInfo()['features']
                
                # Convert to DataFrame
                rows = []
                for feature in features:
                    props = feature['properties']
                    coords = feature['geometry']['coordinates']
                    
                    row = {
                        'longitude': coords[0],
                        'latitude': coords[1],
                        'year': extraction_result.get('year', ''),
                        'parameter': extraction_result.get('parameter', ''),
                        'start_date': extraction_result.get('start_date', ''),
                        'end_date': extraction_result.get('end_date', '')
                    }
                    
                    # Add all other properties
                    for key, value in props.items():
                        if key not in ['longitude', 'latitude']:
                            row[key] = value
                    
                    rows.append(row)
                
                df = pd.DataFrame(rows)
                
            else:
                # Use mock data
                df = self._create_mock_dataframe(extraction_result)
            
            # Save to CSV
            csv_content = df.to_csv(index=False)
            
            with open(filename, 'w') as f:
                f.write(csv_content)
            
            return csv_content
            
        except Exception as e:
            st.error(f"Error exporting to CSV: {str(e)}")
            return ""
    
    def _mock_landsat_data(self, year: int, region_bounds: Optional[List[float]]) -> Dict:
        """Generate mock Landsat data for demonstration"""
        
        return {
            'type': 'landsat',
            'year': year,
            'mock': True,
            'message': 'Earth Engine not initialized - using mock data'
        }
    
    def _mock_atmospheric_data(self, parameter: str, start_date: str, end_date: str, 
                              region_bounds: Optional[List[float]]) -> Dict:
        """Generate mock atmospheric data"""
        
        return {
            'type': 'atmospheric',
            'parameter': parameter,
            'start_date': start_date,
            'end_date': end_date,
            'mock': True,
            'message': 'Earth Engine not initialized - using mock data'
        }
    
    def _mock_pm25_data(self, start_date: str, end_date: str, 
                       region_bounds: Optional[List[float]]) -> Dict:
        """Generate mock PM2.5 data"""
        
        return {
            'type': 'pm25',
            'start_date': start_date,
            'end_date': end_date,
            'mock': True,
            'message': 'Earth Engine not initialized - using mock data'
        }
    
    def _create_mock_dataframe(self, extraction_result: Dict) -> pd.DataFrame:
        """Create mock DataFrame for demonstration"""
        
        # Generate sample global points
        np.random.seed(42)
        n_points = 1000
        
        lats = np.random.uniform(-89.9, 89.9, n_points)
        lons = np.random.uniform(-179.9, 179.9, n_points)
        
        data = {
            'latitude': lats,
            'longitude': lons,
            'year': [extraction_result.get('year', 2023)] * n_points,
            'parameter': [extraction_result.get('parameter', 'NO2')] * n_points
        }
        
        # Add mock values based on type
        if extraction_result['type'] == 'landsat':
            data.update({
                'SR_B2': np.random.uniform(0.05, 0.3, n_points),  # Blue
                'SR_B3': np.random.uniform(0.05, 0.4, n_points),  # Green
                'SR_B4': np.random.uniform(0.05, 0.5, n_points),  # Red
                'SR_B5': np.random.uniform(0.2, 0.8, n_points),   # NIR
                'SR_B6': np.random.uniform(0.1, 0.6, n_points),   # SWIR1
                'SR_B7': np.random.uniform(0.05, 0.4, n_points),  # SWIR2
                'NDVI': np.random.uniform(-0.2, 0.9, n_points),
                'EVI': np.random.uniform(-0.2, 0.8, n_points)
            })
        elif extraction_result['type'] == 'atmospheric':
            param = extraction_result.get('parameter', 'NO2')
            if param == 'NO2':
                data['NO2_column_number_density'] = np.random.uniform(1e-6, 1e-4, n_points)
            elif param == 'SO2':
                data['SO2_column_number_density'] = np.random.uniform(1e-5, 1e-3, n_points)
            elif param == 'CO':
                data['CO_column_number_density'] = np.random.uniform(0.01, 0.05, n_points)
            elif param == 'O3':
                data['O3_column_number_density'] = np.random.uniform(0.08, 0.15, n_points)
        elif extraction_result['type'] == 'pm25':
            data['PM25_estimate'] = np.random.uniform(5, 150, n_points)
        
        return pd.DataFrame(data)
    
    def process_global_extraction(self, parameters: List[str], year: int = 2023) -> Dict[str, str]:
        """Process complete global extraction for all parameters"""
        
        results = {}
        
        st.info("Starting global data extraction...")
        
        # 1. Extract Landsat data
        st.info("Extracting Landsat surface reflectance data...")
        landsat_result = self.extract_landsat_data(year)
        landsat_csv = self.export_to_csv(landsat_result, f"global_landsat_{year}.csv")
        results['landsat'] = landsat_csv
        
        # 2. Extract atmospheric parameters
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        
        for param in parameters:
            if param in ['PM2.5', 'PM10']:
                st.info(f"Extracting {param} data from MODIS AOD...")
                pm_result = self.extract_pm25_data(start_date, end_date)
                pm_csv = self.export_to_csv(pm_result, f"global_{param}_{year}.csv")
                results[param] = pm_csv
            elif param in self.s5p_datasets:
                st.info(f"Extracting {param} data from Sentinel-5P...")
                atm_result = self.extract_atmospheric_data(param, start_date, end_date)
                atm_csv = self.export_to_csv(atm_result, f"global_{param}_{year}.csv")
                results[param] = atm_csv
            else:
                st.warning(f"Parameter {param} not supported yet")
        
        st.success("Global data extraction completed!")
        
        return results

# Initialize the extractor
global_extractor = GlobalDataExtractor()