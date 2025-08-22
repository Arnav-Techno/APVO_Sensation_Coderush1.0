python generate_global_csv.py --year 2023 --params PM2.5,NO2,SO2,CO,O3
"""

import ee
import pandas as pd
import numpy as np
import argparse
import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Optional, Tuple

class GlobalSatelliteExtractor:
    def _init_(self, grid_resolution: float = 0.1):
        """
        Initialize the global satellite data extractor.
        
        Args:
            grid_resolution: Resolution in degrees (default 0.1° ≈ 10km)
        """
        self.grid_resolution = grid_resolution
        self.ee_initialized = False
        
        # Initialize Earth Engine
        try:
            ee.Initialize()
            self.ee_initialized = True
            print("✓ Earth Engine initialized successfully")
        except Exception as e:
            print(f"✗ Earth Engine initialization failed: {e}")
            print("  Run 'earthengine authenticate' first")
        
        # Dataset configurations
        self.datasets = {
            # Landsat datasets
            'landsat': {
                'LC08_Annual': 'LANDSAT/LC08/C02/T1_L2_ANNUAL_GREENEST_TOA',
                'LC09_Annual': 'LANDSAT/LC09/C02/T1_L2_ANNUAL_GREENEST_TOA'
            },
            
            # Sentinel-5P atmospheric datasets
            'sentinel5p': {
                'NO2': 'COPERNICUS/S5P/NRTI/L3_NO2',
                'SO2': 'COPERNICUS/S5P/NRTI/L3_SO2', 
                'CO': 'COPERNICUS/S5P/NRTI/L3_CO',
                'O3': 'COPERNICUS/S5P/NRTI/L3_O3',
                'CH4': 'COPERNICUS/S5P/NRTI/L3_CH4',
                'HCHO': 'COPERNICUS/S5P/NRTI/L3_HCHO'
            },
            
            # MODIS datasets
            'modis': {
                'TERRA_AOD': 'MODIS/061/MOD04_L2',
                'AQUA_AOD': 'MODIS/061/MYD04_L2'
            }
        }
        
        # Parameter to band mapping
        self.band_mapping = {
            'NO2': 'NO2_column_number_density',
            'SO2': 'SO2_column_number_density',
            'CO': 'CO_column_number_density',
            'O3': 'O3_column_number_density',
            'CH4': 'CH4_column_volume_mixing_ratio_dry_air',
            'HCHO': 'tropospheric_HCHO_column_number_density'
        }
    
    def create_global_grid(self, bounds: Optional[List[float]] = None) -> ee.FeatureCollection:
        """
        Create a global sampling grid.
        
        Args:
            bounds: Optional bounding box [min_lon, min_lat, max_lon, max_lat]
                   If None, creates global grid
        
        Returns:
            Earth Engine FeatureCollection of sampling points
        """
        if bounds:
            min_lon, min_lat, max_lon, max_lat = bounds
        else:
            min_lon, min_lat, max_lon, max_lat = -179.9, -89.9, 179.9, 89.9
        
        # Create coordinate ranges
        lat_range = ee.List.sequence(min_lat, max_lat, self.grid_resolution)
        lon_range = ee.List.sequence(min_lon, max_lon, self.grid_resolution)
        
        # Create grid points
        def make_points(lat):
            lat = ee.Number(lat)
            def make_point(lon):
                lon = ee.Number(lon)
                return ee.Feature(
                    ee.Geometry.Point([lon, lat]),
                    {'latitude': lat, 'longitude': lon}
                )
            return lon_range.map(make_point)
        
        points_nested = lat_range.map(make_points)
        points_flat = points_nested.flatten()
        
        print(f"Created sampling grid: {lat_range.size().getInfo()} × {lon_range.size().getInfo()} points")
        
        return ee.FeatureCollection(points_flat)
    
    def extract_landsat_data(self, year: int, output_file: str, 
                           bounds: Optional[List[float]] = None) -> bool:
        """
        Extract Landsat annual composite data.
        
        Args:
            year: Year to extract data for
            output_file: Output CSV file path
            bounds: Optional geographic bounds
            
        Returns:
            Success status
        """
        if not self.ee_initialized:
            return self._generate_mock_landsat_csv(year, output_file, bounds)
        
        try:
            print(f"\n🛰️  Extracting Landsat data for {year}...")
            
            # Load Landsat collection
            dataset_key = 'LC09_Annual' if year >= 2022 else 'LC08_Annual'
            collection = ee.ImageCollection(self.datasets['landsat'][dataset_key]) \
                          .filterDate(f'{year}-01-01', f'{year}-12-31')
            
            if collection.size().getInfo() == 0:
                print(f"No Landsat data available for {year}")
                return False
            
            # Create annual composite
            composite = collection.median()
            
            # Select and rename bands
            bands = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
            band_names = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
            composite = composite.select(bands, band_names)
            
            # Calculate vegetation indices
            ndvi = composite.normalizedDifference(['nir', 'red']).rename('ndvi')
            evi = composite.expression(
                '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
                {
                    'NIR': composite.select('nir'),
                    'RED': composite.select('red'),
                    'BLUE': composite.select('blue')
                }
            ).rename('evi')
            
            # Combine all bands
            final_image = composite.addBands([ndvi, evi])
            
            # Create sampling grid
            grid = self.create_global_grid(bounds)
            
            # Sample the image
            print("Sampling Landsat data at grid points...")
            sampled = final_image.sampleRegions(
                collection=grid,
                properties=['latitude', 'longitude'],
                scale=1000,  # 1km scale
                tileScale=4
            )
            
            # Export to CSV
            return self._export_ee_data_to_csv(sampled, output_file, 'landsat', year)
            
        except Exception as e:
            print(f"Error extracting Landsat data: {e}")
            return False
    
    def extract_atmospheric_data(self, parameter: str, year: int, output_file: str,
                               bounds: Optional[List[float]] = None) -> bool:
        """
        Extract atmospheric parameter data from Sentinel-5P.
        
        Args:
            parameter: Atmospheric parameter (NO2, SO2, CO, O3, CH4, HCHO)
            year: Year to extract data for
            output_file: Output CSV file path
            bounds: Optional geographic bounds
            
        Returns:
            Success status
        """
        if not self.ee_initialized:
            return self._generate_mock_atmospheric_csv(parameter, year, output_file, bounds)
        
        if parameter not in self.datasets['sentinel5p']:
            print(f"Parameter {parameter} not supported")
            return False
        
        try:
            print(f"\n🌫️  Extracting {parameter} data for {year}...")
            
            # Load Sentinel-5P collection
            collection = ee.ImageCollection(self.datasets['sentinel5p'][parameter]) \
                          .filterDate(f'{year}-01-01', f'{year}-12-31')
            
            if collection.size().getInfo() == 0:
                print(f"No {parameter} data available for {year}")
                return False
            
            # Create annual median composite
            band = self.band_mapping[parameter]
            composite = collection.select(band).median()
            
            # Create sampling grid
            grid = self.create_global_grid(bounds)
            
            # Sample the data
            print(f"Sampling {parameter} data at grid points...")
            sampled = composite.sampleRegions(
                collection=grid,
                properties=['latitude', 'longitude'],
                scale=5000,  # 5km scale for S5P
                tileScale=4
            )
            
            # Export to CSV
            return self._export_ee_data_to_csv(sampled, output_file, parameter, year)
            
        except Exception as e:
            print(f"Error extracting {parameter} data: {e}")
            return False
    
    def extract_pm25_data(self, year: int, output_file: str,
                         bounds: Optional[List[float]] = None) -> bool:
        """
        Extract PM2.5 estimates from MODIS AOD data.
        
        Args:
            year: Year to extract data for  
            output_file: Output CSV file path
            bounds: Optional geographic bounds
            
        Returns:
            Success status
        """
        if not self.ee_initialized:
            return self._generate_mock_pm25_csv(year, output_file, bounds)
        
        try:
            print(f"\n🌫️  Extracting PM2.5 data for {year}...")
            
            # Load MODIS AOD collections
            terra_aod = ee.ImageCollection(self.datasets['modis']['TERRA_AOD']) \
                         .filterDate(f'{year}-01-01', f'{year}-12-31')
            aqua_aod = ee.ImageCollection(self.datasets['modis']['AQUA_AOD']) \
                        .filterDate(f'{year}-01-01', f'{year}-12-31')
            
            # Combine collections
            combined_aod = terra_aod.merge(aqua_aod)
            
            if combined_aod.size().getInfo() == 0:
                print(f"No MODIS AOD data available for {year}")
                return False
            
            # Create annual median AOD
            aod = combined_aod.select('Optical_Depth_047').median()
            
            # Convert AOD to PM2.5 estimate (simplified conversion)
            pm25 = aod.multiply(25).rename('pm25_estimate')
            
            # Create sampling grid
            grid = self.create_global_grid(bounds)
            
            # Sample the data
            print("Sampling PM2.5 estimates at grid points...")
            sampled = pm25.sampleRegions(
                collection=grid,
                properties=['latitude', 'longitude'],
                scale=1000,  # 1km scale
                tileScale=4
            )
            
            # Export to CSV
            return self._export_ee_data_to_csv(sampled, output_file, 'PM2.5', year)
            
        except Exception as e:
            print(f"Error extracting PM2.5 data: {e}")
            return False
    
                def _export_ee_data_to_csv(self, ee_data: ee.FeatureCollection, 
            output_file: str, data_type: str, year: int) -> bool:

        
        try:
            print(f"Exporting {data_type} data to {output_file}...")
            
            # Get the data
            features = ee_data.getInfo()['features']
            
            if not features:
                print("No data points found")
                return False
            
            # Convert to DataFrame
            rows = []
            for feature in features:
                props = feature['properties']
                geom = feature['geometry']['coordinates']
                
                row = {
                    'longitude': geom[0],
                    'latitude': geom[1],
                    'year': year,
                    'data_type': data_type,
                    'extraction_date': datetime.now().isoformat()
                }
                
                # Add all measurement values
                for key, value in props.items():
                    if key not in ['longitude', 'latitude']:
                        row[key] = value
                
                rows.append(row)
            
            # Create DataFrame and save
            df = pd.DataFrame(rows)
            df = df.dropna()  # Remove rows with missing data
            
            print(f"Saving {len(df)} data points to {output_file}")
            df.to_csv(output_file, index=False)
            
            print(f"✓ Successfully exported {data_type} data ({len(df)} points)")
            return True
            
        except Exception as e:
            print(f"Export failed: {e}")
            return False
    
    def _generate_mock_landsat_csv(self, year: int, output_file: str, 
                                  bounds: Optional[List[float]]) -> bool:
        """Generate mock Landsat data when EE not available."""
        
        print(f"⚠️  Generating mock Landsat data for {year} (Earth Engine not available)")
        
        # Generate sample points
        if bounds:
            min_lon, min_lat, max_lon, max_lat = bounds
        else:
            min_lon, min_lat, max_lon, max_lat = -179.9, -89.9, 179.9, 89.9
        
        # Create sample grid (reduced for demo)
        np.random.seed(42)
        n_points = min(10000, int((max_lat - min_lat) * (max_lon - min_lon) / (self.grid_resolution ** 2)))
        
        lats = np.random.uniform(min_lat, max_lat, n_points)
        lons = np.random.uniform(min_lon, max_lon, n_points)
        
        # Generate realistic surface reflectance values
        data = {
            'longitude': lons,
            'latitude': lats,
            'year': [year] * n_points,
            'data_type': ['landsat'] * n_points,
            'extraction_date': [datetime.now().isoformat()] * n_points,
            'blue': np.random.uniform(0.05, 0.3, n_points),
            'green': np.random.uniform(0.05, 0.4, n_points),
            'red': np.random.uniform(0.05, 0.5, n_points),
            'nir': np.random.uniform(0.2, 0.8, n_points),
            'swir1': np.random.uniform(0.1, 0.6, n_points),
            'swir2': np.random.uniform(0.05, 0.4, n_points),
            'ndvi': np.random.uniform(-0.2, 0.9, n_points),
            'evi': np.random.uniform(-0.2, 0.8, n_points)
        }
        
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        
        print(f"✓ Mock Landsat data exported ({len(df)} points)")
        return True
    
    def _generate_mock_atmospheric_csv(self, parameter: str, year: int, 
                                     output_file: str, bounds: Optional[List[float]]) -> bool:
        """Generate mock atmospheric data when EE not available."""
        
        print(f"⚠️  Generating mock {parameter} data for {year} (Earth Engine not available)")
        
        if bounds:
            min_lon, min_lat, max_lon, max_lat = bounds
        else:
            min_lon, min_lat, max_lon, max_lat = -179.9, -89.9, 179.9, 89.9
        
        np.random.seed(42)
        n_points = min(8000, int((max_lat - min_lat) * (max_lon - min_lon) / (self.grid_resolution ** 2)))
        
        lats = np.random.uniform(min_lat, max_lat, n_points)
        lons = np.random.uniform(min_lon, max_lon, n_points)
        
        # Generate parameter-specific values
        value_ranges = {
            'NO2': (1e-6, 1e-4),
            'SO2': (1e-5, 1e-3),
            'CO': (0.01, 0.05),
            'O3': (0.08, 0.15),
            'CH4': (1700, 1900),
            'HCHO': (1e-5, 1e-4)
        }
        
        min_val, max_val = value_ranges.get(parameter, (0, 1))
        values = np.random.uniform(min_val, max_val, n_points)
        
        data = {
            'longitude': lons,
            'latitude': lats,
            'year': [year] * n_points,
            'data_type': [parameter] * n_points,
            'extraction_date': [datetime.now().isoformat()] * n_points,
            self.band_mapping.get(parameter, f'{parameter}_value'): values
        }
        
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        
        print(f"✓ Mock {parameter} data exported ({len(df)} points)")
        return True
    
    def _generate_mock_pm25_csv(self, year: int, output_file: str, 
                               bounds: Optional[List[float]]) -> bool:
        """Generate mock PM2.5 data when EE not available."""
        
        print(f"⚠️  Generating mock PM2.5 data for {year} (Earth Engine not available)")
        
        if bounds:
            min_lon, min_lat, max_lon, max_lat = bounds
        else:
            min_lon, min_lat, max_lon, max_lat = -179.9, -89.9, 179.9, 89.9
        
        np.random.seed(42)
        n_points = min(8000, int((max_lat - min_lat) * (max_lon - min_lon) / (self.grid_resolution ** 2)))
        
        lats = np.random.uniform(min_lat, max_lat, n_points)
        lons = np.random.uniform(min_lon, max_lon, n_points)
        
        # Generate PM2.5 values (μg/m³)
        pm25_values = np.random.uniform(5, 150, n_points)
        
        data = {
            'longitude': lons,
            'latitude': lats,
            'year': [year] * n_points,
            'data_type': ['PM2.5'] * n_points,
            'extraction_date': [datetime.now().isoformat()] * n_points,
            'pm25_estimate': pm25_values
        }
        
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        
        print(f"✓ Mock PM2.5 data exported ({len(df)} points)")
        return True


def main():
    parser = argparse.ArgumentParser(description='Extract global satellite data to CSV')
    parser.add_argument('--year', type=int, default=2023, 
                       help='Year to extract data for (default: 2023)')
    parser.add_argument('--params', type=str, 
                       default='PM2.5,NO2,SO2,CO,O3',
                       help='Comma-separated atmospheric parameters')
    parser.add_argument('--landsat', action='store_true',
                       help='Include Landsat surface data')
    parser.add_argument('--output-dir', type=str, default='./global_satellite_data',
                       help='Output directory for CSV files')
    parser.add_argument('--grid-resolution', type=float, default=0.1,
                       help='Grid resolution in degrees (default: 0.1)')
    parser.add_argument('--bounds', type=str, 
                       help='Geographic bounds: min_lon,min_lat,max_lon,max_lat')
    
    args = parser.parse_args()
    
    # Parse parameters
    params = [p.strip() for p in args.params.split(',') if p.strip()]
    
    # Parse bounds
    bounds = None
    if args.bounds:
        try:
            bounds = [float(x) for x in args.bounds.split(',')]
            if len(bounds) != 4:
                raise ValueError("Bounds must have 4 values")
        except ValueError:
            print("Invalid bounds format. Use: min_lon,min_lat,max_lon,max_lat")
            sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize extractor
    print(f"🌍 Global Satellite Data Extractor")
    print(f"Year: {args.year}")
    print(f"Parameters: {', '.join(params)}")
    print(f"Grid resolution: {args.grid_resolution}°")
    print(f"Output directory: {args.output_dir}")
    if bounds:
        print(f"Geographic bounds: {bounds}")
    
    extractor = GlobalSatelliteExtractor(args.grid_resolution)
    
    success_count = 0
    total_extractions = len(params) + (1 if args.landsat else 0)
    
    # Extract Landsat data
    if args.landsat:
        output_file = os.path.join(args.output_dir, f'global_landsat_{args.year}.csv')
        if extractor.extract_landsat_data(args.year, output_file, bounds):
            success_count += 1
    
    # Extract atmospheric parameters
    for param in params:
        output_file = os.path.join(args.output_dir, f'global_{param}_{args.year}.csv')
        
        if param == 'PM2.5':
            if extractor.extract_pm25_data(args.year, output_file, bounds):
                success_count += 1
        else:
            if extractor.extract_atmospheric_data(param, args.year, output_file, bounds):
                success_count += 1
    
    # Summary
    print(f"\n📊 Extraction Summary:")
    print(f"✓ {success_count}/{total_extractions} datasets extracted successfully")
    print(f"📁 Files saved to: {args.output_dir}")
    
    if success_count == total_extractions:
        print("🎉 All extractions completed successfully!")
    else:
        print("⚠️  Some extractions failed. Check logs above for details.")


if _name_ == "_main_":
    main()




 