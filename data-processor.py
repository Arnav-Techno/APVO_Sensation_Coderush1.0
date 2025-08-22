[3:53 am, 23/8/2025] ved (Iot): import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import os
import xarray as xr
from typing import Optional, Dict, Any, Tuple
import streamlit as st

class SentinelDataFetcher:
    """Fetches data from Sentinel Hub API"""
    
    def _init_(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://services.sentinel-hub.com/api/v1"
        self.oauth_url = "https://services.sentinel-hub.com/oauth"
        self.access_token = None
        self.token_expires = None
        
        # Parameter mapping for Sentinel-5P
        self.param_mapping = {
            'NO2': 'NO2',
            'SO2': 'SO2',
            'CO': 'CO',
            'O3': 'O3',
            'CH4': 'CH4',
     …
[3:53 am, 23/8/2025] ved (Iot): data fetcher
[4:35 am, 23/8/2025] ved (Iot): import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from scipy import interpolate
from scipy.spatial.distance import cdist
import xarray as xr

class DataProcessor:
    """Processes and combines satellite data from multiple sources"""
    
    def _init_(self):
        self.grid_resolution = 0.01  # degrees
        
    def calculate_bbox(self, lat: float, lon: float, size_km: float) -> Tuple[float, float, float, float]:
        """Calculate bounding box from center point and size"""
        
        # Approximate degrees per kilometer
        lat_deg_per_km = 1 / 111.32
        lon_deg_per_km = 1 / (111.32 * np.cos(np.radians(lat)))
        
        # Calculate half-size in degrees
        half_size_lat = (size_km / 2) * lat_deg_per_km
        half_size_lon = (size_km / 2) * lon_deg_per_km
        
        # Return [min_lon, min_lat, max_lon, max_lat]
        return (
            lon - half_size_lon,
            lat - half_size_lat,
            lon + half_size_lon,
            lat + half_size_lat
        )
    
    def process_multi_source_data(self, data_sources: List[Tuple[str, np.ndarray]], 
                                bbox: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
        """Process and combine data from multiple satellite sources"""
        
        if not data_sources:
            return None
        
        try:
            # If only one source, return processed single source
            if len(data_sources) == 1:
                return self._process_single_source(data_sources[0][1], bbox)
            
            # Combine multiple sources
            combined_data = []
            weights = []
            
            for source_name, data in data_sources:
                if data is not None:
                    processed_data = self._process_single_source(data, bbox)
                    if processed_data is not None:
                        combined_data.append(processed_data)
                        # Weight based on source reliability
                        weight = self._get_source_weight(source_name)
                        weights.append(weight)
            
            if not combined_data:
                return None
            
            # Weighted average of all sources
            return self._weighted_average(combined_data, weights)
            
        except Exception as e:
            print(f"Error processing multi-source data: {str(e)}")
            return None
    
    def _process_single_source(self, data: np.ndarray, bbox: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
        """Process data from a single source"""
        
        if data is None or data.size == 0:
            return None
        
        try:
            # Handle different data shapes
            if len(data.shape) == 1:
                # Convert 1D to 2D
                size = int(np.sqrt(len(data)))
                if size * size == len(data):
                    data = data.reshape((size, size))
                else:
                    # Create a grid for 1D data
                    grid_size = 50
                    return self._interpolate_1d_to_grid(data, bbox, grid_size)
            
            # Quality control
            data = self._quality_control(data)
            
            # Spatial filtering
            data = self._spatial_filter(data)
            
            return data
            
        except Exception as e:
            print(f"Error processing single source: {str(e)}")
            return None
    
    def _quality_control(self, data: np.ndarray) -> np.ndarray:
        """Apply quality control filters to the data"""
        
        # Remove extreme outliers
        if np.any(~np.isnan(data)):
            mean_val = np.nanmean(data)
            std_val = np.nanstd(data)
            
            # Remove values more than 4 standard deviations from mean
            outlier_mask = np.abs(data - mean_val) > 4 * std_val
            data[outlier_mask] = np.nan
            
            # Remove negative values for pollution parameters
            data[data < 0] = np.nan
        
        return data
    
    def _spatial_filter(self, data: np.ndarray, filter_size: int = 3) -> np.ndarray:
        """Apply spatial smoothing filter"""
        
        try:
            from scipy import ndimage
            
            # Create a valid data mask
            valid_mask = ~np.isnan(data)
            
            if np.sum(valid_mask) < 3:  # Not enough valid data
                return data
            
            # Apply Gaussian filter only to valid data
            smoothed = data.copy()
            
            # Fill NaN values with local mean for filtering
            temp_data = data.copy()
            kernel = np.ones((filter_size, filter_size)) / (filter_size * filter_size)
            
            # Simple moving average for areas with enough data
            from scipy.signal import convolve2d
            valid_conv = convolve2d(valid_mask.astype(float), kernel, mode='same', boundary='symm')
            data_conv = convolve2d(np.nan_to_num(data), kernel, mode='same', boundary='symm')
            
            # Only smooth where we have enough nearby valid data
            smooth_mask = valid_conv > 0.5
            smoothed[smooth_mask] = data_conv[smooth_mask] / valid_conv[smooth_mask]
            
            # Restore original NaN locations
            smoothed[~valid_mask] = np.nan
            
            return smoothed
            
        except ImportError:
            # Fallback without scipy
            return data
        except Exception:
            # Return original data if filtering fails
            return data
    
    def _interpolate_1d_to_grid(self, data: np.ndarray, bbox: Tuple[float, float, float, float], 
                               grid_size: int) -> np.ndarray:
        """Interpolate 1D data to a 2D grid"""
        
        try:
            # Create random spatial points for the 1D data
            np.random.seed(42)  # For reproducibility
            n_points = len(data)
            
            # Generate random points within bbox
            lats = np.random.uniform(bbox[1], bbox[3], n_points)
            lons = np.random.uniform(bbox[0], bbox[2], n_points)
            
            # Create regular grid
            grid_lats = np.linspace(bbox[1], bbox[3], grid_size)
            grid_lons = np.linspace(bbox[0], bbox[2], grid_size)
            grid_lat, grid_lon = np.meshgrid(grid_lats, grid_lons, indexing='ij')
            
            # Remove NaN values
            valid_mask = ~np.isnan(data)
            if np.sum(valid_mask) < 3:
                return np.full((grid_size, grid_size), np.nan)
            
            valid_data = data[valid_mask]
            valid_lats = lats[valid_mask]
            valid_lons = lons[valid_mask]
            
            # Interpolate using griddata
            from scipy.interpolate import griddata
            
            points = np.column_stack((valid_lats, valid_lons))
            grid_points = np.column_stack((grid_lat.ravel(), grid_lon.ravel()))
            
            interpolated = griddata(points, valid_data, grid_points, method='linear', fill_value=np.nan)
            
            return interpolated.reshape((grid_size, grid_size))
            
        except ImportError:
            # Fallback without scipy
            return np.full((grid_size, grid_size), np.nanmean(data))
        except Exception:
            # Fallback
            return np.full((grid_size, grid_size), np.nanmean(data))
    
    def _get_source_weight(self, source_name: str) -> float:
        """Get reliability weight for different data sources"""
        
        weights = {
            'Sentinel-5P': 1.0,  # High spatial resolution
            'MODIS': 0.8,        # Good temporal coverage
            'VIIRS': 0.7,        # Newer instrument
            'OMI': 0.6           # Older but reliable
        }
        
        return weights.get(source_name, 0.5)
    
    def _weighted_average(self, data_arrays: List[np.ndarray], weights: List[float]) -> np.ndarray:
        """Calculate weighted average of multiple data arrays"""
        
        if not data_arrays or not weights:
            return None
        
        # Ensure all arrays have the same shape
        target_shape = data_arrays[0].shape
        processed_arrays = []
        
        for data in data_arrays:
            if data.shape != target_shape:
                # Resize if needed
                data = self._resize_array(data, target_shape)
            processed_arrays.append(data)
        
        # Calculate weighted average
        weighted_sum = np.zeros(target_shape)
        weight_sum = np.zeros(target_shape)
        
        for data, weight in zip(processed_arrays, weights):
            valid_mask = ~np.isnan(data)
            weighted_sum[valid_mask] += data[valid_mask] * weight
            weight_sum[valid_mask] += weight
        
        # Avoid division by zero
        result = np.full(target_shape, np.nan)
        valid_weights = weight_sum > 0
        result[valid_weights] = weighted_sum[valid_weights] / weight_sum[valid_weights]
        
        return result
    
    def _resize_array(self, data: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Resize array to target shape using interpolation"""
        
        try:
            from scipy.ndimage import zoom
            
            # Calculate zoom factors
            zoom_factors = (target_shape[0] / data.shape[0], target_shape[1] / data.shape[1])
            
            # Handle NaN values
            valid_mask = ~np.isnan(data)
            if np.sum(valid_mask) == 0:
                return np.full(target_shape, np.nan)
            
            # Interpolate valid data
            valid_data = np.where(valid_mask, data, 0)
            resized_data = zoom(valid_data, zoom_factors, order=1)
            
            # Resize mask
            resized_mask = zoom(valid_mask.astype(float), zoom_factors, order=0) > 0.5
            
            # Apply mask to result
            resized_data[~resized_mask] = np.nan
            
            return resized_data
            
        except ImportError:
            # Fallback: simple repetition/sampling
            return np.full(target_shape, np.nanmean(data))
        except Exception:
            return np.full(target_shape, np.nanmean(data))
    
    def calculate_statistics(self, data: np.ndarray) -> Dict[str, float]:
        """Calculate statistical measures for the data"""
        
        if data is None or data.size == 0:
            return {}
        
        valid_data = data[~np.isnan(data)]
        
        if len(valid_data) == 0:
            return {}
        
        return {
            'mean': np.mean(valid_data),
            'median': np.median(valid_data),
            'std': np.std(valid_data),
            'min': np.min(valid_data),
            'max': np.max(valid_data),
            'count': len(valid_data),
            'coverage': len(valid_data) / data.size
        }
    
    def temporal_aggregation(self, data_dict: Dict[str, np.ndarray], method: str = 'mean') -> np.ndarray:
        """Aggregate data temporally using specified method"""
        
        if not data_dict:
            return None
        
        data_arrays = [data for data in data_dict.values() if data is not None]
        
        if not data_arrays:
            return None
        
        # Stack arrays
        stacked = np.stack(data_arrays, axis=0)
        
        if method == 'mean':
            return np.nanmean(stacked, axis=0)
        elif method == 'median':
            return np.nanmedian(stacked, axis=0)
        elif method == 'max':
            return np.nanmax(stacked, axis=0)
        elif method == 'min':
            return np.nanmin(stacked, axis=0)
        else:
            return np.nanmean(stacked, axis=0)