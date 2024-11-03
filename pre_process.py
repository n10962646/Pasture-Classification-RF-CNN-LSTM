import os
import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping, Polygon
from scipy import stats
import argparse

def load_data(base_path):
    """Load farm data and GeoJSON files based on the specified base path."""
    farm_data = {}
    geojson_data = {}

    # Load farm CSV files
    farm_data_path = os.path.join(base_path, 'farm_data')
    for filename in os.listdir(farm_data_path):
        if filename.endswith('.csv'):
            farm_name = filename.split('.')[0]
            farm_data[farm_name] = pd.read_csv(os.path.join(farm_data_path, filename))

    # Load GeoJSON files
    geojson_path = os.path.join(base_path, 'geojson')
    for filename in os.listdir(geojson_path):
        if filename.endswith('.geojson'):
            geojson_name = filename.split('.')[0]
            geojson_data[geojson_name] = gpd.read_file(os.path.join(geojson_path, filename))

    return farm_data, geojson_data

def convert_to_numeric(df, numeric_columns):
    """Convert specified columns to numeric and handle errors."""
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def detect_outliers(data):
    """Detect outliers using z-score method."""
    z_scores = np.abs(stats.zscore(data))
    return np.any(z_scores > 3)

def process_indices_from_rasters(combined_gdf, raster_files, base_file_path):
    """Processes NDVI, SAVI, and EVI values from raster files for each paddock geometry."""
    index_results = []
    for raster_file in raster_files:
        file_path = os.path.join(base_file_path, raster_file)
        with rasterio.open(file_path) as src:
            combined_gdf = combined_gdf.to_crs(src.crs)
            for idx, paddock in combined_gdf.iterrows():
                geometry = [mapping(paddock['GEOMETRY'])]
                try:
                    out_image, _ = mask(src, geometry, crop=True)
                except ValueError:
                    index_results.append({
                        'paddock_id': paddock['PADDOCK_ID'],
                        'raster_file': raster_file,
                        'NDVI': np.nan,
                        'SAVI': np.nan,
                        'EVI': np.nan
                    })
                    continue
                
                nir_band, red_band, blue_band = out_image[3, :, :], out_image[2, :, :], out_image[0, :, :]
                ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-10)
                L = 0.5
                savi = ((nir_band - red_band) / (nir_band + red_band + L)) * (1 + L)
                G, C1, C2, L_evi = 2.5, 6, 7.5, 1
                evi = G * ((nir_band - red_band) / (nir_band + C1 * red_band - C2 * blue_band + L_evi))

                ndvi, savi, evi = ndvi.flatten(), savi.flatten(), evi.flatten()
                ndvi, savi, evi = ndvi[ndvi != src.nodata], savi[savi != src.nodata], evi[evi != src.nodata]

                final_ndvi = np.median(ndvi) if detect_outliers(ndvi) else np.mean(ndvi)
                final_savi = np.median(savi) if detect_outliers(savi) else np.mean(savi)
                final_evi = np.median(evi) if detect_outliers(evi) else np.mean(evi)

                index_results.append({
                    'paddock_id': paddock['PADDOCK_ID'],
                    'raster_file': raster_file,
                    'NDVI': final_ndvi,
                    'SAVI': final_savi,
                    'EVI': final_evi
                })
    return pd.DataFrame(index_results)

def aggregate_and_merge_ndvi_data(ndvi_df, combined_gdf, agg_columns=['NDVI', 'SAVI', 'EVI'], merge_col='PADDOCK_ID'):
    """Aggregates NDVI, SAVI, and EVI data by PADDOCK_ID and merges it with combined_gdf."""
    ndvi_aggregated = ndvi_df.groupby(merge_col).agg({col: 'median' for col in agg_columns}).reset_index()
    farm_gjson_ndvi = combined_gdf.merge(ndvi_aggregated, on=merge_col, how='right')
    print("Merged GeoDataFrame Information:")
    farm_gjson_ndvi.info()
    return farm_gjson_ndvi

def remove_creation_date_column(df_list):
    """Removes the 'CREATION_DATE' column from each DataFrame in the list if it exists."""
    updated_dfs = []
    for df in df_list:
        if 'CREATION_DATE' in df.columns:
            df = df.drop(columns=['CREATION_DATE'])
            print("'CREATION_DATE' column removed.")
        updated_dfs.append(df)
    return updated_dfs

def merge_farm_datasets(datasets):
    """Merges multiple datasets with the same columns into a single DataFrame by concatenating them."""
    merged_df = pd.concat(datasets, ignore_index=True)
    print("Merged DataFrame Information:")
    merged_df.info()
    return merged_df

def get_common_columns(df1, df2):
    """Automatically find common columns between two DataFrames for merging."""
    return list(set(df1.columns).intersection(set(df2.columns)))

def main(base_path, raster_path):
    farm_data, geojson_data = load_data(base_path)
    combined_dfs = {}
    ndvi_savi_evi_dfs = {}

    for key, farm_df in farm_data.items():
        if key in geojson_data:
            common_cols = get_common_columns(farm_df, geojson_data[key])
            combined_dfs[key] = pd.merge(farm_df, geojson_data[key], on=common_cols, how='right')

    for name, gdf in combined_dfs.items():
        gdf = gdf.to_crs("EPSG:4326")  # Ensure CRS is set
        raster_files = os.listdir(os.path.join(raster_path, f"{name}-raster"))
        index_df = process_indices_from_rasters(gdf, raster_files, os.path.join(raster_path, f"{name}-raster"))
        ndvi_savi_evi_dfs[name] = aggregate_and_merge_ndvi_data(index_df, gdf)
    
    farm_outputs = list(ndvi_savi_evi_dfs.values())
    updated_farm_outputs = remove_creation_date_column(farm_outputs)
    merged_farms = merge_farm_datasets(updated_farm_outputs)

    # Display merged DataFrame columns as a check
    print("Final merged farm dataset columns:", merged_farms.columns)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process farm data and raster files for NDVI, SAVI, and EVI.')
    parser.add_argument('base_path', type=str, help='Base path to the dataset directory.')
    parser.add_argument('raster_path', type=str, help='Path to the directory containing Sentinel raster folders.')

    args = parser.parse_args()
    main(args.base_path, args.raster_path)
