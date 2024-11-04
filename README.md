# Pasture-Classification-RF-CNN-LSTM

Author: Paul Ruiz (N10962646) Last Modified: 01/11/2024
--------------

## Project Abstract

The growing need for sustainable land management has prompted the use of data-driven methods for monitoring and classifying pasture states, which play a critical role in agricultural productivity. However, accurately classifying diverse pasture states remains challenging due to class imbalance and the complex nature of agricultural landscapes. This study investigated the integration of multi-source data spatial (raster imagery), temporal (time-series), and tabular (farm metadata) to enhance classification accuracy in pasture state monitoring. Traditional single-source models often struggle to capture the nuances of diverse classes, especially in imbalanced datasets. This study addressed this gap by implementing a fusion model that combines Convolutional Neural Networks (CNN) for image analysis, Random Forest (RF) for tabular data and Long Short-Term Memory (LSTM) networks for temporal patterns.

The fusion model was trained and evaluated on multi-source data from Sentinel-2 satellite imagery and farm records, achieving an overall classification accuracy of 84%. Despite the considerable class imbalance, the model demonstrated robust performance, classifying the dominant "Grazing" state while capturing minority states like "Yard" and "Laneway" with reasonable accuracy. Validation accuracy was stable, but slight overfitting was observed for the dominant class, suggesting potential for improvement in handling class imbalance. The model’s success illustrates the value of combining spatial, temporal and static data, as it enabled the model to generalise well across diverse pasture states.

Conclusions indicate that multi-modal data fusion offers a promising approach for agricultural monitoring, providing a more detailed and accurate classification of pasture states. For future research, it is recommended to explore advanced fusion techniques, such as learned feature concatenation, and to incorporate further augmentation methods to further mitigate class imbalance effects.

--------------

## Usage

This project provides a complete pipeline for processing farm data, training classification models and evaluating results using Sentinel-2 satellite imagery. The objective is to classify the pasture state using both tabular farm data and rasterized satellite data.

The following is the data organisation into a structured directory. It comprises five main subdirectories: `farm_data`, `sentinel_rasters` and `geojson`. The `farm_data` directory contains CSV files for each of the five farms, detailing tabular data related to pasture states and other agricultural metrics. The `sentinel_rasters` folder is organized by farm, each containing raster files derived from Sentinel-2 satellite imagery. The `geojson` directory holds GeoJSON files that provide the geographical shapes of each farm. 

```bash
├── dataset
│   ├── farm_data
│   │   ├── Farm-1.csv
│   │   ├── Farm-2.csv
│   │   ├── Farm-3.csv
│   │   ├── Farm-4.csv
│   │   ├── Farm-5.csv
│   ├── sentinel_rasters
│   │   ├── Farm-1-raster
│   │   │   ├── ... (raster files)
│   │   ├── Farm-2-raster
│   │   │   ├── ... (raster files)
│   │   ├── Farm-3-raster
│   │   │   ├── ... (raster files)
│   │   ├── Farm-4-raster
│   │   │   ├── ... (raster files)
│   │   ├── Farm-5-raster
│   │   │   ├── ... (raster files)
│   ├── geojson
│   │   ├── farm-1.geojson
│   │   ├── farm-2.geojson
│   │   ├── farm-3.geojson
│   │   ├── farm-4.geojson
│   │   ├── farm-5.geojson
```


You can install these dependencies with the following command:

```python
pip install numpy pandas scikit-learn tensorflow rasterio geopandas seaborn matplotlib
```

## Project Workflow

The workflow consists of three main steps, managed through individual Python scripts:

### 1. Data Preprocessing <br/>

`process_farm_data.py` was used to preprocess farm data CSV files and Sentinel-2 rasters.

This script prepares non-image features, including extraction and normalization of features, and merges rasterized satellite data with farm data for each paddock The processed data is saved in a standardized format in the specified output directory.

```python
python process_farm_data.py /path/to/dataset /path/to/sentinel_rasters
```

### 2. Model Traning and Testing <br/>

`train.py` was used to train Random Forest, CNN, and LSTM models on the processed dataset.

Models are trained using different random seeds to ensure robustness and capture variability in model performance. After training, each model’s performance metrics and training history are saved to the output directory for analysis.


```python
python train.py /path/to/farm_data_dir /path/to/sentinel_rasters /path/to/output_dir --seed <SEED_VALUE>
```

`test.py` was used to evaluate the trained models on a test dataset and to generate visualizations.

The script loads the trained models and computes predictions on the test set, combining model predictions with defined weights. Visualizations include confusion matrices, accuracy and loss plots, and an actual vs. predicted values plot over time. Results and plots are saved to the output directory for easy review and interpretation. 

```python
python test.py /path/to/test_farm_data_dir /path/to/test_sentinel_rasters /path/to/output_dir
```

### 3. Executing Pipelines <br/>

The `run_experiments.bat` script provides a convenient way to execute the full pipeline from data processing to model training and testing. This script runs `process_farm_data.py`, then `train.py` with multiple random seeds, and finally `test.py` to evaluate the models.

```bash
@echo off
REM ########################
REM ### MAIN EXPERIMENTS ###
REM ########################

REM Set dataset paths
set DATASET_PATH=/path/to/dataset
set RASTER_PATH=/path/to/sentinel_rasters
set OUTPUT_DIR=/path/to/output_dir

REM Step 1: Preprocess farm data
echo Running data processing...
python process_farm_data.py %DATASET_PATH% %RASTER_PATH%
echo Data processing completed.

REM Step 2: Train model with different random seeds
for /L %%S in (0,1,4) do (
    echo Training model with seed %%S...
    python train.py %DATASET_PATH% %RASTER_PATH% %OUTPUT_DIR% --seed %%S
    echo Training completed for seed %%S.
)

REM Step 3: Run testing and save results
echo Running testing...
python test.py %DATASET_PATH% %RASTER_PATH% %OUTPUT_DIR%
echo Testing completed.

REM ############################
REM ### END MAIN EXPERIMENTS ###
REM ############################
```




