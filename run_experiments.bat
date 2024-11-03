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
