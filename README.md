# Aquaculture Installation Volume Calculator

## Overview

This repository provides code for visualizing and calculating the volume of aquaculture installations (it tested for seaweed and mussels) using processed data from a multibeam ecosounder sensor.

## Data Requirements

The uploaded data should have the following six columns in this specific order: 'x', 'y', 'z', 'intensity', 'ping_number', 'beam' and be .txt or .csv format. Please ensure that there are no indices for the columns.

You can find sample data [here](https://drive.google.com/file/d/15rBiuT27kIJQX5pq0yL5JK0sN9E-XCw0/view?usp=drive_link) to process with the developed GUI.

## Requirements

- **Python 3.9.17**: Please install Python 3.9.17 in your environment, as the open3d library is not compatible with more recent versions of Python.
  
- **Dependencies**: Install all necessary Python packages by running the following command in the terminal of your environment:
  
```bash
pip install -r requirements.txt
```
## Running the GUI

To run the graphical user interface (GUI), execute the following command in your terminal:

```bash
python BeamImager_GUI_v2.py
```
![GUI-image](https://github.com/SamiraLashkari/BeamImager/blob/main/GUI_image.png)

## Part 1: uploading data
**Data Upload**:
   - Upload data in .csv or .txt format with columns X, Y, Z, intensity, ping_number, and beam.
   - After upload, view intensity histogram.

**Default Values**:
   - Choose species to set default parameter values.
     
## Part 2: Filtering (red)
**Remove Noise**:
   - Visualize beam number histogram to identify noise.
   - Remove noise using 'Remove Most Frequent'.
   - Adjust bottom data removal ratio for profile noise.

**Setting Threshold**:
   - Adjust threshold if necessary and filter.

**Statistical Outlier Removal**:
   - Set number of neighbors and standard ratio for outlier removal.

**Radius Outlier Removal**:
   - Define minimum points and radius for outlier removal.
## Part3: Clustering (green)
**K-Means Clustering**:
   - Automatically group similar points.
   - Choose number of clusters.

**GMM Clustering**:
   - Identify hidden patterns with less rigid clusters.
   - Choose number of clusters.

**Hdbscan Clustering**:
   - Identify clusters based on data point density.
   - Set minimum cluster size and samples.

**Drawing Clusters**:
    - Manually draw cluster boundaries if needed.
    
## Part4: Volume calculation (yellow)
**Result Section**:
    - Edit, add, remove, and merge clusters to the filnal result
    - Update and check result by clicking on the choosen cluster to add, remove or merge into result.

**Interpolation**:
    - Estimate points between known data points in point cloud of each volume (you can choose any cluster to process the rest).

**Voxelization**:
    - Convert point cloud to voxel grid.
    - Define thresholds for voxelization.
    - Set upper and lower thresholds for weighted voxelization.
    - Calculate voxel weights based on point counts.

**Calculate Result Volume**:
    - Add cluster volumes to total volume.    
    
