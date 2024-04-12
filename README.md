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

1. **Data Upload**:
   - Upload data in .csv or .txt format with columns X, Y, Z, intensity, ping_number, and beam.
   - After upload, view intensity histogram.

2. **Default Values**:
   - Choose species to set default parameter values.

3. **Remove Noise**:
   - Visualize beam number histogram to identify noise.
   - Remove noise using 'Remove Most Frequent'.
   - Adjust bottom data removal ratio for profile noise.

4. **Setting Threshold**:
   - Adjust threshold if necessary and filter.

5. **Statistical Outlier Removal**:
   - Set number of neighbors and standard ratio for outlier removal.

6. **Radius Outlier Removal**:
   - Define minimum points and radius for outlier removal.

7. **K-Means Clustering**:
   - Automatically group similar points.
   - Choose number of clusters.

8. **GMM Clustering**:
   - Identify hidden patterns with less rigid clusters.
   - Choose number of clusters.

9. **Hdbscan Clustering**:
   - Identify clusters based on data point density.
   - Set minimum cluster size and samples.

10. **Drawing Clusters**:
    - Manually draw cluster boundaries if needed.

11. **Result Section**:
    - Edit, add, remove, and merge clusters.
    - Update and check result.

12. **Calculate Result Volume**:
    - Add cluster volumes to total volume.

13. **Interpolation**:
    - Estimate points between known data points.

14. **Voxelization**:
    - Convert point cloud to voxel grid.
    - Define thresholds for voxelization.
    - Set upper and lower thresholds for weighted voxelization.
    - Calculate voxel weights based on point counts.
