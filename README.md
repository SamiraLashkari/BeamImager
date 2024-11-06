# Aquaculture Installation Volume Calculator

A newer version can find here: [here](https://github.com/SamiraLashkari/CloudVolume_multibeam)
 
## Overview

This repository provides code for visualizing and calculating the volume of aquaculture installations (it tested for seaweed and mussels) using processed data from a multibeam ecosounder sensor.

If you have any question, don't hesitate to contact me: samira.lashkari@gmail.com

## Data Requirements

The uploaded data should have the following six columns in this specific order: 'x', 'y', 'z', 'intensity', 'ping_number', 'beam' and be .txt or .csv format. Please ensure that there are no indices for the columns.

You can find sample data [here](https://drive.google.com/file/d/15rBiuT27kIJQX5pq0yL5JK0sN9E-XCw0/view?usp=drive_link) to process with the developed GUI.

## Requirements

It is important to set the exact version of libraries and python. You can download anaconda here which makes it easier to set the desired version of python and libraries:
[here](https://www.anaconda.com/download). 

- **Python 3.9.20**: Please install Python 3.9.20 in your environment, as the open3d library is not compatible with more recent versions of Python.

Open Anaconda powershell prompt and write the command below. You give any name to your environment and replace my_env in command below. 
```bash
conda create -n my_env python=3.9.20
conda activate my_env
```
- **Dependencies**:
  - `open3d==0.17.0`
  - `hdbscan==0.8.33`
  - `pandas==1.5.3`
  - `numpy==1.24.3`
  - `scikit-learn==1.3.0`
  - `matplotlib==3.7.2`

## Installation and running GUI automatically

After setting up your environment with the specified Python version, you can install the `CloudVolume-multibeam` library along with its dependencies by running:

```bash
pip install CloudVolume-multibeam
```
Once installed, you can launch the GUI by running the following command:

```bash
CloudVolume
```
This will launch the application’s GUI.

## Installation and running GUI manually

To clone this repository, follow these steps:

1. Open **Git Bash**.
2. Navigate to the directory where you want to clone the repository.
3. Run the following command:

```bash
git clone https://github.com/SamiraLashkari/CloudVolume_multibeam
```
You can download the repository from website too. 

Then, you can install all dependencies in your environment by running the following command. First, navigate to the `CloudVolume_multibeam` folder in your system:

```bash
cd CloudVolume_multibeam
pip install -r requirements.txt
```

To launch the graphical user interface (GUI), execute the following command in your terminal:

```bash
python CloudVolume/CloudVolume.py
```
![GUI-image](https://github.com/SamiraLashkari/CloudVolume_multibeam/blob/main/GUI_CloudVolume.jpg)

## Uploading data
**Data Upload**:
   - Upload data in .csv or .txt format with columns X, Y, Z, intensity, ping_number, and beam.
   - After upload, view intensity histogram.

**Default Values**:
   - Choose species to set default parameter values.
     
## Filtering (red)
**Remove Noise**:
   - Visualize beam number histogram to identify noise.
   - Remove noise using **Remove Most Frequent**.
   - Adjust bottom data removal ratio for profile noise.

**Setting Threshold**:
   - Adjust threshold if necessary and filter.

**Statistical Outlier Removal**:
   - Set number of neighbors and standard ratio for outlier removal.

**Radius Outlier Removal**:
   - Define minimum points and radius for outlier removal.

## Saving Data as Point Cloud or DataFrame (blue)

At any point in the process before clustering, you can save the data either as a **point cloud** or a **dataframe**:

- **Saving as DataFrame**:  
  This allows you to load the data later directly from the GUI, enabling you to continue processing from where you left off.

- **Saving as Point Cloud**:  
  Point cloud data can be loaded into platforms like **CloudCompare** for visualization and analysis.

Before saving, you can verify the point cloud size by checking the fields **Size of Current Point Cloud** and **Size of Previous Point Cloud** to ensure you are saving the correct version.

This flexibility ensures you can choose the format that best suits your workflow, whether for further processing or visualization.

## Clustering (green)
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
      
  If the clustering algorithms don’t yield satisfactory results—especially in cases where clusters are very close with little distinction—you can manually define boundaries by drawing lines. Here’s how:

  1. Press the **Start** button. This will open another GUI window.
  2. To begin drawing lines between clusters, click the **Draw Curve** button. Each click will allow you to place points and connect them, forming boundaries.
  3. Once you’ve finished drawing borders for one cluster, press **Assign Labels** to label it.
  4. Continue drawing boundaries until all points belong to a cluster.
  5. When finished, press the **Finish** button.
  6. Return to the main GUI and click **Get Clusters** to finalize the clustering.

This will help to refine clusters manually, especially in challenging cases.
![GUI-image](https://github.com/SamiraLashkari/CloudVolume_multibeam/blob/main/Drawing_clusters.jpg)

## Result section (purple)

- **Refining and Managing Result**:

  After completing the initial clustering step, you can add these clusters to the result by pressing the **Add Clusters** button. The clusters will be updated based on their size and assigned to each cluster button, labeled from 1 to 40.

  Additional options for refining clusters include (each cluster for these operations can be selected by checking the square beside its label):
    
  - **Remove Clusters**: Remove selected clusters from the results.
  - **Merge Clusters**: Merge clusters to achieve the desired clustering arrangement.

  After each operation (add, merge, or remove), click the **Update** button to view the changes in the image, which will refresh to reflect the updated clustering.

  To reset or start over, use the following options:
  
  - **Empty Result**: Clears all clusters from the result.
    
  - **Convert All to Pointcloud**: Merges all points back into a single point cloud, allowing you to start the clustering process from the beginning.

    
## Volume calculation (yellow)

**Interpolation**:
- Estimate points between known data points in point cloud of each volume to fill the gap between pings (you can choose any cluster to process the rest)

**Voxelization**:
- Convert point cloud to voxel grid
- Define the size of voxel
- Set upper and lower thresholds for weighted voxelization (the default value for upper and lower threshold is d10 and d25 for number of points)
- Calculate the volume without and with weighted voxelization
   

## Part 6: Saving results (bright green)    

In this section, you can select clusters to save by checking the box next to each cluster and then clicking the **Add Chosen Cluster** button. Once selected, the clusters will be added to the list in the bottom-left corner, with their volumes calculated using both methods mentioned before.

You can remove the last cluster from the list by clicking the **Remove Last Cluster** button. 

When all desired clusters are in the list, click **Save Clusters and Volume**. This action will save:

- A `.csv` file containing all volume information for the selected clusters.
- Individual clusters as `.pcd` files (point cloud format).

This process ensures both the volume data and cluster geometries are preserved for further analysis.
    
