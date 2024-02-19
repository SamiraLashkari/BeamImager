import tkinter as tk
from tkinter import filedialog
import BeamImager
import pandas as pd
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objs as go
from plotly import tools
from plotly.subplots import make_subplots
import plotly.offline as py
from sklearn.mixture import GaussianMixture
import os
import hdbscan
from scipy.spatial import ConvexHull

class DataVisualizationApp:
    def __init__(self, root):

        self.root = root
        self.root.title("Multibeam Imaging App")

        self.processing = False  # Flag to track whether data processing is ongoing
        
        # Field 1: Upload Data and Generate Histogram

        self.upload_label = tk.Label(root, text="Upload Data (.txt)")
        self.upload_label.pack()

        self.upload_frame = tk.Frame(root)
        self.upload_frame.pack()

        self.upload_button = tk.Button(self.upload_frame, text="Upload Data", command=self.upload_data, width=20, height=1)
        self.upload_button.pack(side=tk.LEFT)

        self.generate_histogram_button = tk.Button(self.upload_frame, text="Generate Histogram", command=self.generate_histogram, width=20, height=1)
        self.generate_histogram_button.pack(side=tk.LEFT)

        # Field 2: Set Threshold and Visualize Data

        self.threshold_label = tk.Label(root, text="Set Threshold")
        self.threshold_label.pack()

        self.set_threshold_frame = tk.Frame(root)
        self.set_threshold_frame.pack()

        self.lower_threshold_label = tk.Label(self.set_threshold_frame, text="Lower Threshold:")
        self.lower_threshold_label.pack(side=tk.LEFT)

        self.lower_threshold_entry = tk.Entry(self.set_threshold_frame)
        self.lower_threshold_entry.pack(side=tk.LEFT)

        self.upper_threshold_label = tk.Label(self.set_threshold_frame, text="Upper Threshold:")
        self.upper_threshold_label.pack(side=tk.LEFT)

        self.upper_threshold_entry = tk.Entry(self.set_threshold_frame)
        self.upper_threshold_entry.pack(side=tk.LEFT, pady=10)

        self.set_visualize_button = tk.Button(root, text="Set and Visualize", command=self.set_and_visualize, width=20, height=1)
        self.set_visualize_button.pack()

        #Field 3: Removing Outlier

        self.remove_outliers_label = tk.Label(root, text="Outlier Removal")
        self.remove_outliers_label.pack()

        self.remove_outliers_frame = tk.Frame(root)
        self.remove_outliers_frame.pack()  # Add more vertical padding

        # Statistical Outlier Removal
        self.statistical_label = tk.Label(self.remove_outliers_frame, text="Statistical Outlier Removal")
        self.statistical_label.grid(row=0, column=0, padx=10)

        self.nb_neighbors_label = tk.Label(self.remove_outliers_frame, text="nb_neighbors (default 1000):")
        self.nb_neighbors_label.grid(row=1, column=0, padx=10)

        self.nb_neighbors_entry = tk.Entry(self.remove_outliers_frame)
        self.nb_neighbors_entry.grid(row=1, column=1, padx=10)

        self.std_ratio_label = tk.Label(self.remove_outliers_frame, text="std_ratio (default 2.0):")
        self.std_ratio_label.grid(row=2, column=0, padx=10)

        self.std_ratio_entry = tk.Entry(self.remove_outliers_frame)
        self.std_ratio_entry.grid(row=2, column=1, padx=10)

        self.statistical_button = tk.Button(self.remove_outliers_frame, text="Statistical Outlier Removal", command=self.statistical_remove_outliers)
        self.statistical_button.grid(row=3, column=0, columnspan=3)

        # Radius Outlier Removal
        self.radius_label = tk.Label(self.remove_outliers_frame, text="Radius Outlier Removal")
        self.radius_label.grid(row=0, column=3, padx=10)

        self.nb_points_label = tk.Label(self.remove_outliers_frame, text="nb_points (default 1000):")
        self.nb_points_label.grid(row=1, column=3, padx=10)

        self.nb_points_entry = tk.Entry(self.remove_outliers_frame)
        self.nb_points_entry.grid(row=1, column=4, padx=10)

        self.radius_label = tk.Label(self.remove_outliers_frame, text="Radius (default 0.5):")
        self.radius_label.grid(row=2, column=3, padx=10)

        self.radius_entry = tk.Entry(self.remove_outliers_frame)
        self.radius_entry.grid(row=2, column=4, padx=10)

        self.radius_button = tk.Button(self.remove_outliers_frame, text="Radius Outlier Removal", command=self.radius_remove_outliers)
        self.radius_button.grid(row=3, column=3, columnspan=3)

        # Field 4: K-Means Clustering

        self.clustering_label = tk.Label(root, text="K-Means Clustering")
        self.clustering_label.pack()

        self.density_frame = tk.Frame(root)
        self.density_frame.pack()

        self.density_label = tk.Label(self.density_frame, text="Density Clustering - Enter Number of Clusters:")
        self.density_label.pack(side=tk.LEFT)

        self.density_clusters_entry = tk.Entry(self.density_frame)
        self.density_clusters_entry.pack(side=tk.LEFT)

        self.density_clustering_button = tk.Button(self.density_frame, text="Perform and Visualize Clustering", command=self.perform_Kmean_density_clustering)
        self.density_clustering_button.pack(side=tk.LEFT)

        self.intensity_frame = tk.Frame(root)
        self.intensity_frame.pack()

        self.intensity_label = tk.Label(self.intensity_frame, text="Intensity Clustering - Enter Number of Clusters:")
        self.intensity_label.pack(side=tk.LEFT)

        self.intensity_clusters_entry = tk.Entry(self.intensity_frame)
        self.intensity_clusters_entry.pack(side=tk.LEFT)

        self.intensity_clustering_button = tk.Button(self.intensity_frame, text="Perform and Visualize Clustering", command=self.perform_Kmean_intensity_clustering)
        self.intensity_clustering_button.pack(side=tk.LEFT)

        # Field 5: GMM Clustering

        self.gmm_label = tk.Label(root, text="GMM Clustering")
        self.gmm_label.pack()

        self.density_gmm_frame = tk.Frame(root)
        self.density_gmm_frame.pack()

        self.density_gmm_label = tk.Label(self.density_gmm_frame, text="Density GMM Clustering - Enter Number of Clusters:")
        self.density_gmm_label.pack(side=tk.LEFT)

        self.density_gmm_components_entry = tk.Entry(self.density_gmm_frame)
        self.density_gmm_components_entry.pack(side=tk.LEFT)

        self.density_gmm_clustering_button = tk.Button(self.density_gmm_frame, text="Perform and Visualize Clustering", command=self.perform_density_gmm_clustering)
        self.density_gmm_clustering_button.pack(side=tk.LEFT)

        self.intensity_gmm_frame = tk.Frame(root)
        self.intensity_gmm_frame.pack()

        self.intensity_gmm_label = tk.Label(self.intensity_gmm_frame, text="Intensity GMM Clustering - Enter Number of Clusters:")
        self.intensity_gmm_label.pack(side=tk.LEFT)

        self.intensity_gmm_components_entry = tk.Entry(self.intensity_gmm_frame)
        self.intensity_gmm_components_entry.pack(side=tk.LEFT)

        self.intensity_gmm_clustering_button = tk.Button(self.intensity_gmm_frame, text="Perform and Visualize Clustering", command=self.perform_intensity_gmm_clustering)
        self.intensity_gmm_clustering_button.pack(side=tk.LEFT)

        # Field 6 : hdbscan clustering

        self.hdbscan_label = tk.Label(root, text="Hdbscan Clustering")
        self.hdbscan_label.pack()

        self.hdbscan_frame = tk.Frame(root)
        self.hdbscan_frame.pack()

        self.minClusterSize_label = tk.Label(self.hdbscan_frame, text=" Enter minimum cluster size (default 10000):")
        self.minClusterSize_label.pack(side=tk.LEFT)

        self.minClusterSize_entry = tk.Entry(self.hdbscan_frame)
        self.minClusterSize_entry.pack(side=tk.LEFT)

        self.minNumSamples_label = tk.Label(self.hdbscan_frame, text="Enter minimum number of samples (default 1000):")
        self.minNumSamples_label.pack(side=tk.LEFT)

        self.minNumSamples_entry = tk.Entry(self.hdbscan_frame)
        self.minNumSamples_entry.pack(side=tk.LEFT)

        self.hdbscan_button = tk.Button(self.hdbscan_frame, text="Perform and Visualize Clustering", command=self.perform_hdbscan_clustering_density)
        self.hdbscan_button.pack(side=tk.LEFT)

        # Field 7: Choose Cluster 
        
        self.choose_cluster_frame = tk.Frame(root)
        self.choose_cluster_frame.pack()

        self.choose_cluster_label = tk.Label(self.choose_cluster_frame, text="Choose Cluster (1, 2, 3, etc.):")
        self.choose_cluster_label.pack()

        self.choose_cluster_entry = tk.Entry(self.choose_cluster_frame)
        self.choose_cluster_entry.pack(side=tk.LEFT)

        self.choose_cluster_button = tk.Button(self.choose_cluster_frame, text="Choose the cluster",command= self.choose_cluster)
        self.choose_cluster_button.pack(side=tk.LEFT)

        # Field 8: Interpolation

        self.interpolation_frame = tk.Frame(root)
        self.interpolation_frame.pack()

        self.choose_cluster_label = tk.Label(self.interpolation_frame, text=" Perform interpolation!")
        self.choose_cluster_label.pack()

        self.choose_cluster_button = tk.Button(self.interpolation_frame, text="Start",command= self.interpolete_data_btw_planes, width = 30)
        self.choose_cluster_button.pack(side=tk.LEFT, padx=60)

        # Field 8: Voxelization

        self.voxelization_label = tk.Label(root, text="Voxelization")
        self.voxelization_label.pack()

        self.voxelization_frame = tk.Frame(root)
        self.voxelization_frame.pack()

        self.voxelization_size_label = tk.Label(self.voxelization_frame, text=" The voxel size")
        self.voxelization_size_label.pack(side=tk.LEFT)

        self.voxel_size_entry = tk.Entry(self.voxelization_frame)
        self.voxel_size_entry.pack(side=tk.LEFT)

        self.Voxelization_button = tk.Button(self.voxelization_frame, text="Voxelize and calculate the volume", command=self.voxelize)
        self.Voxelization_button.pack(side=tk.LEFT)

        #Field 9 : Meshing

        self.meshing_label = tk.Label(root, text="Meshing")
        self.meshing_label.pack()

        self.meshing_frame = tk.Frame(root)
        self.meshing_frame.pack()

        self.meshing_label = tk.Label(self.meshing_frame, text=" The radius size to calculate normals")
        self.meshing_label.pack(side=tk.LEFT)

        self.meshing_entry = tk.Entry(self.meshing_frame)
        self.meshing_entry.pack(side=tk.LEFT)

        self.max_nn_label = tk.Label(self.meshing_frame , text=" The maximum number of neighbors")
        self.max_nn_label.pack(side=tk.LEFT)

        self.max_nn_entry = tk.Entry(self.meshing_frame )
        self.max_nn_entry.pack(side=tk.LEFT)

        self.depth_label = tk.Label(self.meshing_frame , text=" The depth in meshing")
        self.depth_label.pack(side=tk.LEFT)

        self.depth_entry = tk.Entry(self.meshing_frame )
        self.depth_entry.pack(side=tk.LEFT)

        self.meshing_button = tk.Button(self.meshing_frame, text="Meshing", command = self.meshing )
        self.meshing_button.pack(side=tk.LEFT)

        self.volume_button = tk.Button(self.meshing_frame, text="Calculate Volume", command = self.calculate_volume)
        self.volume_button.pack(side=tk.LEFT)

        # Button to Access Previous Point Cloud
        self.count = -1
        self.access_previous_button = tk.Button(root, text="Access Previous Point Cloud", command=self.access_previous_point_cloud)
        self.access_previous_button.pack(side=tk.LEFT, padx=10, pady=10)

        # Default parameters for outlier removal
        self.default_nb_neighbors = 1000
        self.default_std_ratio = 2.0
        self.default_nb_points = 1000
        self.default_radius = 0.5

        # Default number of clusters
        self.default_num_clusters = 2
        self.default_minClusterSize = 10000
        self.default_minNumSamples = 1000

        #Default parameters for meshing
        self.default_radius = 0.1
        self.default_max_nn = 30
        self.default_depth = 8
        self.default_voxel_size = 0.1

        # Text Box for Information
        self.text_box = tk.Text(root, height=20, width=60)
        self.text_box.pack(pady=10)

    def print_info(self, message):
        self.text_box.insert(tk.END, message + "\n")
        self.text_box.see(tk.END)  # Scroll to the end of the text

    def upload_data(self):
        
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt"), ("CSV files", "*.csv")])
        if file_path:
            with open(file_path, 'r') as file:
                first_lines = [next(file) for _ in range(5)]  # Read the first 5 lines
                for line in first_lines:
                    if ',' in line:
                        delimiter = ','  # Comma delimiter
                        break
                    elif '\t' in line:
                        delimiter = '\t'  # Tab delimiter
                        break
                else:
                    delimiter = ' '
            
            # Read the data from the text file into a Pandas DataFrame
            self.df = pd.read_csv(file_path, delimiter=delimiter, header=None)
            
            ## Getting geometry and intensity of data
            self.x_data, self.y_data, self.z_data, self.intensity_data, self.ping_number, self.beam = BeamImager.getting_values(self.df)

            # column names
            column_names = ['x', 'y', 'z', 'intensity', 'ping_number', 'beam']

            # Assign column names to the DataFrame
            self.df.columns = column_names

            self.normalized_intensity, self.Max_intensity, self.Min_intensity, self.txt = BeamImager.normalize_intensity(self.intensity_data) 

            ## Choosing a color map for visualizing intensity
            self.colors = BeamImager.color_visualization_pointCloud(self.normalized_intensity)

            ## Adding columns for R, G and B with values
            self.df["R"] = self.colors.T[0]
            self.df["G"] = self.colors.T[1]
            self.df["B"] = self.colors.T[2]
            print(self.df.head())
            print(self.df["intensity"])

            ## Creating an open3D point cloud with geometrical and normalized intensity data
            self.point_cloud = BeamImager.create_pointCloud_object(np.array(self.df.iloc[:, :3]), np.array(self.df.iloc[:, -3:]))

            ## Visualizing the point cloud
            o3d.visualization.draw_geometries([self.point_cloud])

            ## Print number of points
            self.text = f"Number of points: {len(self.point_cloud.points)}"
            self.print_info(self.text)

            self.point_cloud_history = []  # List to store generated point clouds

            ## add the point cloud to the point_cloud_history_list
            self.point_cloud_history.append(self.point_cloud)
    
    def generate_histogram(self):

        BeamImager.histogram_intensity(self.intensity_data, num_bins=200)

    def set_and_visualize(self):
        lower_threshold = float(self.lower_threshold_entry.get())
        upper_threshold = float(self.upper_threshold_entry.get())

        if lower_threshold is not None and upper_threshold is not None:
            # Apply the thresholding and visualize the data
            ## Masking data based on the threshold
            self.mask = BeamImager.mask_threshold(self.df, lower_threshold, upper_threshold)

            ## Filtering the data and keep those within the defined threshold
            self.filtered_points, self.filtered_colors = BeamImager.keep_masked_pointClouds(self.mask, self.df, self.colors[:, :3])

            # Making a point cloud with filtered points based on the threshold
            self.point_cloud = BeamImager.create_pointCloud_object(self.filtered_points, self.filtered_colors)

            ## Visualizing the point cloud
            o3d.visualization.draw_geometries([self.point_cloud])

            ## Print number of points
            self.text = f"Number of points: {len(self.point_cloud.points)}"
            self.print_info(self.text)

            ## add the point cloud to the point_cloud_history_list
            self.point_cloud_history.append(self.point_cloud)
      
    def statistical_remove_outliers(self):
        nb_neighbors = int(self.nb_neighbors_entry.get()) if self.nb_neighbors_entry.get() else self.default_nb_neighbors
        std_ratio = float(self.std_ratio_entry.get()) if self.std_ratio_entry.get() else self.default_std_ratio
        
        # Call the statistical outlier removal function
        cl, ind = self.point_cloud.remove_statistical_outlier(nb_neighbors= nb_neighbors, std_ratio=std_ratio)
        
        # Visualize the result
        BeamImager.display_inlier_outlier(self.point_cloud, ind)
        o3d.visualization.draw_geometries([cl])
        self.point_cloud = cl

        ## Print number of points
        self.text = f"Number of points: {len(self.point_cloud.points)}"
        self.print_info(self.text)

        ## add the point cloud to the point_cloud_history_list
        self.point_cloud_history.append(self.point_cloud)

    def radius_remove_outliers(self):
        nb_points = int(self.nb_points_entry.get()) if self.nb_points_entry.get() else self.default_nb_points
        radius = float(self.radius_entry.get()) if self.radius_entry.get() else self.default_radius
        
        # Call your radius outlier removal function from open3d
        cl, ind = self.point_cloud.remove_radius_outlier(nb_points=nb_points, radius=radius)
        
        # Visualize the result
        BeamImager.display_inlier_outlier(self.point_cloud, ind)
        o3d.visualization.draw_geometries([cl])
        self.point_cloud = cl

        ## Print number of points
        self.text = f"Number of points: {len(self.point_cloud.points)}"
        self.print_info(self.text)

        ## add the point cloud to the point_cloud_history_list
        self.point_cloud_history.append(self.point_cloud)

    def perform_Kmean_density_clustering(self):
        # Perform density-based K-Mean clustering 
        num_clusters = int(self.density_clusters_entry.get()) if self.density_clusters_entry.get() else self.default_num_clusters
        self.clustered_point_clouds = BeamImager.clustering_k_mean_points( self.point_cloud, num_clusters)
        self.cluster_clouds = BeamImager.visualize_clusters(self.clustered_point_clouds)

    def perform_Kmean_intensity_clustering(self):
        # Perform intensity-based K-Mean clustering 
        num_clusters = int(self.intensity_clusters_entry.get()) if self.intensity_clusters_entry.get() else self.default_num_clusters
        self.clustered_point_clouds = BeamImager.clustering_k_mean_colors( self.point_cloud, num_clusters) 
        self.cluster_clouds = BeamImager.visualize_clusters(self.clustered_point_clouds)

    def perform_density_gmm_clustering(self):
        # Perform density-based GMM clustering 
        num_clusters = int(self.density_gmm_components_entry.get()) if self.density_gmm_components_entry.get() else self.default_num_clusters
        self.clustered_point_clouds = BeamImager.gmm_clustering_with_points( self.point_cloud, num_clusters) 
        self.cluster_clouds = BeamImager.visualize_clusters(self.clustered_point_clouds)

    def perform_intensity_gmm_clustering(self):
        # Perform intensity-based GMM clustering 
        num_clusters = int(self.intensity_gmm_components_entry.get()) if self.intensity_gmm_components_entry.get() else self.default_num_clusters
        self.clustered_point_clouds = BeamImager.gmm_clustering_with_intensity( self.point_cloud, num_clusters) 
        self.cluster_clouds = BeamImager.visualize_clusters(self.clustered_point_clouds)
    
    def perform_hdbscan_clustering_density (self):
        # Perform HDBSCAN clustering based on spatial coordinates
        min_cluster_size = int(self.minClusterSize_entry.get()) if self.minClusterSize_entry.get() else self.default_minClusterSize
        min_samples = int(self.minNumSamples_entry.get()) if self.minNumSamples_entry.get() else self.default_minNumSamples
        self.clustered_point_clouds = BeamImager.hdbscan_clustering_for_point_clouds(self.point_cloud, min_cluster_size, min_samples)
        self.cluster_clouds = BeamImager.visualize_clusters(self.clustered_point_clouds)

    def choose_cluster (self):
        cluster_number = int(self.choose_cluster_entry.get()) if self.choose_cluster_entry.get() else 1
        self.point_cloud = self.cluster_clouds[cluster_number -1]
        # Visualize the clustered point clouds
        o3d.visualization.draw_geometries([self.point_cloud])
        self.text = f"Number of points: {len(self.point_cloud.points)}"
        self.print_info(self.text)

    def interpolete_data_btw_planes (self):
        self.df_filtered = BeamImager.pointcloud_to_df (self.point_cloud)
        self.merged_df = BeamImager.merging_ping (self.df, self.df_filtered)
        self.plane_list = BeamImager.get_plane_list (self.merged_df)
        self.interpolated_points_list, self.Average_distance_planes = BeamImager.interpolate_volumes(self.plane_list, interpolation_steps=10)
        self.text = f"The mean of distances between planes: {self.Average_distance_planes}"
        self.print_info(self.text)
        self.merged_cloud = BeamImager.merge_volumes(self.interpolated_points_list)
        o3d.visualization.draw_geometries([self.merged_cloud, self.point_cloud])
        self.point_cloud = self.merged_cloud
        ## add the point cloud to the point_cloud_history_list
        self.point_cloud_history.append(self.point_cloud)

    def voxelize(self):
        voxel_size = float(self.voxel_size_entry.get()) if self.voxel_size_entry.get() else self.default_voxel_size
        self.voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(self.point_cloud, voxel_size)
        o3d.visualization.draw_geometries([self.voxel_grid, self.point_cloud])
        # Calculate the volume of the voxelized point cloud
        voxel_volume = len(self.voxel_grid.get_voxels()) * voxel_size**3
        self.print_info(f"Volume of the point cloud (voxelized):{voxel_volume}")

    def meshing(self):
        radius = float(self.radius_entry.get()) if self.radius_entry.get() else self.default_radius
        max_nn = int(self.max_nn_entry.get()) if self.max_nn_entry.get() else self.default_max_nn
        depth = int(self.depth_entry.get()) if self.depth_entry.get() else self.default_depth
        self.mesh, self.densities = BeamImager.meshing(self.point_cloud, radius, max_nn, depth)

    def calculate_volume(self):  
        # Compute the convex hull
        hull = ConvexHull(np.asarray(self.point_cloud.points))

        # Calculate the volume of the convex hull
        volume_hull = hull.volume
        self.print_info(f"Volume of the convex hull:{volume_hull}")

        # Compute triangle normals
        self.mesh.compute_triangle_normals()
        # Calculate the volume of the mesh
        #volume_mesh = self.mesh.get_volume()
        #self.print_info(f"Volume of the mesh:{volume_mesh}")

        # Compute the axis-aligned bounding box (AABB)
        aabb = self.point_cloud.get_axis_aligned_bounding_box()

        # Get the minimum and maximum coordinates of the cuboid
        min_bound, max_bound = aabb.get_min_bound(), aabb.get_max_bound()

        # Calculate the dimensions of the cuboid
        dimensions = np.abs(max_bound - min_bound)
        # Calculate the volume of the cuboid
        volume = np.prod(dimensions)

        self.print_info("Smallest cuboid:")
        self.print_info(f"Min Bound:{min_bound}")
        self.print_info(f"Max Bound:{max_bound}")
        self.print_info(f"Dimensions: {dimensions}")
        self.print_info(f"Volume: {volume}")

        # Create a cuboid using the dimensions
        cuboid = o3d.geometry.TriangleMesh.create_box(width=dimensions[0], height=dimensions[1], depth=dimensions[2])
        cuboid.translate((min_bound + max_bound) / 2)  # Center the cuboid at the same position as the point cloud

        # Visualize the point cloud and the cuboid
        o3d.visualization.draw_geometries([self.point_cloud, cuboid])  

    def access_previous_point_cloud(self):
        if self.point_cloud_history:
            # Access the most recent point cloud
            previous_point_cloud = self.point_cloud_history[self.count]
            self.count = self.count -1
            # Use the point cloud as needed
            o3d.visualization.draw_geometries([previous_point_cloud]) 
            self.point_cloud =  previous_point_cloud
            ## Print number of points
            self.text = f"Number of points: {len(self.point_cloud.points)}"
            self.print_info(self.text)
  
        else: 
            self.print_info("No previous point cloud available")

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1800x1500")  # the dimensions of window
    app = DataVisualizationApp(root)
    root.mainloop()
