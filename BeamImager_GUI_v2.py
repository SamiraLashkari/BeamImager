import tkinter as tk
from tkinter import filedialog
import open3d as o3d
import BeamImager
import pandas as pd
import numpy as np
from tkinter import * 
from tkinter.ttk import *
from tkinter import messagebox

class DataVisualizationApp:
    def __init__(self, root):
        
        self.root = root
        self.root.title("Multibeam Imaging App")
        self.processing = False  # Flag to track whether data processing is ongoing
        self.cluster_clouds = None 
        
        # Display instruction message box when opening the GUI
        self.show_instruction_message()

        ########################
        # Field 1a: Upload Data and Generate Histogram
        ########################

        self.upload_button = tk.Button(root, text="Upload Data", command= self.message_upload_data, width=22, bg="lightblue")
        self.upload_button.grid(row=0, column=0, sticky=tk.W)

        self.upload_frame = tk.Frame(root)
        self.upload_frame.grid(row=0, column=1, sticky=tk.W)

        self.upload_button = tk.Button(self.upload_frame, text="Upload Data", command=self.upload_data, width=20)
        self.upload_button.grid(row=0, column=1, sticky=tk.W)

        self.generate_histogram_button = tk.Button(self.upload_frame, text="Histogram Intensity", command=self.generate_histogram, width=20)
        self.generate_histogram_button.grid(row=0, column=2, sticky=tk.W)
        
        ########################
        # Field 1b: Setting default values for mussels and sea weed
        ########################

        self.set_default_button = tk.Button(root, text="Get Defulat Values", command= self.message_get_default_values, width=22, bg="lightblue")
        self.set_default_button.grid(row=1, column=0, sticky=tk.W)

        self.set_default_frame = tk.Frame(root)
        self.set_default_frame.grid(row=1, column=1, sticky=tk.W)

        # Add Sea Weed and Mussel buttons
        self.sea_weed_button = tk.Button(self.set_default_frame, text="Sea Weed", command=self.set_default_sea_weed_values, width=20)
        self.sea_weed_button.grid(row=0, column=1, sticky=tk.W)

        self.mussel_button = tk.Button(self.set_default_frame, text="Mussel", command=self.set_default_mussel_values, width=20)
        self.mussel_button.grid(row=0, column=2, sticky=tk.W)

        ########################
        # Field 1c: Remove noise
        ########################

        self.remove_noise_button = tk.Button(root, text="Remove Noise", command= self.message_remove_noise, width=22, bg="lightblue")
        self.remove_noise_button.grid(row=2, column=0, sticky=tk.W)

        self.remove_noise_frame = tk.Frame(root)
        self.remove_noise_frame.grid(row=2, column=1, sticky=tk.W)

        self.beam_histogram_button = tk.Button(self.remove_noise_frame, text="Histogram Beam", command=self.histogram_beam, width=20)
        self.beam_histogram_button.grid(row=0, column=1, sticky=tk.W)

        self.remove_most_frequent_button = tk.Button(self.remove_noise_frame, text="Remove Most Frequent", command=self.remove_most_frequent, width=20)
        self.remove_most_frequent_button.grid(row=0, column=2, sticky=tk.W)

        self.remove_bottom_label = tk.Label(self.remove_noise_frame, text="Removing Bottom (ratio)", width=21)
        self.remove_bottom_label.grid(row=0, column=3, sticky=tk.W)

        self.remove_bottom_entry = tk.Entry(self.remove_noise_frame, width=22)
        self.remove_bottom_entry.grid(row=0, column=4, sticky=tk.W)

        self.remove_bottom_button = tk.Button(self.remove_noise_frame, text="Filter", command=self.remove_bottom_data, width=22)
        self.remove_bottom_button.grid(row=0, column=5, sticky=tk.W)        

        ########################
        # Field 2: Set Threshold and Visualize Data
        ########################

        self.threshold_button = tk.Button(root, text="Set Threshold", command= self.message_setting_threshold, width=22, bg="lightblue")
        self.threshold_button.grid(row=3, column=0, sticky=tk.W)

        self.set_threshold_frame = tk.Frame(root)
        self.set_threshold_frame.grid(row=3, column=1, sticky=tk.W)

        self.lower_threshold_label = tk.Label(self.set_threshold_frame, text="Lower Threshold", width=22)
        self.lower_threshold_label.grid(row=0, column=1, sticky=tk.W)

        self.lower_threshold_entry = tk.Entry(self.set_threshold_frame, width=22)
        self.lower_threshold_entry.grid(row=0, column=2, sticky=tk.W)

        self.upper_threshold_label = tk.Label(self.set_threshold_frame, text="Upper Threshold", width=22)
        self.upper_threshold_label.grid(row=0, column=3, sticky=tk.W)

        self.upper_threshold_entry = tk.Entry(self.set_threshold_frame, width=22)
        self.upper_threshold_entry.grid(row=0, column=4, sticky=tk.W)

        self.set_visualize_button = tk.Button(self.set_threshold_frame, text="Filter", command=self.set_and_visualize, width=22)
        self.set_visualize_button.grid(row=0, column=5, sticky=tk.W)

        ########################
        # Field 3 a : Statistical Outlier Removal
        ########################

        self.statistical_button = tk.Button(root, text="Statistical Outlier Removal",command=self.message_statistical_removal,width=22, bg="lightblue")
        self.statistical_button.grid(row=4, column=0,  sticky=tk.W)

        self.remove_Statistical_outliers_frame = tk.Frame(root, width=22)
        self.remove_Statistical_outliers_frame.grid(row=4, column=1, sticky=tk.W)

        self.nb_neighbors_label = tk.Label(self.remove_Statistical_outliers_frame, text="Number of Neighbours", width=22)
        self.nb_neighbors_label.grid(row=0, column=1, sticky=tk.W)

        self.nb_neighbors_entry = tk.Entry(self.remove_Statistical_outliers_frame, width=22)
        self.nb_neighbors_entry.grid(row=0, column=2, sticky=tk.W)

        # allows setting the threshold level based on the standard deviation of the average distances across the point cloud. 
        # The lower this number the more aggressive the filter will be.
        self.std_ratio_label = tk.Label(self.remove_Statistical_outliers_frame, text="Standard Ratio", width=22)
        self.std_ratio_label.grid(row=0, column=3, sticky=tk.W)

        self.std_ratio_entry = tk.Entry(self.remove_Statistical_outliers_frame, width=22)
        self.std_ratio_entry.grid(row=0, column=4, sticky=tk.W)

        self.statistical_button = tk.Button(self.remove_Statistical_outliers_frame, text="Perform", command=self.statistical_remove_outliers, width=22)
        self.statistical_button.grid(row=0, column=5, sticky=tk.W)

        ########################
        # Field 3 b: Radius Outlier Removal
        ########################

        self.radius_button = tk.Button(root, text="Radius Outlier Removal",command=self.message_radius_removal,width=22, bg="lightblue")
        self.radius_button.grid(row=5, column=0, sticky=tk.W)

        self.remove_Radius_outliers_frame = tk.Frame(root, width=22)
        self.remove_Radius_outliers_frame.grid(row=5, column=1, sticky=tk.W)

        # which lets you pick the minimum amount of points that the sphere should contain
        self.nb_points_label = tk.Label(self.remove_Radius_outliers_frame, text="Number of points", width=22)
        self.nb_points_label.grid(row=0, column=1, sticky=tk.W)

        self.nb_points_entry = tk.Entry(self.remove_Radius_outliers_frame, width=22)
        self.nb_points_entry.grid(row=0, column=2, sticky=tk.W)

        # which defines the radius of the sphere that will be used for counting the neighbors.
        self.radius_label = tk.Label(self.remove_Radius_outliers_frame, text="Radius", width=22)
        self.radius_label.grid(row=0, column=3, sticky=tk.W)

        self.radius_entry = tk.Entry(self.remove_Radius_outliers_frame, width=22)
        self.radius_entry.grid(row=0, column=4, sticky=tk.W)

        self.radius_button = tk.Button(self.remove_Radius_outliers_frame, text="Perform", command=self.radius_remove_outliers, width=22)
        self.radius_button.grid(row=0, column=5, sticky=tk.W)
        
        ########################
        # Field 4: K-Means Clustering
        ########################

        self.clustering_button = tk.Button(root, text="K-Means Clustering",command=self.message_k_mean_clustering, width=22, bg="lightblue")
        self.clustering_button.grid(row=6, column=0 , sticky=tk.W)

        self.density_frame = tk.Frame(root)
        self.density_frame.grid(row=6, column=1, sticky=tk.W)

        self.density_label = tk.Label(self.density_frame, text="Density-Number of Clusters:", width=22)
        self.density_label.grid(row=0, column=0, sticky=tk.W)

        self.density_clusters_entry = tk.Entry(self.density_frame, width=22)
        self.density_clusters_entry.grid(row=0, column=1, sticky=tk.W)

        self.density_clustering_button = tk.Button(self.density_frame, text="Perform", command=self.perform_Kmean_density_clustering, width=22)
        self.density_clustering_button.grid(row=0, column=2, sticky=tk.W)

        self.intensity_label = tk.Label(self.density_frame, text="Intensity-Number of Clusters:", width=22)
        self.intensity_label.grid(row=1, column=0, sticky=tk.W)

        self.intensity_clusters_entry = tk.Entry(self.density_frame, width=22)
        self.intensity_clusters_entry.grid(row=1, column=1, sticky=tk.W)

        self.intensity_clustering_button = tk.Button(self.density_frame, text="Perform", command=self.perform_Kmean_intensity_clustering, width=22)
        self.intensity_clustering_button.grid(row=1, column=2, sticky=tk.W)
        
        ########################
        # Field 5: GMM Clustering
        ########################
        
        self.gmm_button = tk.Button(root, text="GMM Clustering",command=self.message_gmm_clustering, width=22, bg="lightblue")
        self.gmm_button.grid(row=7, column=0, sticky=tk.W)

        self.density_gmm_frame = tk.Frame(root)
        self.density_gmm_frame.grid(row=7, column=1, sticky=tk.W)

        self.density_gmm_label = tk.Label(self.density_gmm_frame, text="Density-Number of Clusters:", width=22)
        self.density_gmm_label.grid(row=0, column=0, sticky=tk.W)

        self.density_gmm_components_entry = tk.Entry(self.density_gmm_frame, width=22)
        self.density_gmm_components_entry.grid(row=0, column=1, sticky=tk.W)

        self.density_gmm_clustering_button = tk.Button(self.density_gmm_frame, text="Perform", command=self.perform_density_gmm_clustering, width=22)
        self.density_gmm_clustering_button.grid(row=0, column=2, sticky=tk.W)


        self.intensity_gmm_label = tk.Label(self.density_gmm_frame, text="Intensity-Number of Clusters:", width=22)
        self.intensity_gmm_label.grid(row=1, column=0, sticky=tk.W)

        self.intensity_gmm_components_entry = tk.Entry(self.density_gmm_frame, width=22)
        self.intensity_gmm_components_entry.grid(row=1, column=1, sticky=tk.W)

        self.intensity_gmm_clustering_button = tk.Button(self.density_gmm_frame, text="Perform", command=self.perform_intensity_gmm_clustering, width=22)
        self.intensity_gmm_clustering_button.grid(row=1, column=2, sticky=tk.W)
       
        ########################
        # Field 6 : hdbscan clustering
        ########################

        self.hdbscan_button = tk.Button(root, text="Hdbscan Clustering",command=self.message_hdbscan_clustering, width=22, bg="lightblue")
        self.hdbscan_button.grid(row=8, column=0, sticky=tk.W)

        self.hdbscan_frame = tk.Frame(root)
        self.hdbscan_frame.grid(row=8, column=1)

        # The minimum number of samples in a group for that group to be considered a cluster; 
        # groupings smaller than this size will be left as noise.
        self.minClusterSize_label = tk.Label(self.hdbscan_frame, text=" Minimum Cluster Size", width=22)
        self.minClusterSize_label.grid(row=0, column=0)

        self.minClusterSize_entry = tk.Entry(self.hdbscan_frame, width=22)
        self.minClusterSize_entry.grid(row=0, column=1)

        self.minNumSamples_label = tk.Label(self.hdbscan_frame, text="Minimum Number of Samples", width=22)
        self.minNumSamples_label.grid(row=0, column=2)

        self.minNumSamples_entry = tk.Entry(self.hdbscan_frame, width=22)
        self.minNumSamples_entry.grid(row=0, column=3)

        self.hdbscan_button = tk.Button(self.hdbscan_frame, text="Perform", command=self.perform_hdbscan_clustering_density, width=22)
        self.hdbscan_button.grid(row=0, column=4)

        ########################
        # Field7: clusters
        ########################

        self.main_frame = tk.Frame(root)
        self.main_frame.grid(row=9, column=1, padx=10, pady=10)

        # Create a frame for the buttons and checkboxes
        self.button_frame = tk.Frame(self.main_frame)
        self.button_frame.grid(row=0, column=1, padx=10, pady=10, sticky=tk.W)

        # Fixed contrast colors
        self.colors_clusters = [
            "#F61a06", "#F6db06", "#58f606", "#0679f6", "#B106f6",
            "#1a9641", "#e377c2", "#91bfdb", "#fee08b", "#377916", 
            "#4bbbec", "#B59cd5", "#a55194", "#0eeef4", "#E7ff08",
            "#ff7f00", "#4d4d4d", "#Aa3a13", "#98df8a", "#28708f",
            "#728307", "#03ffae", "#F50bf8", "#F0b010", "#E6156b",
            "#4575b4", "#313695", "#F6a3df", "#682805", "#D9e285",
            "#A6fd7c", "#052966", "#Fff303", "#Adf2eb", "#D5759e",
            "#Ff5504", "#B2b0ae", "#F0fad2", "#39bf76", "#3f075e"]

        # Create 40 buttons representing clusters with fixed colors
        for cluster_id, color in enumerate(self.colors_clusters, start=1):
            button = tk.Button(self.button_frame, text=f"Cluster {cluster_id}", command=lambda cluster_id=cluster_id: self.choose_cluster(cluster_id), bg=color, width=10, height=2)
            button.grid(row=(cluster_id - 1) % 10, column=(cluster_id - 1) // 10 * 2, padx=5, pady=5, sticky=tk.W)

            # Create a checkbox for volume beside each cluster
            checkbox_var = tk.IntVar()
            checkbox = tk.Checkbutton(self.button_frame, variable=checkbox_var)
            checkbox.grid(row=(cluster_id - 1) % 10, column=(cluster_id - 1) // 10 * 2 + 1, padx=5, pady=5, sticky=tk.W)
     
        self.total_volume_frame = tk.Frame(root)
        self.total_volume_frame.grid(row=10, column=1)
        self.reset_volume = tk.Button(self.total_volume_frame,text="Reset Volume", command= self.reset_volume_function, width=22,  height = 2)
        self.reset_volume.grid(row=0, column=0, padx=10, pady=10)  
        self.total_volume_add = tk.Button(self.total_volume_frame,text="Add to Volume", command= self.append_volume, width=22,  height = 2)
        self.total_volume_add.grid(row=0, column=1, padx=10, pady=10) 
        self.total_volume_button = tk.Button(self.total_volume_frame, text="Total Volume", command= self.calculate_total_volume, width=22,  height = 2)
        self.total_volume_button.grid(row=0, column=2, padx=10, pady=10) 
        self.show_total_volume = tk.Text(self.total_volume_frame, width=22, height=2)  
        self.show_total_volume.grid(row=0, column=3, padx=10, pady=10)  

        ########################
        # Field 8: Interpolation
        ########################

        self.interpolation_frame = tk.Frame(root)
        self.interpolation_frame.grid(row=9, column=0)

        self.interpolation_button = tk.Button(self.interpolation_frame, text=" Interpolation",command=self.message_interpolation, width=22, bg="lightblue")
        self.interpolation_button.grid(row=0, column=0, sticky=tk.W)

        self.interpolation_button = tk.Button(self.interpolation_frame, text="Start",command= self.interpolate_data_btw_planes, width = 22)
        self.interpolation_button.grid(row=0, column=1, pady=40)

        ########################
        # Field 8: Voxelization
        ########################

        self.voxelization_button = tk.Button(self.interpolation_frame, text="Voxelization",command=self.message_voxelization, width=22, bg="lightblue")
        self.voxelization_button.grid(row=1, column=0, sticky=tk.W)

        self.voxelization_size_label = tk.Label(self.interpolation_frame, text=" The voxel size", width=22)
        self.voxelization_size_label.grid(row=2, column=0)

        self.voxel_size_entry = tk.Entry(self.interpolation_frame, width=22)
        self.voxel_size_entry.grid(row=2, column=1)

        self.checkbox_var = tk.BooleanVar()
        self.checkbox = tk.Checkbutton(self.interpolation_frame,text= 'Weighted Voxelization', variable = self.checkbox_var)
        self.checkbox.grid(row=3, column=0, pady=20)

        self.Lower_Threshold_label = tk.Label(self.interpolation_frame, text=" Lower Threshold", width=22)
        self.Lower_Threshold_label.grid(row=4, column=0)

        self.Lower_Threshold_entry = tk.Entry(self.interpolation_frame, width=22)
        self.Lower_Threshold_entry.grid(row=4, column=1)

        self.Upper_Threshold_label = tk.Label(self.interpolation_frame, text=" Upper Threshold", width=22)
        self.Upper_Threshold_label.grid(row=5, column=0)

        self.Upper_Threshold_entry = tk.Entry(self.interpolation_frame, width=22)
        self.Upper_Threshold_entry.grid(row=5, column=1)

        self.Voxelization_button = tk.Button(self.interpolation_frame, text="Voxelize", command=self.voxelize, width=22)
        self.Voxelization_button.grid(row=6, column=0, pady=60) 

        self.Volume_button = tk.Button(self.interpolation_frame, text="Caculate Volume", command=self.calculate_volume_voxelization, width=22)
        self.Volume_button.grid(row=6, column=1, pady=60) 

        ####################
        # Save point cloud
        ####################
        self.save_point_cloud = tk.Button(self.interpolation_frame, text="Saving the Current Point Cloud", command=self.saving_point_cloud, width = 30, height= 4, padx=100)
        self.save_point_cloud.grid(row=7, column=0, columnspan=2)

        # Button to Access Previous Point Cloud
        #self.count = 0
        self.access_previous_button = tk.Button(self.interpolation_frame, text="Access Previous Point Cloud", command=self.access_previous_point_cloud, width = 30, height= 4, padx=100)
        self.access_previous_button.grid(row=8, column=0, columnspan=2)

        # Text Box for Information
        self.text_frame = tk.Frame(root)
        self.text_frame.grid(row=0, column= 2,rowspan =30, columnspan=2)
        self.text_box = tk.Text( self.text_frame,height = 40)
        self.text_box.grid(row=0, column= 0,rowspan =20,columnspan=3, pady = 100)
        self.text_delete = tk.Button( self.text_frame, text="Delete!", command=self.delete_text, width=22)
        self.text_delete.grid(row=40, column=0)

    def clear_entry(self):
        # Delete all default values
        self.lower_threshold_entry.delete(0, tk.END)
        self.upper_threshold_entry.delete(0, tk.END)
        self.nb_neighbors_entry.delete(0, tk.END)
        self.std_ratio_entry.delete(0, tk.END)
        self.nb_points_entry.delete(0, tk.END)
        self.radius_entry.delete(0, tk.END)
        self.density_clusters_entry.delete(0, tk.END)
        self.intensity_clusters_entry.delete(0, tk.END)
        self.density_gmm_components_entry.delete(0, tk.END)
        self.intensity_gmm_components_entry.delete(0, tk.END)
        self.minClusterSize_entry.delete(0, tk.END)
        self.minNumSamples_entry.delete(0, tk.END)
        self.Upper_Threshold_entry.delete(0, tk.END)
        self.Lower_Threshold_entry.delete(0, tk.END)

    def set_default_sea_weed_values(self):

        # Clear all the default values to set the new one
        self.clear_entry()

        # Default threshold for filtering
        self.default_low_threshold = -58
        self.lower_threshold_entry.insert(0, self.default_low_threshold)
        self.default_high_threshold = -30
        self.upper_threshold_entry.insert(0, self.default_high_threshold)

        # Default parameters for outlier removal
        self.default_nb_neighbors = 1000
        self.nb_neighbors_entry.insert(0, self.default_nb_neighbors)
        self.default_std_ratio = 2.0
        self.std_ratio_entry.insert(0, self.default_std_ratio)
        self.default_nb_points = 1000
        self.nb_points_entry.insert(0, self.default_nb_points) 
        self.default_radius = 0.5
        self.radius_entry.insert(0,self.default_radius)

        # Default number of clusters
        self.default_num_clusters = 2
        self.density_clusters_entry.insert(0, self.default_num_clusters)
        self.intensity_clusters_entry.insert(0, self.default_num_clusters)
        self.density_gmm_components_entry.insert(0, self.default_num_clusters)
        self.intensity_gmm_components_entry.insert(0, self.default_num_clusters)
        self.default_minClusterSize = 10000
        self.minClusterSize_entry.insert(0, self.default_minClusterSize)
        self.default_minNumSamples = 1000
        self.minNumSamples_entry.insert(0, self.default_minNumSamples)

        #Default parameters for meshing
        self.default_radius = 0.1
        self.default_max_nn = 30
        self.default_depth = 8
        self.default_voxel_size = 0.1

        self.default_Upper_Threshold = 20
        self.Upper_Threshold_entry.insert(0, self.default_Upper_Threshold)
        self.default_Lower_Threshold = 1
        self.Lower_Threshold_entry.insert(0, self.default_Lower_Threshold)

        # Show message box after finishing the uploading
        messagebox.showinfo("Default values for Seaweed set!", "Go to 'Set Threshold' section and change the values if needed.Then push the 'Fiter' button.")

    def set_default_mussel_values(self):
        # Clear all the default values to set the new one
        self.clear_entry()
        
        # Default threshold for filtering
        self.default_low_threshold = 20
        self.lower_threshold_entry.insert(0, self.default_low_threshold)
        self.default_high_threshold = 2000
        self.upper_threshold_entry.insert(0, self.default_high_threshold)

        # Default parameters for outlier removal
        self.default_nb_neighbors = 100
        self.nb_neighbors_entry.insert(0, self.default_nb_neighbors)
        self.default_std_ratio = 2.0
        self.std_ratio_entry.insert(0, self.default_std_ratio)
        self.default_nb_points = 100
        self.nb_points_entry.insert(0, self.default_nb_points) 
        self.default_radius = 0.5
        self.radius_entry.insert(0,self.default_radius)

        # Default number of clusters
        self.default_num_clusters = 2
        self.density_clusters_entry.insert(0, self.default_num_clusters)
        self.intensity_clusters_entry.insert(0, self.default_num_clusters)
        self.density_gmm_components_entry.insert(0, self.default_num_clusters)
        self.intensity_gmm_components_entry.insert(0, self.default_num_clusters)
        self.default_minClusterSize = 300
        self.minClusterSize_entry.insert(0, self.default_minClusterSize)
        self.default_minNumSamples = 200
        self.minNumSamples_entry.insert(0, self.default_minNumSamples)

        #Default parameters for meshing
        self.default_radius = 0.1
        self.default_max_nn = 30
        self.default_depth = 8
        self.default_voxel_size = 0.1   

        self.default_Upper_Threshold = 20
        self.Upper_Threshold_entry.insert(0, self.default_Upper_Threshold)
        self.default_Lower_Threshold = 1
        self.Lower_Threshold_entry.insert(0, self.default_Lower_Threshold)

        # Show message box after removing bottom data
        messagebox.showinfo("Default values for Mussel set!", "Go to 'Remove Noise' section and check the histogram of beam, if filtering is needed.")

    def delete_text(self):
        self.text_box.delete(1.0, tk.END)

    def show_instruction_message(self):
        instruction = "Welcome to the Data Visualization App!\nPlease follow the instructions to proceed. \nUpload data (.txt, .csv) by clicking on 'Upload Data' button.\n\nYou can visualize the histogram of intensity by pushing 'Histogram Intensity' button."
        messagebox.showinfo("Instructions", instruction)    

    def show_cluster_info(self, cluster_id):
        # Replace this with your logic to get and display cluster information
        cluster_info = (cluster_id)
        messagebox.showinfo(f"Cluster {cluster_id} Information", cluster_info)

    def print_info(self, message):
        self.text_box.insert(tk.END, message + "\n")
        self.text_box.see(tk.END)  # Scroll to the end of the text

    def reset_volume_function(self):
        self.total_volume = 0 

    def saving_point_cloud(self):
        if self.point_cloud is None:
            print("No point cloud to save.")
            return

        # Ask the user for the file name and directory
        file_path = filedialog.asksaveasfilename(defaultextension=".pcd", filetypes=[("PCD files", "*.ply"), ("All files", "*.*")])

        if file_path:
            # Save the point cloud to the specified file
            o3d.io.write_point_cloud(file_path, self.point_cloud)
            print(f"Point cloud saved to {file_path}")

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
            self.volume_values = [] #List to store volumes of clusters

            ## add the point cloud to the point_cloud_history_list
            self.point_cloud_history.append(self.point_cloud)   
            print(self.df.head())  

            # Show message box after finishing the uploading
            messagebox.showinfo("Data Uploaded!", "Data uploaded! You can see the histogram of intensity by pushing 'Histogram Intensity'.\nThen you can choose the corresponding species to set all default values for parameters.")
    
    def append_volume(self):
        self.volume_values.append(self.voxel_volume)

    def message_upload_data(self):
        messagebox.showinfo("Data Uploade", "You can upload your data in .csv and .txt format. The data should have X, Y, Z, intensity, ping_number and beam columns.\n\nYou can see the histogram of intensity by pushing 'Histogram Intensity'.")
    
    def message_get_default_values(self):
        messagebox.showinfo("Get Default Values", "In this part you can choose the corresponding species and set default values for all parameters.")
    
    def message_remove_noise(self):
        messagebox.showinfo("Remove Noise", "This part is for removing structural noise.\n\nFirst you can visualize the histogram of Beam numbers. In case of Noise the number of a beam would be significantly higher than other beams. If there was such a case you can remove it by pusshing 'Remove Most Frequent' button. \n\nIn some cases the data of the buttom because of scattering of buttom of the profile can cause noise. You can choose the rartio of bottom data which you would like to remove from data by inserting the ratiro and pussing filter button.")
    
    def message_setting_threshold(self):
        messagebox.showinfo("Setting Threshold","In this part you can change the threshold if needed.Then push the 'Fiter' button.")
    
    def message_statistical_removal(self):
        messagebox.showinfo("Statistical Outlier Removal", "identifies and eliminates data points that are further away from their neighbours compared to the average points. Two parameters should set: \n\nNumber of neighbors: specifies how many neighbors are taken into account in order to calculate the average distance for a given point\n\nStandard ratio: allows setting the threshold level based on the standard deviation of the average distances across the point cloud. \nThe lower this number the more aggressive the filter will be.")
    
    def message_radius_removal(self):
        messagebox.showinfo("Radius Outlier Removal", "removes points that have few neighbors in a given sphere around them. Two parameters can be used to tune the filter to your data:\n\nNumber of points: which lets you pick the minimum amount of points that the sphere should contain\n\nRadius: which defines the radius of the sphere that will be used for counting the neighbors.")
    
    def message_k_mean_clustering(self):
        messagebox.showinfo("K_Means clustering","K-Means is like a sorting algorithm for data. It automatically groups similar points together, creating clusters. \n\nIn our app, it organizes points based on their positions and intensities. You can check to see which one works better for your case.\n\nYou can choose the number of cluster, then the algorithm sort your point in the defined number of groups.")
    
    def message_gmm_clustering(self):
        messagebox.showinfo("GMM Clustering","GMM clustering is another clustering method which looks for hidden pattern with less rigid clusters. It assumes that the data is generated by a mix of several Gaussian distribution.\n\n In our app, it organizes points based on their positions and intensities. You can check to see which one works better for your case.\n\nYou can choose the number of cluster, then the algorithm sort your point in the defined number of groups.")
    
    def message_hdbscan_clustering(self):
        messagebox.showinfo("Hdbscan Clustering","It stands for Hierarchical Density-Based Spatial Clustering of applications with Noise.\n It identifies clusters based on the density of data points. Clusters are areas where points are closer together.\n\nMinimum Cluster Size: The smallest number of points required to form a cluster. Adjust this if you want smaller or larger clusters.\n\nMinimum Samples: The minimum number of points needed to consider a region dense. Play with this to control sensitivity.  ")

    def message_interpolation(self):
        messagebox.showinfo("Interpolation","Estimate points that fall between known data points to fill the gap between pings.")

    def message_voxelization(self):
        messagebox.showinfo("Voxelization","Converting point cloud into a regular grid of volumetric pixels (voxels) to calculate the volume of the detected object by counting the number of voxels. \n\nWeighted voxelization extends the concept of voxelization by assigning weights to each voxel based on the number of points they contain. \n\nInsert higher threshold and lower threshold to continue. \n\nLower Threshold: the minimum number of points required for a voxel to be considered valid \n\n Upper Threshold: the minimum point count at which a voxel is given full weight \n\nWeight calculation: Voxels with point counts between the lower and upper thresholds are assigned weights that gradually increase, reflecting the data's significance within the voxel.")

    def generate_histogram(self):
        BeamImager.histogram(self.intensity_data, "Intensity", num_bins=200)

    def histogram_beam(self):
        BeamImager.histogram(self.df["beam"], "Beam", num_bins=200)
    
    def remove_most_frequent(self):
        # Find the most frequent value in the specified column
        most_frequent_value = self.df['beam'].mode().iloc[0]

        # Count rows before removal
        rows_before_removal = len(self.df)

        # Remove rows where the specified column has the most frequent value
        df_filtered = self.df[self.df['beam'] != most_frequent_value]

        # Count rows after removal
        rows_after_removal = len(df_filtered)

        # Display or use the most frequent value
        print(f'Most Frequent Value: {most_frequent_value}')

        # Display the number of rows removed
        rows_removed = rows_before_removal - rows_after_removal
        print(f'Rows Removed: {rows_removed}')
        self.df = df_filtered.copy()

        # Show message box after finishing the uploading
        messagebox.showinfo(f'{rows_removed} rows Removed!', "Go and remove data from the bottom if needed.")

    def remove_bottom_data(self):
        percent = float(self.remove_bottom_entry.get())

        self.filtered_df, self.removed_df = BeamImager.filter_noise(self.df, percent)
        BeamImager.display_inlier_outlier_bottom(self.filtered_df, self.removed_df)   
        self.df = self.filtered_df.copy()
        print(self.df.head())
        print(len(self.df))

        # Show message box after removing bottom data
        messagebox.showinfo("Data filtered!", "Go to 'Set Threshold' section and change the values if needed.Then push the 'Fiter' button.")

    def set_and_visualize(self):
        lower_threshold = float(self.lower_threshold_entry.get())
        upper_threshold = float(self.upper_threshold_entry.get())

        if lower_threshold is not None and upper_threshold is not None:
            # Apply the thresholding and visualize the data
            ## Masking data based on the threshold
            self.mask = BeamImager.mask_threshold(self.df, lower_threshold, upper_threshold)

            ## Choosing a color map for visualizing intensity
            self.colors = BeamImager.color_visualization_pointCloud(BeamImager.normalize_intensity(self.df['intensity'])[0])

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

            # Show message box after filtering
            messagebox.showinfo("Data filtered!", "It's time to remove outliers. There are two methods here. Statistical Outlier Removal and Radius Outlier Removal. To get information on each method push the coressponded button.")
   
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
        self.cluster_clouds = BeamImager.visualize_clusters(self.clustered_point_clouds, self.colors_clusters)

    def perform_Kmean_intensity_clustering(self):
        # Perform intensity-based K-Mean clustering 
        num_clusters = int(self.intensity_clusters_entry.get()) if self.intensity_clusters_entry.get() else self.default_num_clusters
        self.clustered_point_clouds = BeamImager.clustering_k_mean_colors( self.point_cloud, num_clusters) 
        self.cluster_clouds = BeamImager.visualize_clusters(self.clustered_point_clouds, self.colors_clusters)

    def perform_density_gmm_clustering(self):
        # Perform density-based GMM clustering 
        num_clusters = int(self.density_gmm_components_entry.get()) if self.density_gmm_components_entry.get() else self.default_num_clusters
        self.clustered_point_clouds = BeamImager.gmm_clustering_with_points( self.point_cloud, num_clusters) 
        self.cluster_clouds = BeamImager.visualize_clusters(self.clustered_point_clouds, self.colors_clusters)

    def perform_intensity_gmm_clustering(self):
        # Perform intensity-based GMM clustering 
        num_clusters = int(self.intensity_gmm_components_entry.get()) if self.intensity_gmm_components_entry.get() else self.default_num_clusters
        self.clustered_point_clouds = BeamImager.gmm_clustering_with_intensity( self.point_cloud, num_clusters) 
        self.cluster_clouds = BeamImager.visualize_clusters(self.clustered_point_clouds, self.colors_clusters)

    def perform_hdbscan_clustering_density (self):
        # Perform HDBSCAN clustering based on spatial coordinates
        min_cluster_size = int(self.minClusterSize_entry.get()) if self.minClusterSize_entry.get() else self.default_minClusterSize
        min_samples = int(self.minNumSamples_entry.get()) if self.minNumSamples_entry.get() else self.default_minNumSamples
        self.clustered_point_clouds = BeamImager.hdbscan_clustering_for_point_clouds(self.point_cloud, min_cluster_size, min_samples)
        self.cluster_clouds = BeamImager.visualize_clusters(self.clustered_point_clouds, self.colors_clusters)

    def choose_cluster(self, id):
        point_cloud = None  # Initialize point_cloud to None
        
        if self.cluster_clouds is not None and 0 < id <= len(self.cluster_clouds):
            clusters_original = BeamImager.visualize_selected_cluster(self.cluster_clouds, id - 1)
            point_cloud = clusters_original[id - 1]  
            self.cluster_clouds = clusters_original

        if point_cloud is not None:
            # Visualize the selected cluster
            self.point_cloud = point_cloud
            o3d.visualization.draw_geometries([self.point_cloud])

        else:
            print("Invalid cluster ID or cluster_clouds is None.")   
        
    def interpolate_data_btw_planes (self):
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
        self.voxel_size = float(self.voxel_size_entry.get()) if self.voxel_size_entry.get() else self.default_voxel_size
        self.voxel_grid, self.unique_voxels, self.voxel_counts = BeamImager.count_point_per_voxel(self.point_cloud, self.voxel_size)
        o3d.visualization.draw_geometries([self.voxel_grid, self.point_cloud])
        self.print_info(f"Average number of count per voxel is:{self.voxel_counts.mean()}")
        self.print_info(f"The number of count that 10% of the voxels are equal to or below:{np.percentile(self.voxel_counts, 10)}")
        self.print_info(f"Miinimum number of points in a voxel is:{self.voxel_counts.min()}")
        self.print_info(f"Maximum number of points in a voxel is:{self.voxel_counts.max()}")

    def calculate_volume_voxelization(self):
        self.threshold_count_up= int(self.Upper_Threshold_entry.get()) if self.Upper_Threshold_entry.get() else self.default_Upper_Threshold
        self.threshold_count_down = int(self.Lower_Threshold_entry.get()) if self.Lower_Threshold_entry.get() else self.default_Lower_Threshold

        if self.checkbox_var.get() is False:
            # Calculate the volume of the voxelized point cloud
            self.voxel_volume = len(self.voxel_grid.get_voxels()) * self.voxel_size**3

        else:  
            self.dense_point_cloud, self.voxel_volume = BeamImager.custom_voxelization(self.voxel_grid,self.unique_voxels,self.voxel_counts, self.voxel_size, self.threshold_count_up , self.threshold_count_down)

        self.print_info(f"Volume of the point cloud (voxelized):{self.voxel_volume}")

    def calculate_total_volume(self):
        self.total_volume = sum(self.volume_values)  

        # Display the result in the Text widget
        self.show_total_volume.config(state=tk.NORMAL)  # Enable editing
        self.show_total_volume.delete(1.0, tk.END)  # Clear existing text
        self.show_total_volume.insert(tk.END, f"Sum: {self.total_volume:.2f}")
        self.show_total_volume.config(state=tk.DISABLED)  # Disable editing  

    def access_previous_point_cloud(self):

        if len(self.point_cloud_history)>1:

            # Access the most recent point cloud
            previous_point_cloud = self.point_cloud_history[len(self.point_cloud_history)-2]
            del self.point_cloud_history[-1]
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
    root.geometry("1800x1000")  # the dimensions of window

    app = DataVisualizationApp(root)
    root.mainloop()
