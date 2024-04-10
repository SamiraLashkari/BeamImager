import tkinter as tk
from tkinter import filedialog
import open3d as o3d
import BeamImager
import pandas as pd
import numpy as np
from tkinter import * 
from tkinter.ttk import *
from tkinter import messagebox
from sklearn.decomposition import PCA
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.path import Path
from collections import Counter

class CurveDrawingApp:
    def __init__(self, master, point_cloud):
        self.point_cloud = point_cloud
        self.master = master
    
        # Convert to numpy array
        data = np.asarray(point_cloud.points)
        self.original_color = np.asarray(point_cloud.colors) * 255.0
        self.points_3d = data.copy()

        # Apply PCA to find the plane and project to 2D
        pca = PCA(n_components=3)
        pca.fit(self.points_3d)
        points_2d = pca.transform(self.points_3d)[:, :2]
        self.points_2d = points_2d
        plt.close('all')

        self.fig, self.ax = plt.subplots()
        self.scatter = self.ax.scatter(points_2d[:, 0], points_2d[:, 1], c='blue')

        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas.draw()

        self.draw_button = tk.Button(master, text="Draw Curve", command=self.start_curve_drawing)
        self.draw_button.pack()

        self.assign_labels_button = tk.Button(master, text="Assign Labels", command=self.assign_labels, state=tk.DISABLED)
        self.assign_labels_button.pack()

        self.finish_button = tk.Button(master, text="Finish", command=self.finish_curve_drawing, state=tk.DISABLED)
        self.finish_button.pack()

        self.curve_points = []

        # Additional attributes for storing labels
        self.labels_assigned = set()
        self.points_3d_with_labels = np.zeros((len(self.points_3d), 4), dtype=np.float64)
        self.points_3d_with_labels[:, :3] = np.asarray(self.point_cloud.points)
        self.points_3d_with_labels[:, 1] = -1  # Initialize labels as -1

    def start_curve_drawing(self):
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.draw_button.config(state=tk.DISABLED)
        self.finish_button.config(state=tk.NORMAL)
        self.assign_labels_button.config(state=tk.NORMAL)

    def on_click(self, event):
        x, y = event.xdata, event.ydata
        if x is not None and y is not None:
            self.curve_points.append([x, y])
            self.ax.plot([p[0] for p in self.curve_points], [p[1] for p in self.curve_points], color='red')
            self.canvas.draw()

    def finish_curve_drawing(self):
        self.fig.canvas.mpl_disconnect(self.cid)
        self.finish_button.config(state=tk.DISABLED)
        self.assign_labels_button.config(state=tk.NORMAL)
        #self.visualize_button.config(state=tk.NORMAL)  # Enable the Visualize Labels button
        self.label_counts = Counter(self.points_3d_with_labels[:, -1])
        # Print the counts
        for label, count in self.label_counts.items():
            print(f"{label}: {count}")

        # Create a copy of the original 3D points with labels
        self.labeled_points = np.copy(self.points_3d_with_labels)

        # Extract XYZ coordinates and labels
        xyz = self.labeled_points[:, :3]
        
        labels = self.labeled_points[:, -1]
        labels = [int(value) for value in labels]

        num_labels = len(set(labels)) - (1 if -1 in labels else 0)

        # Create a list to store separate clusters
        self.clustered_point_clouds = [[] for _ in range(num_labels)]

        for i, label in enumerate(labels):
            point = np.asarray(self.point_cloud.points)[i]  # Extract spatial coordinates (x, y, z)
            intensity = np.asarray(self.original_color)[i]  # Intensity values (R, G, B)
            cluster_name = f"Cluster {label }"  # Assign cluster name based on order
            self.clustered_point_clouds[label].append((point, intensity, cluster_name))    

    def assign_labels(self):
        # Separate data on either side of the curve
        mask = self.get_points_on_side(self.points_2d, self.curve_points)

        # Placeholder logic: Assign a new label
        new_label = max(self.labels_assigned, default=0) + 1
        self.labels_assigned.add(new_label)

        # Ensure mask is of boolean type
        mask = np.array(mask, dtype=bool)

        # Ensure labels is of integer type
        labels = np.array(self.points_3d_with_labels[:, -1], dtype=int)

        # Update the original 3D data with labels
        labels[mask] = new_label
        self.points_3d_with_labels[:, -1] = labels

        # Update scatter plot for the assigned label
        self.scatter.set_color(['red' if p else 'blue' for p in mask])
        self.canvas.draw()

        # Disable assign labels button after assigning
        self.assign_labels_button.config(state=tk.DISABLED)

        # Enable next label button if there are more labels to assign
        if len(self.labels_assigned) < len(set(np.unique(self.points_3d_with_labels[:, -1]))):
            # Reset curve drawing for the next label
            self.curve_points = []
            self.scatter.set_color(['red' if p else 'blue' for p in mask])  # Restore original colors
            self.canvas.draw()

            # Enable draw button for the next label
            self.draw_button.config(state=tk.NORMAL)

            # Enable assign labels button for the next label
            self.assign_labels_button.config(state=tk.NORMAL)

            # Enable visualize button if all labels are assigned
            if len(self.labels_assigned) == len(set(np.unique(self.points_3d_with_labels[:, -1]))):
                self.visualize_button.config(state=tk.NORMAL)

    def get_clustred_point_clouds(self):
        return(self.clustered_point_clouds)

    def get_points_on_side(self, data, curve_points):
        # Create a path using the drawn curve points
        path = Path(curve_points)

        # Check if each point in the 2D data is on the positive side of the curve
        mask = path.contains_points(data)

        return mask

class DataVisualizationApp:
    def __init__(self, master):
        
        self.master = master
        self.master.title("Multibeam Imaging App")
        self.processing = False  # Flag to track whether data processing is ongoing
        self.cluster_clouds = None 
        
        # Display instruction message box when opening the GUI
        self.show_instruction_message()

        ########################
        # Field 1a: Upload Data and Generate Histogram
        ########################

        self.upload_button = tk.Button(master, text="Upload Data", command= self.message_upload_data, width=22, bg="lightblue")
        self.upload_button.grid(row=0, column=0, sticky=tk.W)

        self.upload_frame = tk.Frame(master)
        self.upload_frame.grid(row=0, column=1, sticky=tk.W)

        self.upload_button = tk.Button(self.upload_frame, text="Upload Data", command=self.upload_data, width=20)
        self.upload_button.grid(row=0, column=1, sticky=tk.W)

        self.generate_histogram_button = tk.Button(self.upload_frame, text="Histogram Intensity", command=self.generate_histogram, width=20)
        self.generate_histogram_button.grid(row=0, column=2, sticky=tk.W)
        
        ########################
        # Field 1b: Setting default values for mussels and sea weed
        ########################

        self.set_default_button = tk.Button(master, text="Get Defulat Values", command= self.message_get_default_values, width=22, bg="lightblue")
        self.set_default_button.grid(row=1, column=0, sticky=tk.W)

        self.set_default_frame = tk.Frame(master)
        self.set_default_frame.grid(row=1, column=1, sticky=tk.W)

        # Add Sea Weed and Mussel buttons
        self.sea_weed_button = tk.Button(self.set_default_frame, text="Sea Weed", command=self.set_default_sea_weed_values, width=20)
        self.sea_weed_button.grid(row=0, column=1, sticky=tk.W)

        self.mussel_button = tk.Button(self.set_default_frame, text="Mussel", command=self.set_default_mussel_values, width=20)
        self.mussel_button.grid(row=0, column=2, sticky=tk.W)

        ########################
        # Field 1c: Remove noise
        ########################

        self.remove_noise_button = tk.Button(master, text="Remove Noise", command= self.message_remove_noise, width=22, bg="lightblue")
        self.remove_noise_button.grid(row=2, column=0, sticky=tk.W)

        self.remove_noise_frame = tk.Frame(master, pady=5)
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

        self.threshold_button = tk.Button(master, text="Set Threshold", command= self.message_setting_threshold, width=22, bg="lightblue")
        self.threshold_button.grid(row=3, column=0, sticky=tk.W)

        self.set_threshold_frame = tk.Frame(master, bg='#b35d62')
        self.set_threshold_frame.grid(row=3, column=1, sticky=tk.W)

        self.lower_threshold_label = tk.Label(self.set_threshold_frame, text="Lower Threshold", width=22, bg='#b35d62')
        self.lower_threshold_label.grid(row=0, column=1, sticky=tk.W)

        self.lower_threshold_entry = tk.Entry(self.set_threshold_frame, width=22)
        self.lower_threshold_entry.grid(row=0, column=2, sticky=tk.W)

        self.upper_threshold_label = tk.Label(self.set_threshold_frame, text="Upper Threshold", width=22, bg='#b35d62')
        self.upper_threshold_label.grid(row=0, column=3, sticky=tk.W)

        self.upper_threshold_entry = tk.Entry(self.set_threshold_frame, width=22)
        self.upper_threshold_entry.grid(row=0, column=4, sticky=tk.W)

        self.set_visualize_button = tk.Button(self.set_threshold_frame, text="Filter", command=self.set_and_visualize, width=22, bg='#b35d62')
        self.set_visualize_button.grid(row=0, column=5, sticky=tk.W)

        ########################
        # Field 3 a : Statistical Outlier Removal
        ########################

        self.statistical_button = tk.Button(master, text="Statistical Outlier Removal",command=self.message_statistical_removal,width=22, bg="lightblue")
        self.statistical_button.grid(row=4, column=0,  sticky=tk.W)

        self.remove_Statistical_outliers_frame = tk.Frame(master, width=22, bg = '#728a69')
        self.remove_Statistical_outliers_frame.grid(row=4, column=1, sticky=tk.W)

        self.nb_neighbors_label = tk.Label(self.remove_Statistical_outliers_frame, text="Number of Neighbours", width=22, bg = '#728a69')
        self.nb_neighbors_label.grid(row=0, column=1, sticky=tk.W)

        self.nb_neighbors_entry = tk.Entry(self.remove_Statistical_outliers_frame, width=22)
        self.nb_neighbors_entry.grid(row=0, column=2, sticky=tk.W)

        # allows setting the threshold level based on the standard deviation of the average distances across the point cloud. 
        # The lower this number the more aggressive the filter will be.
        self.std_ratio_label = tk.Label(self.remove_Statistical_outliers_frame, text="Standard Ratio", width=22, bg = '#728a69')
        self.std_ratio_label.grid(row=0, column=3, sticky=tk.W)

        self.std_ratio_entry = tk.Entry(self.remove_Statistical_outliers_frame, width=22)
        self.std_ratio_entry.grid(row=0, column=4, sticky=tk.W)
        self.statistical_button = tk.Button(self.remove_Statistical_outliers_frame, text="Filter", command=self.statistical_remove_outliers, width=22, bg = '#728a69')
        self.statistical_button.grid(row=0, column=5, sticky=tk.W)

        ########################
        # Field 3 b: Radius Outlier Removal
        ########################

        self.radius_button = tk.Button(master, text="Radius Outlier Removal",command=self.message_radius_removal,width=22, bg="lightblue")
        self.radius_button.grid(row=5, column=0, sticky=tk.W)

        self.remove_Radius_outliers_frame = tk.Frame(master, width=22, bg = '#728a69')
        self.remove_Radius_outliers_frame.grid(row=5, column=1, sticky=tk.W)

        # let you pick the minimum amount of points that the sphere should contain
        self.nb_points_label = tk.Label(self.remove_Radius_outliers_frame, text="Number of points", width=22, bg = '#728a69')
        self.nb_points_label.grid(row=0, column=1, sticky=tk.W)

        self.nb_points_entry = tk.Entry(self.remove_Radius_outliers_frame, width=22)
        self.nb_points_entry.grid(row=0, column=2, sticky=tk.W)

        # define the radius of the sphere that will be used for counting the neighbors.
        self.radius_label = tk.Label(self.remove_Radius_outliers_frame, text="Radius", width=22, bg = '#728a69')
        self.radius_label.grid(row=0, column=3, sticky=tk.W)

        self.radius_entry = tk.Entry(self.remove_Radius_outliers_frame, width=22)
        self.radius_entry.grid(row=0, column=4, sticky=tk.W)

        self.radius_button = tk.Button(self.remove_Radius_outliers_frame, text="Filter", command=self.radius_remove_outliers, width=22, bg = '#728a69')
        self.radius_button.grid(row=0, column=5, sticky=tk.W)
        
        ########################
        # Field 4: K-Means Clustering
        ########################

        self.clustering_button = tk.Button(master, text="K-Means Clustering",command=self.message_k_mean_clustering, width=22, bg="lightblue")
        self.clustering_button.grid(row=6, column=0 , sticky=tk.W)

        self.density_frame = tk.Frame(master, pady=10, bg = "#dec231")
        self.density_frame.grid(row=6, column=1, sticky=tk.W)

        self.density_label = tk.Label(self.density_frame, text="Density-Number of Clusters:", width=22, bg = "#dec231")
        self.density_label.grid(row=0, column=0, sticky=tk.W)

        self.density_clusters_entry = tk.Entry(self.density_frame, width=22)
        self.density_clusters_entry.grid(row=0, column=1, sticky=tk.W)

        self.density_clustering_button = tk.Button(self.density_frame, text="Perform", command=self.perform_Kmean_density_clustering, width=22, bg = "#dec231")
        self.density_clustering_button.grid(row=0, column=2, sticky=tk.W)

        self.intensity_label = tk.Label(self.density_frame, text="Intensity-Number of Clusters:", width=22, bg = "#dec231")
        self.intensity_label.grid(row=1, column=0, sticky=tk.W)

        self.intensity_clusters_entry = tk.Entry(self.density_frame, width=22)
        self.intensity_clusters_entry.grid(row=1, column=1, sticky=tk.W)

        self.intensity_clustering_button = tk.Button(self.density_frame, text="Perform", command=self.perform_Kmean_intensity_clustering, width=22, bg = "#dec231")
        self.intensity_clustering_button.grid(row=1, column=2, sticky=tk.W)
        
        ########################
        # Field 5: GMM Clustering
        ########################
        
        self.gmm_button = tk.Button(master, text="GMM Clustering",command=self.message_gmm_clustering, width=22, bg="lightblue")
        self.gmm_button.grid(row=7, column=0, sticky=tk.W)

        self.density_gmm_frame = tk.Frame(master, pady=5, bg = "#dec231")
        self.density_gmm_frame.grid(row=7, column=1, sticky=tk.W)

        self.density_gmm_label = tk.Label(self.density_gmm_frame, text="Density-Number of Clusters:", width=22, bg = "#dec231")
        self.density_gmm_label.grid(row=0, column=0, sticky=tk.W)

        self.density_gmm_components_entry = tk.Entry(self.density_gmm_frame, width=22)
        self.density_gmm_components_entry.grid(row=0, column=1, sticky=tk.W)

        self.density_gmm_clustering_button = tk.Button(self.density_gmm_frame, text="Perform", command=self.perform_density_gmm_clustering, width=22, bg = "#dec231")
        self.density_gmm_clustering_button.grid(row=0, column=2, sticky=tk.W)


        self.intensity_gmm_label = tk.Label(self.density_gmm_frame, text="Intensity-Number of Clusters:", width=22, bg = "#dec231")
        self.intensity_gmm_label.grid(row=1, column=0, sticky=tk.W)

        self.intensity_gmm_components_entry = tk.Entry(self.density_gmm_frame, width=22)
        self.intensity_gmm_components_entry.grid(row=1, column=1, sticky=tk.W)

        self.intensity_gmm_clustering_button = tk.Button(self.density_gmm_frame, text="Perform", command=self.perform_intensity_gmm_clustering, width=22, bg = "#dec231")
        self.intensity_gmm_clustering_button.grid(row=1, column=2, sticky=tk.W)
       
        ########################
        # Field 6 : hdbscan clustering
        ########################

        self.hdbscan_button = tk.Button(master, text="Hdbscan Clustering",command=self.message_hdbscan_clustering, width=22, bg="lightblue")
        self.hdbscan_button.grid(row=8, column=0, sticky=tk.W)

        self.hdbscan_frame = tk.Frame(master, pady=5, bg = "#dec231")
        self.hdbscan_frame.grid(row=8, column=1, sticky=tk.W)

        # The minimum number of samples in a group for that group to be considered a cluster; 
        # groupings smaller than this size will be left as noise.
        self.minClusterSize_label = tk.Label(self.hdbscan_frame, text=" Minimum Cluster Size", width=22, bg = "#dec231")
        self.minClusterSize_label.grid(row=0, column=0, sticky=tk.W)

        self.minClusterSize_entry = tk.Entry(self.hdbscan_frame, width=22)
        self.minClusterSize_entry.grid(row=0, column=1, sticky=tk.W)

        self.minNumSamples_label = tk.Label(self.hdbscan_frame, text="Minimum Number of Samples", width=22, bg = "#dec231")
        self.minNumSamples_label.grid(row=1, column=0, sticky=tk.W)

        self.minNumSamples_entry = tk.Entry(self.hdbscan_frame, width=22)
        self.minNumSamples_entry.grid(row=1, column=1, sticky=tk.W)

        self.hdbscan_button = tk.Button(self.hdbscan_frame, text="Perform", command=self.perform_hdbscan_clustering_density, width=22, bg = "#dec231")
        self.hdbscan_button.grid(row=1, column=2, sticky=tk.W)

        ########################
        # Clustering by drawing
        ########################

        self.drawing_cluster_button = tk.Button(master, text="Drawing cluster",command=self.message_drawing_cluster, width=22, bg="lightblue")
        self.drawing_cluster_button.grid(row=9, column=0, sticky=tk.W)

        self.drawing_frame = tk.Frame(master, pady=10, bg = "#dec231")
        self.drawing_frame.grid(row=9, column=1, sticky=tk.W)

        self.drawing_cluster_label = tk.Label(self.drawing_frame, text="Drawing the borders", width=22, bg = "#dec231")
        self.drawing_cluster_label.grid(row=0, column=0, sticky=tk.W)

        self.drawing_cluster_start = tk.Button(self.drawing_frame, text="Start", command=self.segmentation, width=18, bg = "#dec231")
        self.drawing_cluster_start.grid(row=0, column=1, sticky=tk.W)

        self.drawing_cluster_get_cluster = tk.Button(self.drawing_frame, text="Get clusters", command=self.get_clusters, width=22, bg = "#dec231")
        self.drawing_cluster_get_cluster.grid(row=0, column=2, sticky=tk.W)

        ########################
        # Field 7: clusters
        ########################

        self.main_frame = tk.Frame(master)
        self.main_frame.grid(row=12, column=1, padx=10, pady=10)

        self.checkbox_show_cluster_var = tk.BooleanVar() # check if you want to show each cluster seperately    
        self.checkbox_show_cluster = tk.Checkbutton(self.main_frame,text= 'Showing the choosen cluster seperately', variable=self.checkbox_show_cluster_var)
        self.checkbox_show_cluster.grid(row=0, column=1, pady=10)

        # Create a frame for adding , removing, joining, visualizing the clusters as result
        self.edit_frame = tk.Frame(self.main_frame, pady=5, bg = "#dec231")
        self.edit_frame.grid(row=1, column=2, padx=10, pady=10)

        self.result_volume_section = tk.Button(self.edit_frame, text = "Result Section", command = self.message_result_section, width =22, bg="lightblue", pady=5)
        self.result_volume_section.grid(row=0, column=0)

        self.start_results = tk.Button(self.edit_frame, text="Empty result", command = self.empty_result_list, width=22, bg = "#dec231")
        self.start_results.grid(row=1, column=0)

        self.merge_clusters_tk = tk.Button(self.edit_frame, text="Merge clusters", command = self.merge_clusters, width=22, bg = "#dec231")
        self.merge_clusters_tk.grid(row=2, column=0)

        self.remove_clusters_tk = tk.Button(self.edit_frame, text="Remove clusters", command = self.remove_clusters, width=22, bg = "#dec231")
        self.remove_clusters_tk.grid(row=3, column=0)

        self.continue_clustering = tk.Button(self.edit_frame, text="Add clusters",command = self.add_to_results, width=22, bg = "#dec231")
        self.continue_clustering.grid(row=4, column=0)

        self.update_clusters_tk = tk.Button(self.edit_frame, text="Update clusters", command= self.update_clusters, width=22, bg = "#dec231")
        self.update_clusters_tk.grid(row=5, column=0)

        self.results_as_point_cloud = tk.Button(self.edit_frame, text="Convert all to point cloud", command= self.get_as_point_cloud , width=22, bg = "#dec231")
        self.results_as_point_cloud.grid(row=6, column=0)

        # Create a frame for the buttons and checkboxes
        self.button_frame = tk.Frame(self.main_frame)
        self.button_frame.grid(row=1, column=1, padx=10, pady=10, sticky=tk.W)

        # Fixed contrast colors
        self.colors_clusters = [
            "#F61a06", "#F6db06", "#58f606", "#0679f6", "#B106f6",
            "#1a9641", "#e377c2", "#91bfdb", "#fee08b", "#28708f", 
            "#03ffae", "#B59cd5", "#a55194", "#0eeef4", "#E7ff08",
            "#ff7f00", "#4d4d4d", "#Aa3a13", "#98df8a", "#377916",
            "#728307", "#4bbbec", "#F50bf8", "#F0b010", "#E6156b",
            "#4575b4", "#313695", "#F6a3df", "#682805", "#D9e285",
            "#A6fd7c", "#052966", "#Fff303", "#Adf2eb", "#D5759e",
            "#Ff5504", "#B2b0ae", "#F0fad2", "#39bf76", "#3f075e"]

        self.checked_clusters = []  # List to store checked cluster IDs
        self.result_clusters = []  # List to store the result of clusters
        self.checkbox_vars = [] # Create checkbox variables  

        # Create 40 buttons representing clusters with fixed colors
        for cluster_id, color in enumerate(self.colors_clusters, start=1):
            button = tk.Button(self.button_frame, text=f"Cluster {cluster_id}", command=lambda cluster_id=cluster_id: self.choose_cluster(cluster_id), bg=color, width=8, height=2)
            button.grid(row=(cluster_id -1 ) % 10, column=(cluster_id - 1) // 10 * 2, padx=5, pady=5, sticky=tk.W)

            # Create a checkbox for volume beside each cluster
            checkbox_var = tk.IntVar()
            self.checkbox_vars.append(checkbox_var)
            checkbox = tk.Checkbutton(self.button_frame, variable=checkbox_var, command=lambda cluster_id=cluster_id, checkbox_var=checkbox_var: self.checkbox_checked(cluster_id, checkbox_var))
            checkbox.grid(row=(cluster_id -1) % 10, column=(cluster_id - 1) // 10 * 2 + 1, padx=5, pady=5, sticky=tk.W)
     
     
        self.total_volume_frame = tk.Frame(self.edit_frame, bg='#7FC3D8')
        self.total_volume_frame.grid(row=9, column=0, padx=10, pady=10, sticky=tk.W)
        self.calculation_volume_section = tk.Button(self.total_volume_frame, text ="Volume Result", command= self.message_result_volume, width=22, bg="lightblue")
        self.calculation_volume_section.grid(row=0, column=0, padx=10, pady=5) 
        self.reset_volume = tk.Button(self.total_volume_frame,text="Reset Volume", command= self.reset_volume_function, width=22)
        self.reset_volume.grid(row=1, column=0, padx=10, pady=5)  
        self.total_volume_add = tk.Button(self.total_volume_frame,text="Add to Volume", command= self.append_volume, width=22)
        self.total_volume_add.grid(row=2, column=0, padx=10, pady=5) 
        self.total_volume_button = tk.Button(self.total_volume_frame, text="Total Volume", command= self.calculate_total_volume, width=22)
        self.total_volume_button.grid(row=3, column=0, padx=10, pady=5) 
        self.show_total_volume = tk.Text(self.total_volume_frame, width=20, height = 3)  
        self.show_total_volume.grid(row=4, column=0, padx=10, pady=5)  

        ########################
        # Field 8: Interpolation
        ########################

        self.interpolation_frame = tk.Frame(master, bg = "lightgray")
        self.interpolation_frame.grid(row=12, column=0, pady=30)

        self.interpolation_var = tk.BooleanVar() # check if you want to show each cluster seperately    
        self.interpolation = tk.Checkbutton(self.interpolation_frame,text= 'Showing the interpolated points', variable=self.interpolation_var, bg="lightgray")
        self.interpolation.grid(row=1, column=0, sticky=tk.W)

        self.interpolation_button = tk.Button(self.interpolation_frame, text=" Interpolation",command=self.message_interpolation, width=22, bg="lightblue")
        self.interpolation_button.grid(row=0, column=0, sticky=tk.W)

        self.number_interpolated_point_label = tk.Label(self.interpolation_frame, text=" The Number of interpolatied points", bg="lightgray")
        self.number_interpolated_point_label.grid(row=2, column=0, sticky=tk.W)

        self.number_interpolated_point_entry = tk.Entry(self.interpolation_frame, width=5)
        self.number_interpolated_point_entry.grid(row=2, column=1)

        self.interpolation_button = tk.Button(self.interpolation_frame, text="Start",command= self.interpolate_data_btw_planes, width = 15)
        self.interpolation_button.grid(row=3, column=0)

        ########################
        # Field 8: Voxelization
        ########################

        self.voxelization_button = tk.Button(self.interpolation_frame, text="Voxelization",command=self.message_voxelization, width=22, bg="lightblue")
        self.voxelization_button.grid(row=4, column=0, sticky=tk.W)

        self.voxelization_size_label = tk.Label(self.interpolation_frame, text=" The voxel size", width=22, bg="lightgray")
        self.voxelization_size_label.grid(row=5, column=0, sticky=tk.W)

        self.voxel_size_entry = tk.Entry(self.interpolation_frame, width=5)
        self.voxel_size_entry.grid(row=5, column=1)

        self.checkbox_var = tk.BooleanVar()
        self.checkbox = tk.Checkbutton(self.interpolation_frame,text= 'Weighted Voxelization', variable = self.checkbox_var, bg="lightgray")
        self.checkbox.grid(row=7, column=0, sticky=tk.W)

        self.Lower_Threshold_label = tk.Label(self.interpolation_frame, text=" Lower Threshold", width=22, bg="lightgray")
        self.Lower_Threshold_label.grid(row=9, column=0, sticky=tk.W)

        self.Lower_Threshold_entry = tk.Entry(self.interpolation_frame, width=5)
        self.Lower_Threshold_entry.grid(row=9, column=1)

        self.Upper_Threshold_label = tk.Label(self.interpolation_frame, text=" Upper Threshold", width=22, bg="lightgray")
        self.Upper_Threshold_label.grid(row=11, column=0, sticky=tk.W)

        self.Upper_Threshold_entry = tk.Entry(self.interpolation_frame, width=5)
        self.Upper_Threshold_entry.grid(row=11, column=1)

        self.Voxelization_button = tk.Button(self.interpolation_frame, text="Voxelize", command=self.voxelize, width=15)
        self.Voxelization_button.grid(row=13, column=0) 

        self.Volume_button = tk.Button(self.interpolation_frame, text="Caculate Volume", command=self.calculate_volume_voxelization, width=15)
        self.Volume_button.grid(row=14, column=0) 

        ####################
        # Save point cloud
        ####################
        self.save_point_cloud = tk.Button(self.interpolation_frame, text="Saving the Current Point Cloud", command=self.saving_point_cloud, width = 25, height= 4, padx=40)
        self.save_point_cloud.grid(row=15, column=0, columnspan=2)

        #Save data with ping and beam to be able load them later 
        self.save_data_frame = tk.Button(self.interpolation_frame, text= "Save current point cloud as dataframe", command= self.saving_df, width= 25, height= 4, padx=40)
        self.save_data_frame.grid (row= 19, column=0, columnspan = 2)

        # Button to Access Previous Point Cloud
        #self.count = 0
        self.access_previous_button = tk.Button(self.interpolation_frame, text="Access Previous Point Cloud", command=self.access_previous_point_cloud, width = 25, height= 4, padx=40)
        self.access_previous_button.grid(row=20, column=0, columnspan=2)

        # Text Box for Information
        self.text_frame = tk.Frame(master)
        self.text_frame.grid(row=0, column= 2,rowspan =30, columnspan=2)
        self.text_box = tk.Text( self.text_frame,height = 50)
        self.text_box.grid(row=0, column= 0,rowspan =30,columnspan=2, pady = 5, padx = 10)
        self.text_delete = tk.Button( self.text_frame, text="Erase!", command=self.delete_text, width=22)
        self.text_delete.grid(row=32, column=0, sticky=tk.W, padx = 10)

    def clear_entry(self):
        # Delete all default values
        self.lower_threshold_entry.delete(0, tk.END)
        self.upper_threshold_entry.delete(0, tk.END)
        self.remove_bottom_entry.delete(0, tk.END)
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
        self.number_interpolated_point_entry.delete(0, tk.END)    
        self.voxel_size_entry.delete(0, tk.END)

    def set_default_sea_weed_values(self):

        # Clear all the default values to set the new one
        self.clear_entry()

        # Default threshold for filtering
        self.default_low_threshold = -58
        self.lower_threshold_entry.insert(0, self.default_low_threshold)
        self.default_high_threshold = -30
        self.upper_threshold_entry.insert(0, self.default_high_threshold)

        # Default percentage to remove data from the bottom
        self.default_percent = 0.15
        self.remove_bottom_entry.insert(0, self.default_percent)

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

        self.default_number_interpolation = 10
        self.number_interpolated_point_entry.insert(0, self.default_number_interpolation)

        self.voxel_size_default = 0.15
        self.voxel_size_entry.insert(0, self.voxel_size_default)

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

        # Default percentage to remove data from the bottom
        self.default_percent = 0.15
        self.remove_bottom_entry.insert(0, self.default_percent)

        # Default parameters for outlier removal
        self.default_nb_neighbors = 200
        self.nb_neighbors_entry.insert(0, self.default_nb_neighbors)
        self.default_std_ratio = 2.0
        self.std_ratio_entry.insert(0, self.default_std_ratio)
        self.default_nb_points = 200
        self.nb_points_entry.insert(0, self.default_nb_points) 
        self.default_radius = 0.5
        self.radius_entry.insert(0,self.default_radius)

        # Default number of clusters
        self.default_num_clusters = 2
        self.density_clusters_entry.insert(0, self.default_num_clusters)
        self.intensity_clusters_entry.insert(0, self.default_num_clusters)
        self.density_gmm_components_entry.insert(0, self.default_num_clusters)
        self.intensity_gmm_components_entry.insert(0, self.default_num_clusters)
        self.default_minClusterSize = 100
        self.minClusterSize_entry.insert(0, self.default_minClusterSize)
        self.default_minNumSamples = 100
        self.minNumSamples_entry.insert(0, self.default_minNumSamples)
 

        self.default_Upper_Threshold = 20
        self.Upper_Threshold_entry.insert(0, self.default_Upper_Threshold)
        self.default_Lower_Threshold = 1
        self.Lower_Threshold_entry.insert(0, self.default_Lower_Threshold)

        self.default_number_interpolation = 10
        self.number_interpolated_point_entry.insert(0, self.default_number_interpolation)

        self.voxel_size_default = 0.15
        self.voxel_size_entry.insert(0, self.voxel_size_default)

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
        self.reset_volume = True
        self.volume_values= [] 
        self.calculate_total_volume()

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

    def saving_df(self):
        if self.point_cloud is None:
            print("No point cloud to save.")
            return
        
        # Ask the user for the file name and directory
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])

        if file_path:
            self.df_filtered = BeamImager.pointcloud_to_df (self.point_cloud)
            self.merged_df = BeamImager.merging_ping (self.df, self.df_filtered)
            self.merged_df.iloc[:, :-3].to_csv(file_path, index =False, header=False)

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

    def message_drawing_cluster(self):
        messagebox.showinfo("Drawing clusters", "If you couldn't find the right  clustering method for making the boundary, you can draw the boundary yourself. ")  

    def message_result_section(self):
        messagebox.showinfo("Result section","In this section you can edit the end result by adding , removing and merging clusters to the result and update the result to check what is in it")      

    def message_result_volume(self):
        messagebox.showinfo("Calculate result", "In this section you can add the volume of clusters to the total volume if you need to have the sum of volumes for different clusters")

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
        percent = float(self.remove_bottom_entry.get()) if self.remove_bottom_entry.get() else self.default_percent
        
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

    def empty_result_list (self):
        self.result_clusters = []
        
    def merge_clusters (self):
    
        # Create a new list to store the merged clusters
        merged_clusters = []

        # Get which clusters in the result should merge together
        indices = self.checked_clusters

        # Iterate through each index in the indices list
        for i in range(len(self.result_clusters)):
            # Check if the index is not in the list of indices to merge
            if i+1  not in indices:
                # If not in the list, append the cluster to the new list
                merged_clusters.append(self.result_clusters[i])
        # Merge the clusters specified by the indices list
        merged_cluster = o3d.geometry.PointCloud()
        for index in indices:
            merged_cluster += self.result_clusters[index-1]
        
        # Append the merged cluster to the new list
        merged_clusters.append(merged_cluster)
        self.result_clusters = merged_clusters

    def remove_clusters (self):
        # Create a new list to store the remaining clusters
        remaining_clusters = []
        # Get which clusters in the result should merge together
        indices = self.checked_clusters
        
        # Iterate through each index in the indices list
        for index in range(len(self.result_clusters)):
            # Check if the index is not in the list of indices to remove
            if index + 1 not in indices:
                # If not in the list, append the cluster to the new list
                remaining_clusters.append(self.result_clusters[index])
                
        self.result_clusters = remaining_clusters        

    def add_to_results (self):
        self.result_clusters.extend(self.cluster_clouds)

    def update_clusters (self):
        num_clusters = len(self.result_clusters)
        print(num_clusters)
        sorted_point_clouds = sorted(self.result_clusters, key=BeamImager.calculate_point_cloud_size, reverse=True)
        self.result_clusters = sorted_point_clouds
        self.checked_clusters = []
        plt.close('all')

        fig_clusters = plt.figure()
        ax = fig_clusters.add_subplot(111, projection='3d')        

        for i, pc in enumerate(sorted_point_clouds):
            points = np.asarray(pc.points)    
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], label=f'Cluster {i + 1}', c=self.colors_clusters[i])

        # Display sizes of sorted clusters
        sizes = [BeamImager.calculate_point_cloud_size(pc) for pc in sorted_point_clouds]
        for i, size in enumerate(sizes):
            txt_cluster = f'Cluster {i + 1} (Size: {size} points)'
            print(txt_cluster)
            
        ax.legend()
        plt.show()
        
        self.reset_checkboxes()  

    def reset_checkboxes(self):
        # Iterate through each checkbox variable and set it to unchecked
        for checkbox_var in self.checkbox_vars:
            checkbox_var.set(0) 

    def get_as_point_cloud (self):
        self.point_cloud = o3d.geometry.PointCloud()
        for point_cloud in self.result_clusters:
            self.point_cloud += point_cloud    
        self.text = f"{self.point_cloud}"
        self.print_info(self.text)

    def segmentation (self):
        if self.point_cloud is not None:

            # Create Tkinter window and run the app
            segmentation_window = tk.Toplevel(self.master)
            self.segmentation_app = CurveDrawingApp(segmentation_window, self.point_cloud)
            
    def get_clusters(self):
        if self.segmentation_app is not None:
                self.clustered_point_clouds = self.segmentation_app.get_clustred_point_clouds() 
        self.cluster_clouds = BeamImager.visualize_clusters(self.clustered_point_clouds, self.colors_clusters)

    def choose_cluster(self, id):
        point_cloud = None  # Initialize point_cloud to None
        self.cluster_number = f"Cluster {id}"
        self.text = f"Cluster {id} was choosen"
        self.print_info(self.text)
    
        if self.result_clusters is not None and 0 < id <= len(self.result_clusters):
            clusters_original = BeamImager.visualize_selected_cluster(self.result_clusters, id - 1)
            point_cloud = clusters_original[id - 1]  
            self.result_clusters = clusters_original

        if point_cloud is not None:
            # Visualize the selected cluster
            self.point_cloud = point_cloud
            if self.checkbox_show_cluster_var.get() is True:
                o3d.visualization.draw_geometries([self.point_cloud])

        else:
            print("Invalid cluster ID or result_clusters is None.")   
            print(self.point_cloud)

    def checkbox_checked(self, cluster_id, checkbox_var):
        if checkbox_var.get() == 1:
            #print(f"Checkbox for Cluster {cluster_id} is checked")
            # Add the checked cluster ID to the list
            if cluster_id not in self.checked_clusters:
                self.checked_clusters.append(cluster_id)
        else:
            print(f"Checkbox for Cluster {cluster_id} is unchecked")
            # Remove the unchecked cluster ID from the list
            if cluster_id in self.checked_clusters:
                self.checked_clusters.remove(cluster_id)
        print("Updating Checked Clusters:", self.checked_clusters)       
  
    def interpolate_data_btw_planes (self):
        
        self.df_filtered = BeamImager.pointcloud_to_df (self.point_cloud)
        self.merged_df = BeamImager.merging_ping (self.df, self.df_filtered)
        self.plane_list = BeamImager.get_plane_list (self.merged_df)
        number_point = int(self.number_interpolated_point_entry.get())
        self.interpolated_points_list, self.Average_distance_planes = BeamImager.interpolate_volumes(self.plane_list, interpolation_steps= number_point)
        self.text = f"The mean of distances between planes: {self.Average_distance_planes}"
        self.print_info(self.text)
        self.merged_cloud = BeamImager.merge_volumes(self.interpolated_points_list)
        if self.interpolation_var.get() is True:
            o3d.visualization.draw_geometries([self.merged_cloud, self.point_cloud])
        self.point_cloud = self.merged_cloud
        ## add the point cloud to the point_cloud_history_list
        self.point_cloud_history.append(self.point_cloud)

    def voxelize(self):
        self.voxel_size = float(self.voxel_size_entry.get()) if self.voxel_size_entry.get() else self.default_voxel_size
        self.voxel_grid, self.unique_voxels, self.voxel_counts = BeamImager.count_point_per_voxel(self.point_cloud, self.voxel_size)
        #o3d.visualization.draw_geometries([self.voxel_grid, self.point_cloud])
        #self.print_info(f"Average number of count per voxel is:{self.voxel_counts.mean()}")
        self.print_info(f"The number of count that 10% of the voxels are equal to or below:{int(np.percentile(self.voxel_counts, 10))}")
        self.print_info(f"The number of count that 25% of the voxels are equal to or below:{int(np.percentile(self.voxel_counts, 25))}")
        self.print_info(f"The median of number of count:{int(np.percentile(self.voxel_counts, 50))}")

    def calculate_volume_voxelization(self):
        self.threshold_count_up= int(self.Upper_Threshold_entry.get()) if self.Upper_Threshold_entry.get() else self.default_Upper_Threshold
        self.threshold_count_down = int(self.Lower_Threshold_entry.get()) if self.Lower_Threshold_entry.get() else self.default_Lower_Threshold

        if self.checkbox_var.get() is False:
            # Calculate the volume of the voxelized point cloud
            print(len(self.voxel_grid.get_voxels()))
            self.voxel_volume =  len(self.voxel_grid.get_voxels()) * self.voxel_size**3
            self.print_info(f"Volume (voxelization) of {self.cluster_number}: {self.voxel_volume:.6f} m3")

        else:  
            self.voxel_volume = BeamImager.custom_voxelization(self.voxel_grid,self.unique_voxels,self.voxel_counts, self.voxel_size, self.threshold_count_up , self.threshold_count_down)
            self.print_info(f"Volume (customized voxelization) of {self.cluster_number}: {self.voxel_volume:.6f} m3")
        
    def calculate_total_volume(self):
        self.total_volume = sum(self.volume_values)  

        # Display the result in the Text widget
        self.show_total_volume.config(state=tk.NORMAL)  # Enable editing
        self.show_total_volume.delete(1.0, tk.END)  # Clear existing text
        self.show_total_volume.insert(tk.END, f"Sum Volume:\n{self.total_volume:.6f} m3")
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