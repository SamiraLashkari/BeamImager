import pandas as pd
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import os
import hdbscan
import copy

def getting_values (df_data):

    # This function extract X, Y, Z, and Intensity columns of the loaded data
    x_data = df_data.iloc[:, 0]
    y_data = df_data.iloc[:, 1]
    z_data = df_data.iloc[:, 2]
    intensity_data = df_data.iloc[:, 3]
    ping_number = df_data.iloc[:, 4]
    beam_angle = df_data.iloc[:, 5]

    return (x_data, y_data, z_data, intensity_data, ping_number, beam_angle)

def normalize_intensity (intensity):

    # This function print the max and min values for intensity and normalize the values of intensity between 0 to 1
    
    Max_intensity = intensity.max()
    Min_intensity = intensity.min()
    txt = f"max is {Max_intensity} and min is {Min_intensity}"
    print(f"max is {Max_intensity} and min is {Min_intensity}")
    normalized_intensity = (intensity - Min_intensity) / (Max_intensity - Min_intensity)
    
    return (normalized_intensity, Max_intensity, Min_intensity, txt)

def histogram (value, value_name , num_bins):

    # Create the histogram
    n, bins, patches = plt.hist(value, bins = num_bins, color ='blue', alpha=0.7, edgecolor='black', log=True)
    plt.xlabel(f'{value_name}')
    plt.ylabel('Log Frequency')
    plt.title(f'Histogram {value_name} with Log Frequency')
    plt.grid(True)
    plt.show()

def filter_lowest_percent(group, thresholds):

    filtered_data = group[group['z'] > thresholds]
    removed_data = group[group['z'] <= thresholds]

    return (filtered_data, removed_data)   

def filter_noise(df, percent): 
    thresholds = df.groupby('beam')['z'].quantile(percent)

    # Apply the filtering function to each group
    result = df.groupby('beam', group_keys=False).apply(lambda group: filter_lowest_percent(group, thresholds.loc[group['beam'].iloc[0]]))

    # Unpack the result into filtered_df and removed_df
    filtered_df, removed_df = zip(*result)

    # Concatenate the filtered and removed dataframes
    filtered_df = pd.concat(filtered_df, ignore_index=True)
    removed_df = pd.concat(removed_df, ignore_index=True)

    return(filtered_df, removed_df)

def display_inlier_outlier_bottom(df_filtered_beam_z, df_removed_beam_z):
    # Create point clouds
    filtered_points = np.column_stack((df_filtered_beam_z['x'], df_filtered_beam_z['y'], df_filtered_beam_z['z']))
    rest_of_points = np.column_stack((df_removed_beam_z['x'], df_removed_beam_z['y'], df_removed_beam_z['z']))

    filtered_colors = np.tile(np.array([0, 1, 0]), (len(filtered_points), 1))  # Green color for filtered points
    rest_of_colors = np.tile(np.array([1, 0, 0]), (len(rest_of_points), 1))  # Red color for removed points

    # Combine points and colors
    filtered_point_cloud = o3d.geometry.PointCloud()
    filtered_point_cloud.points = o3d.utility.Vector3dVector(filtered_points)
    filtered_point_cloud.colors = o3d.utility.Vector3dVector(filtered_colors)

    rest_of_point_cloud = o3d.geometry.PointCloud()
    rest_of_point_cloud.points = o3d.utility.Vector3dVector(rest_of_points)
    rest_of_point_cloud.colors = o3d.utility.Vector3dVector(rest_of_colors)

    # Visualize using open3d
    o3d.visualization.draw_geometries([filtered_point_cloud, rest_of_point_cloud])

def create_pointCloud_object (points_data, colors_data):

    #This function create an Open3D point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_data)
    point_cloud.colors = o3d.utility.Vector3dVector(colors_data)  # Extract RGB from colormap
    return (point_cloud)

def save_pointCloud (point_cloud, name):

    #This function save the point cloud as a PCD file
    output_pcd_file = name
    o3d.io.write_point_cloud(output_pcd_file, point_cloud)
    print(f"Point cloud saved")

def color_visualization_pointCloud( intensity):

    # This function can choose a different colormap based on the different intensity (not black and white)
    
    colormap = cm.get_cmap('viridis')  # You can choose a different colormap
    colors = colormap(intensity) # Otherwise it will be black and white

    return(colors)

def color_to_intensity(point_cloud):
    colors = np.asarray(point_cloud.colors)
    intensities = np.mean(colors, axis=1)
    
    return intensities

def visualize_selected_cluster(clusters, selected_index, other_intensity=0.7):
    # Check if the specified index is valid
    #if selected_index < 0 or selected_index >= len(clusters):
    #    raise ValueError("Invalid cluster index")
    if clusters is not None:
        clusters_original = copy.deepcopy(clusters)

    # Create a combined point cloud of all clusters except the selected one
    clusters_except_selected = [clusters[i] for i in range(len(clusters)) if i != selected_index]
    combined_point_cloud = o3d.geometry.PointCloud()
    combined_point_cloud.points = o3d.utility.Vector3dVector(np.vstack([np.asarray(cluster.points) for cluster in clusters_except_selected]))
    combined_point_cloud.colors = o3d.utility.Vector3dVector(np.vstack([np.asarray(cluster.colors) for cluster in clusters_except_selected]))

    # Get the selected cluster
    selected_cluster = clusters[selected_index]

    # Visualize the selected cluster with original intensity and others with gray color
    selected_colors = np.asarray(selected_cluster.colors)
    selected_colors[:, :] = [1.0, 0.0, 0.0]  # Set to white for better visibility
    selected_cluster.colors = o3d.utility.Vector3dVector(selected_colors)

    o3d.visualization.draw_geometries([combined_point_cloud, selected_cluster])
    return(clusters_original)

def pointcloud_to_df (point_cloud): 
    point_cloud_data = np.asarray(point_cloud.points)  # Stack coordinates and colors horizontally
    columns = ['x', 'y', 'z']
    df_filtered = pd.DataFrame(point_cloud_data, columns=columns)

    return(df_filtered)

def merging_ping (df, df_filtered): 
    df_no_duplicates = df.drop_duplicates(subset=['x', 'y', 'z'])
    merged_df = pd.merge(df_filtered, df_no_duplicates, on=['x', 'y', 'z'], how='left')

    return(merged_df)

def get_plane_list (merged_df):
    
    group_column = 'ping_number'  # Specify the column for grouping the data
    columns_to_keep = ['x', 'y', 'z', 'R', 'G', 'B'] # Specify the three columns you want to keep in the new DataFrames
    grouped_data = merged_df.groupby(group_column) # Group the DataFrame by the specified column
    plane_list = [] # Initialize a list to store the split DataFrames

    # Iterate over groups and create new DataFrames with three specified columns for each plane
    for group_value, group_df in grouped_data:
        new_df = group_df[columns_to_keep].copy()
        plane_list.append(new_df)

    return(plane_list)    

def interpolate_volumes(planes, interpolation_steps=10):
    # Check if there are at least two volumes
    if len(planes) < 2:
        raise ValueError("At least two volumes are required for interpolation.")
    
    # Interpolation range along the straight line connecting volumes
    interpolation_range = np.linspace(0, 1, interpolation_steps)

    # Interpolate between volumes
    interpolated_points_list = []

    # a list of distances between planes
    all_distances = []

    for t in interpolation_range:
        interpolated_points = []
        for i in range(len(planes)-1):
            volume1_points = np.asarray(planes[i])
            volume2_points = np.asarray(planes[i+1])

            # Calculate the pairwise distances between all points on the two planes
            distances = np.linalg.norm(volume1_points[:, np.newaxis, :] - volume2_points, axis=2)

            # Find the minimum distance
            min_distance = np.min(distances)
            all_distances.append(min_distance)

            # Interpolate between points on the two planes
            interpolated_points.extend((1 - t) * p1 + t * p2 for p1, p2 in zip(volume1_points, volume2_points))

        interpolated_points_list.append(interpolated_points)

    Average_distance_planes = sum(all_distances) / len(all_distances)
    print(f"The mean of distances between planes: {Average_distance_planes}")

    return interpolated_points_list, Average_distance_planes

def merge_volumes(interpolated_points_list):
    # Combine the interpolated points from all volumes
    merged_points = np.concatenate(interpolated_points_list, axis=0)

    # Create an Open3D PointCloud for the merged points
    merged_cloud = o3d.geometry.PointCloud()
    merged_cloud.points = o3d.utility.Vector3dVector(merged_points[:, :3])
    #merged_cloud.colors = o3d.utility.Vector3dVector(merged_points[:, 3:6]) 

    return merged_cloud

def custom_callback(vis,threshold_low, threshold_high):

    # We define a callback function to save the point cloud and close the window
    
    if not vis.poll_events():
        vis.capture_screen_image(f'threshold_low is "{threshold_low} "and threshold_high is" {threshold_high} ".png')  # Save the visualization as an image
        vis.destroy_window()  # Close the visualization window

def visualization_pointCloud(point_cloud):

    # Visualize the point cloud and apply the custom callback
    o3d.visualization.draw_geometries_with_animation_callback([point_cloud], custom_callback)

def mask_threshold(df, threshold_low, threshold_high): 

    # Create a mask for points that meet the threshold condition
    mask = (df['intensity'] >= threshold_low) & (df['intensity']  <= threshold_high)  # Adjust the condition based on your requirement
    # Apply the mask to filter the DataFrame
    
    return (mask)

def keep_masked_pointClouds(mask, df, colors):

    # This function use the mask boolean array to filter the data

    filtered_points = np.array(df.iloc[:, :3])[mask, :]  # Select rows where mask is True
    filtered_colors = colors[mask, :]

    return(filtered_points, filtered_colors)

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

def outlier_removal_downsampled(pointCloud, voxel_size= None, every_k_points= None, nb_neighbors= 10, std_ratio=2.0):
    
    if voxel_size !=  None: 
        pointCloud = pointCloud.voxel_down_sample(voxel_size= voxel_size)

    if every_k_points !=  None: 
        pointCloud = pointCloud.uniform_down_sample(every_k_points = every_k_points)  

    pointCloud_filtered, ind = pointCloud.remove_statistical_outlier(nb_neighbors=nb_neighbors,std_ratio=std_ratio)
    return(pointCloud_filtered, ind)

def all_clusters_together(cluster_labels,points):
    
    # Convert cluster labels to colors
    unique_labels = np.unique(cluster_labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    # Create a color map for the clusters
    cluster_color_map = {}
    for label, color in zip(unique_labels, colors):
        cluster_color_map[label] = color

    # Create an array of cluster colors (three color values per point)
    cluster_colors = np.array([cluster_color_map[label][:3] for label in cluster_labels], dtype=np.float64)

    # Create an Open3D PointCloud object with colors
    clustered_point_cloud = o3d.geometry.PointCloud()
    clustered_point_cloud.points = o3d.utility.Vector3dVector(points)
    clustered_point_cloud.colors = o3d.utility.Vector3dVector(cluster_colors)

    # Visualize the clustered point cloud using Open3D
    o3d.visualization.draw_geometries([clustered_point_cloud])   
             
def clustering_k_mean_colors( point_cloud, number_cluster):

    # Extract the points from the point cloud and convert them to a NumPy array
    points = np.asarray(point_cloud.points)

    # Extract the intensity channels (R, G, B) from the point cloud's colors
    colors = np.asarray(point_cloud.colors) * 255.0  # Normalize to [0, 255]

    # Combine the spatial coordinates (x, y, z) with the intensity channels (R, G, B)
    features = np.hstack((points, colors))

    # Create a K-Means model
    kmeans = KMeans(n_clusters=number_cluster, random_state=0)

    # Fit the model to your data
    kmeans.fit(features)

    # Get cluster labels for each point
    labels = kmeans.labels_

    #using the function wrote for showing all clusters together with different colors
    all_clusters_together(labels, points)

    # Create a list to store separate clusters
    clustered_point_clouds = [[] for _ in range(number_cluster)]

    for i, label in enumerate(labels):
        point = points[i]  # Extract spatial coordinates (x, y, z)
        intensity = colors[i]  # Intensity values (R, G, B)
        cluster_name = f"Cluster {label + 1}"  # Assign cluster name based on order
        clustered_point_clouds[label].append((point, intensity, cluster_name))
        
    return (clustered_point_clouds)  

def clustering_k_mean_points( point_cloud, number_cluster):

    # Extract the points from the point cloud and convert them to a NumPy array
    points = np.asarray(point_cloud.points)

    # Extract the intensity channels (R, G, B) from the point cloud's colors
    colors = np.asarray(point_cloud.colors) * 255.0  # Normalize to [0, 255]

    # Combine the spatial coordinates (x, y, z) with the intensity channels (R, G, B)
    features = points

    # Create a K-Means model
    kmeans = KMeans(n_clusters=number_cluster, random_state=0)

    # Fit the model to your data
    kmeans.fit(features)

    # Get cluster labels for each point
    labels = kmeans.labels_

    #using the function wrote for showing all clusters together with different colors
    all_clusters_together(labels, points)

    # Create a list to store separate clusters
    clustered_point_clouds = [[] for _ in range(number_cluster)]

    for i, label in enumerate(labels):
        point = points[i]  # Extract spatial coordinates (x, y, z)
        intensity = colors[i]  # Intensity values (R, G, B)
        cluster_name = f"Cluster {label + 1}"  # Assign cluster name based on order
        clustered_point_clouds[label].append((point, intensity, cluster_name))
    return (clustered_point_clouds)   

def gmm_clustering_with_intensity(point_cloud, num_clusters):
    # Convert the Open3D point cloud to a NumPy array
    points = np.asarray(point_cloud.points)

    # Extract the intensity channels (R, G, B) from the point cloud's colors
    colors = np.asarray(point_cloud.colors) * 255.0  # Normalize to [0, 255]

    # Combine spatial coordinates (X, Y, Z) and RGB colors into feature matrix
    features = np.hstack((points, colors))

    # Perform Gaussian Mixture Model (GMM) clustering
    gmm = GaussianMixture(n_components=num_clusters, random_state=0)

    # Get cluster labels for each point
    cluster_labels = gmm.fit_predict(features)

    #using the function wrote for showing all clusters together with different colors
    all_clusters_together(cluster_labels, points)

    # Create a list to store separate clusters
    clustered_point_clouds = [[] for _ in range(num_clusters)]

    for i, label in enumerate(cluster_labels):
        point = points[i]  # Extract spatial coordinates (x, y, z)
        intensity = colors[i]  # Intensity values (R, G, B)
        cluster_name = f"Cluster {label + 1}"  # Assign cluster name based on order
        clustered_point_clouds[label].append((point, intensity, cluster_name))
    return (clustered_point_clouds)

def gmm_clustering_with_points(point_cloud, num_clusters):
    # Convert the Open3D point cloud to a NumPy array
    points = np.asarray(point_cloud.points)

    # Extract the intensity channels (R, G, B) from the point cloud's colors
    colors = np.asarray(point_cloud.colors) * 255.0  # Normalize to [0, 255]

    # Combine spatial coordinates (X, Y, Z) and RGB colors into feature matrix
    features = points

    # Perform Gaussian Mixture Model (GMM) clustering
    gmm = GaussianMixture(n_components=num_clusters, random_state=0)

    # Get cluster labels for each point
    cluster_labels = gmm.fit_predict(features)

    #using the function wrote for showing all clusters together with different colors
    all_clusters_together(cluster_labels, points)

    # Create a list to store separate clusters
    clustered_point_clouds = [[] for _ in range(num_clusters)]

    for i, label in enumerate(cluster_labels):
        point = points[i]  # Extract spatial coordinates (x, y, z)
        intensity = colors[i]  # Intensity values (R, G, B)
        cluster_name = f"Cluster {label + 1}"  # Assign cluster name based on order
        clustered_point_clouds[label].append((point, intensity, cluster_name))
    return (clustered_point_clouds)

def hdbscan_clustering_for_point_clouds(point_cloud, min_cluster_size, min_samples):

    # Extract the points from the point cloud and convert them to a NumPy array
    points = np.asarray(point_cloud.points)

    # Extract the intensity channels (R, G, B) from the point cloud's colors
    colors = np.asarray(point_cloud.colors) * 255.0  # Normalize to [0, 255]

    # Perform HDBSCAN clustering based on spatial coordinates
    clusterer = hdbscan.HDBSCAN(min_samples=min_samples, min_cluster_size=min_cluster_size)
    clusterer.fit_predict(points)

    labels = clusterer.labels_

    #using the function wrote for showing all clusters together with different colors
    all_clusters_together(labels, points)

    num_labels = len(set(labels)) - (1 if -1 in labels else 0)

    # Create a list to store separate clusters
    clustered_point_clouds = [[] for _ in range(num_labels)]

    for i, label in enumerate(labels):
        point = points[i]  # Extract spatial coordinates (x, y, z)
        intensity = colors[i]  # Intensity values (R, G, B)
        cluster_name = f"Cluster {label + 1}"  # Assign cluster name based on order
        clustered_point_clouds[label].append((point, intensity, cluster_name))

    return (clustered_point_clouds)  

def calculate_point_cloud_size(point_cloud):
    return len(np.asarray(point_cloud.points))

def visualize_clusters(clustered_point_clouds, cluster_colors):
    plt.close('all')
    cluster_clouds = []

    for cluster_points in clustered_point_clouds:
        cluster_points, intensities, cluster_names = zip(*cluster_points)
        cluster_points = np.array(cluster_points)
        intensities = np.array(intensities) / 255.0

        cluster_cloud = o3d.geometry.PointCloud()
        cluster_cloud.points = o3d.utility.Vector3dVector(cluster_points)
        cluster_cloud.colors = o3d.utility.Vector3dVector(intensities)

        cluster_clouds.append(cluster_cloud)

    num_clusters = len(cluster_clouds)
    sorted_point_clouds = sorted(cluster_clouds, key=calculate_point_cloud_size, reverse=True)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #print(sorted_point_clouds)

    for i, pc in enumerate(sorted_point_clouds):
        points = np.asarray(pc.points)
        #print(points)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], label=f'Cluster {i + 1}', c=cluster_colors[i])

    # Display sizes of sorted clusters
    sizes = [calculate_point_cloud_size(pc) for pc in sorted_point_clouds]
    for i, size in enumerate(sizes):
        txt_cluster = f'Cluster {i + 1} (Size: {size} points)'
        print(txt_cluster)
        
    ax.legend()
    plt.show()

    return sorted_point_clouds

def keep_largest_n_percent (df, column_index, n = 2):
    # Calculate the number of rows to keep (10 percent of the total)
    num_rows_to_keep = int(len(df) * 0.01 * n)

    # Sort the DataFrame by the specified column in descending order
    sorted_df = df.sort_values(by=df.columns[column_index], ascending=False)

    # Keep the largest 10 percent of rows
    largest_n_percent_df = sorted_df.head(num_rows_to_keep)

    return (largest_n_percent_df)

def meshing(point_cloud, radius, max_nn, depth):
    # Compute normals for the point cloud
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))

    # Create a surface mesh using Poisson surface reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=depth)

    # Visualize the mesh
    o3d.visualization.draw_geometries([point_cloud, mesh])

    return(mesh,densities)

def count_point_per_voxel(point_cloud, voxel_size):

    # Use VoxelGrid.create_from_point_cloud without colors
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size = voxel_size)

    # Get voxel indices
    voxel_indices = np.floor(np.asarray(point_cloud.points) / voxel_size).astype(int)

    # Count points per voxel
    unique_voxels, voxel_counts = np.unique(voxel_indices, axis=0, return_counts=True)

    return( voxel_grid, unique_voxels, voxel_counts)    

def custom_voxelization(voxel_grid, unique_voxels,voxel_counts, voxel_size, threshold_count_up, threshold_count_down):
   
    # Filter voxels based on the threshold count
    dense_voxels = unique_voxels[voxel_counts > threshold_count_up]

    # Filter voxels based on the threshold count and calculate the volume
    df = pd.DataFrame(voxel_counts, columns = ["count"])

    # Create a new column based on conditions
    df['weight'] = 0  # Initialize with 0
    #print(len(df['weight']))
    df.loc[df['count'] > threshold_count_up, 'weight'] = 1
    df.loc[(df['count'] > threshold_count_down) & (df['count'] <= threshold_count_up), 'weight'] = (df['count'] - threshold_count_down) / (threshold_count_up - threshold_count_down)
    #print(df['weight'].values)
    column_sum = df['weight'].sum()
    #print(column_sum)
    volume = column_sum * voxel_size**3

    return  volume

def voxelize(point_cloud, voxel_size):

    downsampled_pcd = point_cloud.voxel_down_sample(voxel_size)

    # Get voxel coordinates
    voxel_coords = np.floor(np.asarray(downsampled_pcd.points) / voxel_size).astype(int)

    # Assign a constant gray color to each voxel
    gray_color = [0.5, 0.5, 0.5]
    voxel_colors = np.full((len(voxel_coords), 3), gray_color)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size, voxel_colors)
    o3d.visualization.draw_geometries([voxel_grid, point_cloud])
    # Calculate the volume of the voxelized point cloud
    voxel_volume = len(voxel_grid.get_voxels()) * voxel_size**3
    return voxel_volume 

def analysis_for_a_folder (data_folder): 
    
    # Initialize a dictionary to store row counts
    pointclouds = {}
    count = 0
    # Iterate over files in the folder
    for root, dirs, files in os.walk(data_folder):
        for file_name in files:
            
            # Construct the full path to the data file
            file_path = os.path.join(root, file_name)

            # Read the dataset into a DataFrame
            df_data = pd.read_csv(file_path)  # Replace with the appropriate data loading function

            ## Getting geometry and intensity of data
            x_data, y_data, z_data, intensity_data = getting_values(df_data)

            ## normalize the intensity to have them in a range of 0 to 1
            intensity_n, txt = normalize_intensity(intensity_data.values)

            ## Choosing a color map for visualization intensity 
            colors = color_visualization_pointCloud(intensity_n)

            ## Creating an open3D point cloud with geometrical and normalized intensity data
            point_cloud = create_pointCloud_object(np.array(df_data.iloc[:, :3]), colors[:,:3])

            ## Visualizing the pointcloud
            o3d.visualization.draw_geometries([point_cloud])

            ## Masking data based on the threshold
            mask = mask_threshold(intensity_data.values, -62, -38)

            ## Filtering the data and keep those within defined threshold
            filtered_points, filtered_colors = keep_masked_pointClouds(mask, df_data, colors[:, :3])

            #making point cloud with filtered points based on the threshold
            point_cloud_filtered_threshold = create_pointCloud_object(filtered_points, filtered_colors)

            ## Visualizing the filtered pointcloud
            o3d.visualization.draw_geometries([point_cloud_filtered_threshold])

            pointCloud_filtered_downsampled, ind = outlier_removal_downsampled(point_cloud_filtered_threshold, voxel_size= 0.001, every_k_points= None, nb_neighbors= 1000, std_ratio=0.1)
            display_inlier_outlier(point_cloud_filtered_threshold, ind)
            o3d.visualization.draw_geometries([pointCloud_filtered_downsampled])

            # Specify the index of the column to use for filtering
            column_index = 3

            # Keep the largest  by default 2 percent (but you can chanege n=2) of values in the specified column
            result_df = keep_largest_n_percent(df_data, column_index)

            x_data_l, y_data_l, z_data_l, intensity_data_l = getting_values (result_df)

            ## normalize the intensity to have them in a range of 0 to 1
            intensity_n_l, txt = normalize_intensity(intensity_data_l.values)

            ## Choosing a color map for visualization intensity
            colors_l = color_visualization_pointCloud(intensity_n_l)

            ## Creating an open3D point cloud with geometrical and normalized intensity data
            point_cloud_l = create_pointCloud_object(np.array(result_df.iloc[:, :3]), colors_l[:,:3])

            ## Visualizing the pointcloud
            o3d.visualization.draw_geometries([point_cloud_l])

            cl, ind = point_cloud_l.remove_radius_outlier(nb_points=1000, radius=0.5)
            display_inlier_outlier(point_cloud_l, ind)
            o3d.visualization.draw_geometries([cl])

            # Construct a variable name
            variable_name = f'point_cloud_{count + 1}'
            print(variable_name)
            count = count + 1

            # Save the variables in the dictionary
            pointclouds[variable_name] = (point_cloud, point_cloud_filtered_threshold, pointCloud_filtered_downsampled, point_cloud_l, cl ) 

    return (pointclouds)

def intersection_pointclouds (pointclouds):
    voxel_size = 0.0001 
    point_list = []

    for variable_name, num in pointclouds.items():
        common_voxels =  np.asarray(num[0].points)
        for i in range(len(num)-1):
            #o3d.visualization.draw_geometries([num[i+1]])
            # Downsample the point clouds to a common grid
            voxel_grid = num[i+1].voxel_down_sample(voxel_size)
            # Convert to NumPy arrays for further processing
            voxels = np.asarray(voxel_grid.points)
            common_voxels = set(map(tuple, common_voxels)) & set(map(tuple, voxels))

        original_pc = num[0]
        points_without_colors = common_voxels

        # Initialize lists to store the extracted points and colors
        extracted_points = []
        extracted_colors = []

        # Iterate through the original point cloud
        for point, color in zip(np.asarray(original_pc.points), np.asarray(original_pc.colors)):
            point_tuple = tuple(point)
            # Check if the point exists in the set of points without colors
            if point_tuple in points_without_colors:
                extracted_points.append(point)
                extracted_colors.append(color)

        # Create a new point cloud object with extracted points and colors
        new_pc = o3d.geometry.PointCloud()
        new_pc.points = o3d.utility.Vector3dVector(extracted_points)
        new_pc.colors = o3d.utility.Vector3dVector(extracted_colors)

        # Visualize the common region
        #o3d.visualization.draw_geometries([new_pc ])
        point_list.append(new_pc)

    return(point_list, pointclouds)   