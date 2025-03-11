"""
This module contains additional functions needed for data training and testing.

Functions:
- get_device: Checks if GPU is available for use. If not, defaults to CPU.

"""
import torch
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial import cKDTree
from scipy.ndimage import binary_fill_holes
from collections import defaultdict

def get_device():
    """
    This function checks if a GPU is available for use. If not, it defaults to CPU.
    """
    # Step 2: Check if a GPU is available
    if torch.cuda.is_available():
        # Step 3: Set the device to GPU
        device = torch.device("cuda")
        print("GPU is available. Using GPU.")
    else:
        # Step 3: Default to CPU
        device = torch.device("cpu")
        print("GPU is not available. Using CPU.")
            # Step 4: Return the device
    return device

def von_mises_stress(sigma_x, sigma_y, sigma_z, tau_xy):
    term1 = (sigma_x - sigma_y) ** 2
    term2 = (sigma_y - sigma_z) ** 2
    term3 = (sigma_z - sigma_x) ** 2
    term4 = 6 * tau_xy ** 2
    von_mises = torch.sqrt(0.5 * (term1 + term2 + term3 + term4))
    return von_mises



def find_principal_stresses_3d(sigma_x, sigma_y, sigma_z, tau_xy, tau_yz, tau_zx):
    """
    Compute the principal stresses for a 3D stress state.

    Args:
        sigma_x (torch.Tensor): Stress component in the x direction.
        sigma_y (torch.Tensor): Stress component in the y direction.
        sigma_z (torch.Tensor): Stress component in the z direction.
        tau_xy (torch.Tensor): Shear stress component in the xy plane.
        tau_yz (torch.Tensor): Shear stress component in the yz plane.
        tau_zx (torch.Tensor): Shear stress component in the zx plane.

    Returns:
        torch.Tensor: Principal stresses (sigma_1, sigma_2, sigma_3).
    """
    # Create the stress tensor
    stress_tensor = torch.stack([
        torch.stack([sigma_x, tau_xy, tau_zx], dim=-1),
        torch.stack([tau_xy, sigma_y, tau_yz], dim=-1),
        torch.stack([tau_zx, tau_yz, sigma_z], dim=-1)
    ], dim=-2)

    # Compute the eigenvalues (principal stresses) of the stress tensor
    principal_stresses, _ = torch.linalg.eigh(stress_tensor)

    # Sort the principal stresses in descending order
    principal_stresses, _ = torch.sort(principal_stresses, descending=True, dim=-1)

    return principal_stresses

def creat_image(mesh_nodes, x_min,x_max,y_min,y_max,pixels_per_mm_x,pixels_per_mm_y):
    node_coords = mesh_nodes[:, :2]
    node_values = mesh_nodes[:, 2]

    # Physical dimensions of the geometry

    # Compute grid resolution based on pixel density
    width = int((x_max - x_min) * pixels_per_mm_x)
    height = int((y_max - y_min) * pixels_per_mm_y)

    # Create pixel grid (center coordinates)
    dx = 1.0 / pixels_per_mm_x  # Pixel width in mm
    dy = 1.0 / pixels_per_mm_y  # Pixel height in mm
    x_coords = np.arange(x_min + dx / 2, x_max, dx)
    y_coords = np.arange(y_min + dy / 2, y_max, dy)

    # Build KDTree for fast spatial searches
    tree = cKDTree(node_coords)

    # Allocate image
    image = np.zeros((height, width))

    # Interpolate pixel values
    for row, py in enumerate(y_coords):
        for col, px in enumerate(x_coords):
            # Define pixel bounds
            x_left = px - dx / 2
            x_right = px + dx / 2
            y_bottom = py - dy / 2
            y_top = py + dy / 2

            # Find nodes inside the pixel bounds
            indices = np.where(
                (node_coords[:, 0] >= x_left) & (node_coords[:, 0] < x_right) &
                (node_coords[:, 1] >= y_bottom) & (node_coords[:, 1] < y_top)
            )[0]

            # Skip empty pixels
            if len(indices) == 0:
                continue

            # Compute distances to pixel center
            distances = np.sqrt((node_coords[indices, 0] - px) ** 2 + 
                                (node_coords[indices, 1] - py) ** 2)

            # Avoid division by zero for coincident points
            weights = 1 / (distances ** 2 + 1e-10)
            weights /= np.sum(weights)  # Normalize weights

            # Compute weighted pixel value
            image[row, col] = np.sum(weights * node_values[indices])
    geom_image = np.zeros((len(image), len(image[0])))
    geom_image[image > 0] = 1
    return image, geom_image

import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import binary_fill_holes

def creat_image2(mesh_nodes, x_min, x_max, y_min, y_max, pixels_per_mm_x, pixels_per_mm_y):
    node_coords = mesh_nodes[:, :2]
    node_values = mesh_nodes[:, 2]

    # Compute grid resolution based on pixel density
    width = int((x_max - x_min) * pixels_per_mm_x)
    height = int((y_max - y_min) * pixels_per_mm_y)

    # Create pixel grid (center coordinates)
    dx = 1.0 / pixels_per_mm_x  # Pixel width in mm
    dy = 1.0 / pixels_per_mm_y  # Pixel height in mm
    x_coords = np.arange(x_min + dx / 2, x_max, dx)
    y_coords = np.arange(y_min + dy / 2, y_max, dy)

    # Allocate image
    image = np.zeros((height, width))

    # Interpolate pixel values
    for row, py in enumerate(y_coords):
        for col, px in enumerate(x_coords):
            # Define pixel bounds
            x_left = px - dx / 2
            x_right = px + dx / 2
            y_bottom = py - dy / 2
            y_top = py + dy / 2

            # Find nodes inside the pixel bounds
            indices = np.where(
                (node_coords[:, 0] >= x_left) & (node_coords[:, 0] < x_right) &
                (node_coords[:, 1] >= y_bottom) & (node_coords[:, 1] < y_top)
            )[0]

            # Skip empty pixels
            if len(indices) == 0:
                continue

            # Compute distances to pixel center
            distances = np.sqrt((node_coords[indices, 0] - px) ** 2 + 
                                (node_coords[indices, 1] - py) ** 2)

            # Avoid division by zero for coincident points
            weights = 1 / (distances ** 2 + 1e-10)
            weights /= np.sum(weights)  # Normalize weights

            # Compute weighted pixel value
            image[row, col] = np.sum(weights * node_values[indices])

    # Create initial geometry mask
    geom_image = np.zeros_like(image)
    geom_image[image > 0] = 1  # Mark detected geometry

    # Fill holes inside the geometry to ensure continuity
    filled_geom_image = binary_fill_holes(geom_image).astype(np.uint8)

    return image, filled_geom_image

from scipy.interpolate import griddata
from scipy.ndimage import binary_closing

def improved_creat_image(mesh_nodes, x_min, x_max, y_min, y_max, pixels_per_mm_x, pixels_per_mm_y):
    node_coords = mesh_nodes[:, :2]
    node_values = mesh_nodes[:, 2]

    # Compute grid resolution based on pixel density
    width = int((x_max - x_min) * pixels_per_mm_x)
    height = int((y_max - y_min) * pixels_per_mm_y)

    # Create pixel grid (center coordinates)
    dx = 1.0 / pixels_per_mm_x  # Pixel width in mm
    dy = 1.0 / pixels_per_mm_y  # Pixel height in mm
    x_coords = np.linspace(x_min, x_max, width)
    y_coords = np.linspace(y_min, y_max, height)
    X, Y = np.meshgrid(x_coords, y_coords)

    # Interpolate missing values using nearest neighbors
    interpolated_image = griddata(node_coords, node_values, (X, Y), method='nearest', fill_value=0)

    # Create binary mask from interpolated data
    geom_image = np.zeros_like(interpolated_image)
    geom_image[interpolated_image > 0] = 1

    # Apply morphological closing to remove small holes
    closed_geom_image = binary_closing(geom_image, iterations=3)

    # Fill any remaining holes inside the geometry
    final_geom_image = binary_fill_holes(closed_geom_image).astype(np.uint8)

    return interpolated_image, final_geom_image

# The function now:
# - Uses `griddata` to interpolate the mesh and ensure a continuous geometry representation.
# - Applies morphological closing to remove gaps.
# - Uses `binary_fill_holes` to ensure no holes remain inside the geometry.

def create_filled_geom(geom_img, acc):
    points = np.array(np.where(geom_img == 1)).T

    # Create KD-tree for efficient nearest neighbor search
    tree = cKDTree(points)

    # Parameters
    k = 3  # number of nearest neighbors to connect
    max_distance = acc  # maximum distance to consider for connections

    # Create output image
    connected_img = np.zeros_like(geom_img)

    # For each point, connect to its k nearest neighbors
    for point in points:
        # Find k+1 nearest neighbors (includes the point itself)
        distances, indices = tree.query(point, k=k+1)
        
        # Skip the first index (distance to self = 0)
        for neighbor_idx in indices[1:]:
            if distances[indices == neighbor_idx] <= max_distance:
                neighbor = points[neighbor_idx]
                
                # Create line between points
                y1, x1 = point
                y2, x2 = neighbor
                
                # Create points along the line
                num_points = max(abs(x2-x1), abs(y2-y1)) * 2
                x = np.linspace(x1, x2, int(num_points))
                y = np.linspace(y1, y2, int(num_points))
                
                # Round to nearest pixel coordinates
                x = np.round(x).astype(int)
                y = np.round(y).astype(int)
                
                # Set pixels along the line to 1
                connected_img[y, x] = 1

    # Add original points to ensure we don't lose any
    connected_img[geom_img == 1] = 1
    connected_img = binary_fill_holes(connected_img)
    return connected_img

def create_weighted_stress_field(stress_img, radius=5, decay_factor=2):
    # Get coordinates and values of stress points
    points = np.argwhere(stress_img != 0)  # y, x coordinates
    values = stress_img[stress_img != 0]    # stress values
    
    # Create KD-tree for efficient nearest neighbor search
    tree = cKDTree(points)
    
    # Create output array
    interpolated = np.zeros_like(stress_img, dtype=float)
    
    # Get all pixel coordinates
    y_coords, x_coords = np.mgrid[0:stress_img.shape[0], 0:stress_img.shape[1]]
    all_points = np.column_stack((y_coords.ravel(), x_coords.ravel()))
    
    # Find neighbors within radius for all points
    neighbors = tree.query_ball_point(all_points, radius)
    
    # Compute weighted stress for each point
    for idx, (y, x) in enumerate(all_points):
        if len(neighbors[idx]) > 0:
            # Get distances to neighbors
            neighbor_points = points[neighbors[idx]]
            neighbor_values = values[neighbors[idx]]
            
            # Compute distances
            distances = np.sqrt(np.sum((neighbor_points - [y, x])**2, axis=1))
            
            # Avoid division by zero
            distances = np.maximum(distances, 0.0001)
            
            # Compute weights using inverse distance weighting
            weights = 1.0 / (distances ** decay_factor)
            weights /= np.sum(weights)
            
            # Calculate weighted average
            interpolated[y, x] = np.sum(weights * neighbor_values)
    
    return interpolated


def find_worst_principal_stress(sigma_x, sigma_y, sigma_z, tau_xy, tau_yz, tau_zx):
    PS = find_principal_stresses_3d(sigma_x, sigma_y, sigma_z, tau_xy, tau_yz, tau_zx)
    return torch.max(PS, dim=-1).values



def find_edge_nodes(elements):
    # Step 1: Create a list of all edges
    edges = []
    for element in elements:
        # Create edges for the current element
        edges.append((element[0], element[1]))
        edges.append((element[1], element[2]))
        edges.append((element[2], element[0]))
    
    # Step 2: Count how many times each edge appears
    edge_count = defaultdict(int)
    for edge in edges:
        # Sort the edge to handle edge direction (e.g., (1, 2) and (2, 1) are the same)
        sorted_edge = tuple(sorted(edge))
        edge_count[sorted_edge] += 1
    
    # Step 3: Identify boundary edges and collect edge nodes
    edge_nodes = set()
    for edge, count in edge_count.items():
        if count == 1:  # Boundary edge
            edge_nodes.add(edge[0])
            edge_nodes.add(edge[1])
    
    return list(edge_nodes)
