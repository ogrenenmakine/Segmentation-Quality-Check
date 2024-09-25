import numpy as np
import nibabel as nib
from scipy.special import sph_harm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.measure import marching_cubes
from scipy.interpolate import griddata

def load_and_combine_masks_volume(mask_filenames):
    """
    Load masks from given filenames and combine them into a single 3D volume.
    """
    # Assume all masks have the same shape and affine
    first_img = nib.load(mask_filenames[0])
    combined_mask_data = np.zeros(first_img.shape, dtype=bool)
    
    for path in mask_filenames:
        print(f"Loading mask: {path}")
        mask_img = nib.load(path)
        mask_data = mask_img.get_fdata() > 0
        combined_mask_data |= mask_data  # Logical OR to combine masks
    
    affine = first_img.affine
    return combined_mask_data, affine

def extract_surface(mask_data, level=0.5):
    """
    Extract the surface of a binary mask using the Marching Cubes algorithm.
    """
    print("Extracting surface using Marching Cubes...")
    verts, faces, normals, values = marching_cubes(mask_data, level=level)
    print(f"Extracted {verts.shape[0]} vertices from the surface.")
    return verts

def cartesian_to_spherical(x, y, z):
    """
    Convert Cartesian coordinates to spherical coordinates.
    """
    rho = np.sqrt(x**2 + y**2 + z**2)
    # Avoid division by zero
    rho_safe = np.where(rho == 0, np.finfo(float).eps, rho)
    theta = np.arctan2(y, x)  # Azimuthal angle
    phi = np.arccos(np.clip(z / rho_safe, -1.0, 1.0))  # Polar angle
    return np.stack([rho, theta, phi], axis=-1)

def spherical_to_cartesian(rho, theta, phi):
    """
    Convert spherical coordinates to Cartesian coordinates.
    """
    x = rho * np.sin(phi) * np.cos(theta)
    y = rho * np.sin(phi) * np.sin(theta)
    z = rho * np.cos(phi)
    return np.stack([x, y, z], axis=-1)

def compute_sh_coefficients_grid(rho_grid, theta_grid, phi_grid, l_max):
    """
    Compute spherical harmonics coefficients on a regular grid.
    """
    coeffs = []
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            Y_lm = sph_harm(m, l, theta_grid, phi_grid)
            # Integration over the sphere using numpy's sum and accounting for the Jacobian determinant sin(phi)
            coeff = np.sum(rho_grid * Y_lm.conj() * np.sin(phi_grid))
            coeffs.append((l, m, coeff))
    return coeffs

def reconstruct_shape_from_sh(coeffs, theta_grid, phi_grid):
    """
    Reconstruct rho values from spherical harmonics coefficients.
    """
    result = np.zeros(theta_grid.shape, dtype=complex)
    for (l, m, c) in coeffs:
        Y_lm = sph_harm(m, l, theta_grid, phi_grid)
        result += c * Y_lm
    return result

def visualize_sh_coefficients(coeffs, l_max):
    """
    Visualize the magnitudes of spherical harmonics coefficients.
    """
    magnitudes = [np.abs(c[2]) for c in coeffs]
    indices = range(len(magnitudes))
    plt.figure(figsize=(10, 6))
    plt.bar(indices, magnitudes)
    plt.xlabel('Coefficient Index')
    plt.ylabel('Magnitude')
    plt.title(f'Spherical Harmonics Coefficients Magnitudes (l_max={l_max})')
    plt.show()

def visualize_reconstructed_shape(coords, title="3D Shape"):
    """
    Visualize a 3D point cloud.
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=1, alpha=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    plt.show()

def main():
    # Define the mask filenames
    mask_filenames = [
        'lung_lower_lobe_left.nii.gz',
        'lung_lower_lobe_right.nii.gz',
        'lung_upper_lobe_left.nii.gz',
        'lung_upper_lobe_right.nii.gz',
        'lung_middle_lobe_right.nii.gz'
    ]
    
    # Load and combine masks into a volume
    print("Loading and combining masks into a volume...")
    combined_mask_data, affine = load_and_combine_masks_volume(mask_filenames)
    print("Combined mask volume shape:", combined_mask_data.shape)
    
    # Extract the surface of the combined mask
    print("Extracting surface from combined mask...")
    surface_verts = extract_surface(combined_mask_data)

    # After extracting surface vertices
    print("Surface vertices shape:", surface_verts.shape)

    # Visualize the surface vertices
    print("Visualizing surface vertices...")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Use a subset of points for faster rendering
    subset_size = min(10000, len(surface_verts))
    subset_indices = np.random.choice(len(surface_verts), subset_size, replace=False)
    subset_verts = surface_verts[subset_indices]

    # Plot the subset of vertices
    scatter = ax.scatter(subset_verts[:, 0], subset_verts[:, 1], subset_verts[:, 2], 
                        c=subset_verts[:, 2], cmap='viridis', s=1, alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Surface Vertices Visualization (Subset)')

    # Add a color bar
    plt.colorbar(scatter, ax=ax, label='Z-coordinate')

    # Adjust the viewing angle for better visibility
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.show()
    
    # Apply affine transformation to surface vertices to get world coordinates
    surface_world_coords = nib.affines.apply_affine(affine, surface_verts)
    print("Surface world coordinates shape:", surface_world_coords.shape)
    
    # Convert surface coordinates to spherical coordinates
    print("Converting surface coordinates to spherical coordinates...")
    rho_theta_phi = cartesian_to_spherical(
        surface_world_coords[:, 0],
        surface_world_coords[:, 1],
        surface_world_coords[:, 2]
    )
    rho = rho_theta_phi[:, 0]
    theta = rho_theta_phi[:, 1]
    phi = rho_theta_phi[:, 2]
    
    # Check for invalid spherical coordinates
    invalid_indices = np.isnan(rho_theta_phi).any(axis=1) | np.isinf(rho_theta_phi).any(axis=1)
    if np.sum(invalid_indices) > 0:
        print(f"Warning: Found {np.sum(invalid_indices)} invalid spherical coordinates. These will be removed.")
        # Remove invalid entries
        rho = rho[~invalid_indices]
        theta = theta[~invalid_indices]
        phi = phi[~invalid_indices]
    
    # Interpolate rho onto a regular grid of theta and phi
    print("Interpolating data onto a regular grid...")
    num_theta = 100  # Adjust grid resolution as needed
    num_phi = 100
    theta_grid, phi_grid = np.meshgrid(
        np.linspace(-np.pi, np.pi, num_theta),
        np.linspace(0, np.pi, num_phi)
    )
    
    # Prepare data for interpolation
    points = np.column_stack((theta, phi))
    values = rho
    grid_points = np.column_stack((theta_grid.flatten(), phi_grid.flatten()))
    
    # Perform interpolation
    print("Performing grid interpolation...")
    rho_grid = griddata(points, values, grid_points, method='linear', fill_value=0)
    rho_grid = rho_grid.reshape(theta_grid.shape)
    
    # Handle any NaNs in the grid
    rho_grid = np.nan_to_num(rho_grid)
    
    # Compute spherical harmonics coefficients
    l_max = 15  # Adjust as needed
    print(f"Computing spherical harmonics coefficients up to l_max={l_max}...")
    coeffs = compute_sh_coefficients_grid(rho_grid, theta_grid, phi_grid, l_max)
    print(f"Computed {len(coeffs)} spherical harmonics coefficients.")
    
    # Visualize the coefficients
    print("Visualizing spherical harmonics coefficients...")
    visualize_sh_coefficients(coeffs, l_max)
    
    # Reconstruct the shape from coefficients
    print("Reconstructing shape from spherical harmonics coefficients...")
    rho_reconstructed = reconstruct_shape_from_sh(coeffs, theta_grid, phi_grid)
    
    # Convert back to Cartesian coordinates
    coords_reconstructed = spherical_to_cartesian(rho_reconstructed, theta_grid, phi_grid)
    coords_reconstructed = coords_reconstructed.reshape(-1, 3)
    
    # Visualize the reconstructed shape
    print("Visualizing reconstructed shape...")
    visualize_reconstructed_shape(coords_reconstructed, title="Reconstructed Shape")
    
    print("Processing complete.")

if __name__ == "__main__":
    main()