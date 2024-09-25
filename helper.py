import numpy as np
from scipy.special import sph_harm
import matplotlib.pyplot as plt 
from skimage.measure import marching_cubes
import pyshtools as pysh

def convert_cartesian_to_spherical(vertices):
    """
    Convert the vertices in cartesian coordinates to spherical coordinates.
    """
    r = np.sqrt(vertices[:, 0]**2 + vertices[:, 1]**2 + vertices[:, 2]**2)
    theta = np.arccos(vertices[:, 2] / r)
    phi = np.arctan2(vertices[:, 1], vertices[:, 0])
    return np.stack((r, theta, phi), axis=-1)

def convert_spherical_to_cartesian(vertices):
    """
    Convert the vertices in spherical coordinates to cartesian coordinates.
    """
    x = vertices[:, 0] * np.sin(vertices[:, 1]) * np.cos(vertices[:, 2])
    y = vertices[:, 0] * np.sin(vertices[:, 1]) * np.sin(vertices[:, 2])
    z = vertices[:, 0] * np.cos(vertices[:, 1])
    return np.stack((x, y, z), axis=-1)



def visualize_mesh(vertices, title):
    """
    Visualize the binary mask.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=1)
    ax.set_title(title)
    plt.show()

def compute_spherical_harmonics(vertices, lmax):
    """
    Compute the spherical harmonic coefficients of a 3D mesh.
    
    Parameters:
    vertices (ndarray): Nx3 array of mesh vertex coordinates
    lmax (int): Maximum degree of spherical harmonics to compute
    
    Returns:
    coeffs (ndarray): Complex spherical harmonic coefficients
    """
    # Convert vertices to spherical coordinates
    x, y, z = vertices.T
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    
    # Compute spherical harmonic coefficients
    coeffs = np.zeros((lmax+1, 2*lmax+1), dtype=complex)
    for l in range(lmax+1):
        for m in range(-l, l+1):
            Y = sph_harm(m, l, phi, theta)
            coeffs[l, m+l] = np.sum(Y * r**l * np.sin(theta))
    
    return coeffs
