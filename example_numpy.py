#!/usr/bin/env python3
"""
Example: Using CGAL mesh optimization with numpy arrays directly
"""

import numpy as np
import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'build'))
from cgal_mesh_optimizer import optimize_mesh

# Create a simple tetrahedral mesh
points = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.5, 1.0, 0.0],
    [0.5, 0.5, 1.0],
    [2.0, 0.0, 0.0],  # Additional points for second tetrahedron
    [1.5, 1.0, 0.0],
    [1.5, 0.5, 1.0]
], dtype=np.float64)

# Two tetrahedra
tetrahedra = np.array([
    [0, 1, 2, 3],
    [1, 4, 5, 6]
], dtype=np.int32)

# Boundary triangles (all outer faces)
boundary_faces = np.array([
    [0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3],  # First tetra faces
    [1, 4, 5], [1, 4, 6], [1, 5, 6], [4, 5, 6]   # Second tetra faces
], dtype=np.int32)

print("Original points:")
print(points)

# Method 1: Using dict format
print("\n=== Using dict format ===")
cells_dict = {'tetra': tetrahedra}
optimized_points_dict = optimize_mesh(points.copy(), cells_dict, boundary_faces,
                                    max_iteration=50, verbose=True)

# Method 2: Using list format
print("\n=== Using list format ===")
cells_list = [tetrahedra]
optimized_points_list = optimize_mesh(points.copy(), cells_list, boundary_faces,
                                    max_iteration=50, verbose=True)

print(f"\nBoth methods produce identical results: {np.allclose(optimized_points_dict, optimized_points_list)}")
print("Optimized points:")
print(optimized_points_dict)