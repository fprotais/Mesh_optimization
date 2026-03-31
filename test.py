import cgal_mesh_optimization as cmo
import meshio
import numpy as np
from cgal_mesh_optimizer import optimize_meshio_mesh

# Read the mesh
mesh = meshio.read("B0.mesh")

print(f"Original mesh: {len(mesh.points)} vertices, {sum(len(cb.data) for cb in mesh.cells)} cells")

# Method 1: Using the low-level API directly
print("\n=== Method 1: Direct API ===")
data = cmo.MeshOptimizationData()
data.points = mesh.points.astype(np.float64)
data.tetrahedra = mesh.cells_dict["tetra"].astype(np.int32)
data.selected_triangles = mesh.cells_dict["triangle"].astype(np.int32)
data.surface_projection_setting = cmo.ProjectionSetting.LOCKED_SELECTION
data.max_iteration = 10
data.verbose = True

optimized_points_direct = cmo.optimize_mesh(data)
print("Direct API optimization completed")

# Method 2: Using the high-level wrapper
print("\n=== Method 2: High-level wrapper ===")
optimized_mesh = optimize_meshio_mesh(mesh, max_iteration=10, verbose=True)
print("Wrapper optimization completed")

# Compare results
print(f"\nResults comparison:")
print(f"Direct API points shape: {optimized_points_direct.shape}")
print(f"Wrapper mesh points shape: {optimized_mesh.points.shape}")
print(f"Points are identical: {np.allclose(optimized_points_direct, optimized_mesh.points)}")

# Save the optimized mesh
optimized_mesh.write("B0_optimized_wrapper.mesh")
print("Optimized mesh saved to B0_optimized_wrapper.mesh")