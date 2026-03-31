# CGAL Mesh Optimization Python Binding

This project provides Python bindings for CGAL's tetrahedral mesh optimization using conformal mapping.

## Features

- **Efficient C++ backend**: Uses CGAL's high-performance mesh optimization algorithms
- **Zero-copy data transfer**: Direct numpy array manipulation without copying
- **Multiple element types**: Supports tetrahedra, hexahedra, pyramids, and wedges
- **Flexible input formats**: Works with meshio meshes or raw numpy arrays
- **Easy-to-use API**: Simple wrapper functions for common use cases

## Installation

1. Build the CGAL module:
```bash
mkdir build
cd build
cmake ..
make -j
```

2. Install Python dependencies:
```bash
pip install meshio numpy
```

## Usage

### Command Line Tool

The easiest way to optimize a mesh file:

```bash
python optimize_mesh.py input.mesh output.mesh --max-iter 100 --verbose
```

Options:
- `--max-iter INT`: Maximum iterations (default: 100)
- `--boundary-weight FLOAT`: Boundary weight (default: 1.0)
- `--min-edge-size FLOAT`: Minimum edge size (default: 1e-6)
- `--verbose`: Enable verbose output
- `--surface-projection`: Projection mode: none, locked, query, input (default: locked)

### Python API

#### Using meshio meshes:

```python
import meshio
from cgal_mesh_optimizer import optimize_meshio_mesh

# Load mesh
mesh = meshio.read("input.mesh")

# Optimize
optimized_mesh = optimize_meshio_mesh(mesh,
                                    max_iteration=100,
                                    verbose=True)

# Save result
optimized_mesh.write("output.mesh")
```

#### Using numpy arrays:

```python
import numpy as np
from cgal_mesh_optimizer import optimize_mesh

# Define mesh data
points = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype=np.float64)
tetrahedra = np.array([[0,1,2,3]], dtype=np.int32)
boundary_faces = np.array([[0,1,2], [0,1,3], [0,2,3], [1,2,3]], dtype=np.int32)

# Method 1: Dict format
cells = {'tetra': tetrahedra}
optimized_points = optimize_mesh(points, cells, boundary_faces)

# Method 2: List format
cells = [tetrahedra]  # [tetra, hexa, pyramid, wedge]
optimized_points = optimize_mesh(points, cells, boundary_faces)
```

#### Direct API (advanced):

```python
import cgal_mesh_optimization as cmo

data = cmo.MeshOptimizationData()
data.points = points
data.tetrahedra = tetrahedra
data.selected_triangles = boundary_faces
data.surface_projection_setting = cmo.ProjectionSetting.LOCKED_SELECTION
data.max_iteration = 100

cmo.optimize_mesh(data)  # Modifies data.points in place
```

## Supported Element Types

- **Tetrahedra**: 4 vertices
- **Hexahedra**: 8 vertices
- **Pyramids**: 5 vertices
- **Wedges**: 6 vertices

## Surface Projection Modes

- **locked**: Lock boundary vertices to their current positions
- **none**: No boundary constraints
- **query**: Use custom query functions (advanced)
- **input**: Use provided target surface data (advanced)

## Examples

See the following files for examples:
- `test.py`: Comparison of direct API vs wrapper
- `example_numpy.py`: Using numpy arrays directly
- `optimize_mesh.py`: Command-line tool

## Performance

- Zero-copy data transfer between Python and C++
- Multi-threaded optimization using OpenMP
- Efficient memory usage with in-place operations

# Mesh optimization executable

# Python Binding

# License
Usage of the Mesh_optimization package of the CGAL library is under GPL and cannot be used as such in commercial applications. This code extend to that by being under AGPL. Alternative licensing is available by contacting [Geometry Factory](https://geometryfactory.com/contactform/). 
