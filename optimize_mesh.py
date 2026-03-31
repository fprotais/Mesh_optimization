#!/usr/bin/env python3
"""
CGAL Mesh Optimization Tool

This script optimizes tetrahedral meshes using CGAL's conformal optimization.

Usage:
    python optimize_mesh.py input.mesh output.mesh [options]

Options:
    --max-iter INT          Maximum number of iterations (default: 100)
    --boundary-weight FLOAT Boundary weight parameter (default: 1.0)
    --min-edge-size FLOAT   Minimum edge size (default: 1e-6)
    --verbose               Enable verbose output
    --surface-projection    Surface projection mode: none, locked, query, input (default: locked)
"""

import argparse
import sys
import os

# Add the build directory to Python path for the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'build'))

try:
    import meshio
    from cgal_mesh_optimizer import optimize_meshio_mesh
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure meshio is installed and the CGAL module is built.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Optimize tetrahedral meshes using CGAL')
    parser.add_argument('input', help='Input mesh file')
    parser.add_argument('output', help='Output mesh file')
    parser.add_argument('--max-iter', type=int, default=100,
                       help='Maximum number of iterations (default: 100)')
    parser.add_argument('--boundary-weight', type=float, default=1.0,
                       help='Boundary weight parameter (default: 1.0)')
    parser.add_argument('--min-edge-size', type=float, default=1e-6,
                       help='Minimum edge size (default: 1e-6)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--surface-projection', choices=['none', 'locked', 'query', 'input'],
                       default='locked', help='Surface projection mode (default: locked)')

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        sys.exit(1)

    try:
        # Read input mesh
        print(f"Reading mesh from {args.input}...")
        mesh = meshio.read(args.input)

        # Count elements
        n_vertices = len(mesh.points)
        n_cells = sum(len(cell_block.data) for cell_block in mesh.cells)
        print(f"Mesh loaded: {n_vertices} vertices, {n_cells} cells")

        # Optimize mesh
        print("Starting optimization...")
        optimized_mesh = optimize_meshio_mesh(
            mesh,
            max_iteration=args.max_iter,
            boundary_weight=args.boundary_weight,
            minimum_edge_size=args.min_edge_size,
            verbose=args.verbose,
            surface_projection=args.surface_projection
        )

        # Write output mesh
        print(f"Writing optimized mesh to {args.output}...")
        optimized_mesh.write(args.output)

        print("Optimization completed successfully!")

    except Exception as e:
        print(f"Error during optimization: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()