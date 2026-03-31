import numpy as np
import cgal_mesh_optimization as cmo
import meshio


def optimize_mesh(points, cells, boundary_faces=None, curve_edges=None, target_surface=None, target_curve_network=None, surface_query=None, curve_query=None, **kwargs):
    """
    Optimize a tetrahedral mesh using CGAL conformal optimization.

    Parameters:
    -----------
    points : numpy.ndarray
        Vertex coordinates, shape (n_points, 3)
    cells : dict or list
        Cell connectivity. Can be:
        - dict: {'tetra': array, 'hexahedra': array, ...}
        - list: [tetra_array, hexa_array, ...] (assumes tetra, hexa, pyramid, wedge order)
    boundary_faces : numpy.ndarray, optional
        Boundary face connectivity for surface projection, shape (n_faces, 3)
    curve_edges : numpy.ndarray, optional
        Curve network edge connectivity, shape (n_edges, 2)
    target_surface : dict, optional
        Target surface data with keys 'points', 'triangles', 'patch_ids'
    target_curve_network : dict, optional
        Target curve network data with keys 'points', 'edges', 'patch_ids'
    surface_query : callable, optional
        Lambda function for surface projection queries (currently not fully supported due to binding limitations)
    curve_query : callable, optional
        Lambda function for curve projection queries (currently not fully supported due to binding limitations)
    **kwargs : optimization parameters
        max_iteration : int, default 100
        boundary_weight : float, default 1.0
        minimum_edge_size : float, default 1e-6
        verbose : bool, default False
        surface_projection : str, default 'locked'
            'none', 'locked', 'query', 'input'
        curves_projection : str, default 'none'
            'none', 'query', 'input'

    Returns:
    --------
    numpy.ndarray
        Optimized vertex coordinates, shape (n_points, 3)

    Examples:
    ---------
    # Using dict format
    cells = {'tetra': tetra_array, 'hexahedra': hexa_array}
    optimized_points = optimize_mesh(points, cells, boundary_faces)

    # Using list format
    cells = [tetra_array, hexa_array, pyramid_array, wedge_array]
    optimized_points = optimize_mesh(points, cells)

    # With custom parameters
    optimized_points = optimize_mesh(points, cells,
                                   max_iteration=200,
                                   boundary_weight=2.0,
                                   verbose=True)

    # With curve network
    optimized_points = optimize_mesh(points, cells,
                                   boundary_faces=boundary_faces,
                                   curve_edges=curve_edges)

    # With target surfaces
    target_surf = {
        'points': surface_points,
        'triangles': surface_triangles,
        'patch_ids': surface_patch_ids
    }
    optimized_points = optimize_mesh(points, cells,
                                   target_surface=target_surf,
                                   surface_projection='input')
    """
    # Create optimization data structure
    data = cmo.MeshOptimizationData()

    # Set points
    data.points = points.astype(np.float64)

    # Handle cells - support both dict and list formats
    if isinstance(cells, dict):
        if 'tetra' in cells or 'tetrahedra' in cells:
            key = 'tetra' if 'tetra' in cells else 'tetrahedra'
            data.tetrahedra = cells[key].astype(np.int32)
        if 'hexahedra' in cells or 'hexa' in cells:
            key = 'hexahedra' if 'hexahedra' in cells else 'hexa'
            data.hexahedra = cells[key].astype(np.int32)
        if 'pyramid' in cells or 'pyramids' in cells:
            key = 'pyramid' if 'pyramid' in cells else 'pyramids'
            data.pyramids = cells[key].astype(np.int32)
        if 'wedge' in cells or 'wedges' in cells:
            key = 'wedge' if 'wedge' in cells else 'wedges'
            data.wedges = cells[key].astype(np.int32)
    elif isinstance(cells, list):
        # Assume order: tetrahedra, hexahedra, pyramids, wedges
        if len(cells) > 0 and cells[0] is not None:
            data.tetrahedra = cells[0].astype(np.int32)
        if len(cells) > 1 and cells[1] is not None:
            data.hexahedra = cells[1].astype(np.int32)
        if len(cells) > 2 and cells[2] is not None:
            data.pyramids = cells[2].astype(np.int32)
        if len(cells) > 3 and cells[3] is not None:
            data.wedges = cells[3].astype(np.int32)
    else:
        raise ValueError("cells must be a dict or list of numpy arrays")

    # Set boundary faces if provided
    if boundary_faces is not None:
        data.selected_triangles = boundary_faces.astype(np.int32)
        surface_proj = kwargs.get('surface_projection', 'locked').lower()
        if surface_proj == 'none':
            data.surface_projection_setting = cmo.ProjectionSetting.NONE
        elif surface_proj == 'locked':
            data.surface_projection_setting = cmo.ProjectionSetting.LOCKED_SELECTION
        elif surface_proj == 'query':
            data.surface_projection_setting = cmo.ProjectionSetting.QUERY_FUNCTION
        elif surface_proj == 'input':
            data.surface_projection_setting = cmo.ProjectionSetting.INPUT_DATA
        else:
            raise ValueError(f"Unknown surface_projection: {surface_proj}")

    # Set curve edges if provided
    if curve_edges is not None:
        data.selected_edges = curve_edges.astype(np.int32)
        curves_proj = kwargs.get('curves_projection', 'none').lower()
        if curves_proj == 'none':
            data.curves_projection_setting = cmo.ProjectionSetting.NONE
        elif curves_proj == 'query':
            data.curves_projection_setting = cmo.ProjectionSetting.QUERY_FUNCTION
        elif curves_proj == 'input':
            data.curves_projection_setting = cmo.ProjectionSetting.INPUT_DATA
        else:
            raise ValueError(f"Unknown curves_projection: {curves_proj}")

    # Set target surface if provided
    if target_surface is not None:
        data.target_surface.points = target_surface['points'].astype(np.float64)
        data.target_surface.triangles = target_surface['triangles'].astype(np.int32)
        if 'patch_ids' in target_surface:
            data.target_surface.patch_ids = target_surface['patch_ids'].astype(np.int32)

    # Set target curve network if provided
    if target_curve_network is not None:
        data.target_curve_network.points = target_curve_network['points'].astype(np.float64)
        data.target_curve_network.edges = target_curve_network['edges'].astype(np.int32)
        if 'patch_ids' in target_curve_network:
            data.target_curve_network.patch_ids = target_curve_network['patch_ids'].astype(np.int32)

    # Set query functions if provided
    if surface_query is not None:
        data.set_surface_query(surface_query)
    if curve_query is not None:
        data.set_curve_query(curve_query)

    # Set optimization parameters
    data.max_iteration = kwargs.get('max_iteration', 100)
    data.boundary_weight = kwargs.get('boundary_weight', 1.0)
    data.minimum_edge_size = kwargs.get('minimum_edge_size', 1e-6)
    data.verbose = kwargs.get('verbose', False)

    # Run optimization
    optimized_points = cmo.optimize_mesh(data)

    return optimized_points


def optimize_meshio_mesh(mesh, **kwargs):
    """
    Optimize a meshio mesh object.

    Parameters:
    -----------
    mesh : meshio.Mesh
        Input mesh
    **kwargs : optimization parameters
        Same as optimize_mesh function

    Returns:
    --------
    meshio.Mesh
        Optimized mesh
    """
    # Extract cells
    cells = {}
    boundary_faces = None

    for cell_block in mesh.cells:
        cell_type = cell_block.type
        if cell_type == 'tetra':
            cells['tetra'] = cell_block.data
        elif cell_type == 'hexahedron':
            cells['hexahedra'] = cell_block.data
        elif cell_type == 'pyramid':
            cells['pyramid'] = cell_block.data
        elif cell_type == 'wedge':
            cells['wedge'] = cell_block.data
        elif cell_type == 'triangle' and boundary_faces is None:
            # Use triangles as boundary faces if no explicit boundary provided
            boundary_faces = cell_block.data

    # If no boundary faces found in cells, try to find them
    if boundary_faces is None and 'triangle' in mesh.cells_dict:
        boundary_faces = mesh.cells_dict['triangle']

    # Optimize
    optimized_points = optimize_mesh(mesh.points, cells, boundary_faces, **kwargs)

    # Create new mesh with optimized points
    return meshio.Mesh(points=optimized_points, cells=mesh.cells)