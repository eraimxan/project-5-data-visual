#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import open3d as o3d
import numpy as np
import copy
import sys
from pathlib import Path

MODEL_PATH = "model.ply"
VOXEL_SIZE = 0.05
POISSON_DEPTH = 9

def print_header(title):
    print("\n" + "="*80)
    print(title)
    print("="*80)

def mesh_stats(mesh, name="Mesh"):
    n_vertices = np.asarray(mesh.vertices).shape[0] if isinstance(mesh, o3d.geometry.TriangleMesh) else 0
    n_triangles = np.asarray(mesh.triangles).shape[0] if isinstance(mesh, o3d.geometry.TriangleMesh) else 0
    has_vtx_colors = isinstance(mesh, o3d.geometry.TriangleMesh) and mesh.has_vertex_colors()
    has_vtx_normals = isinstance(mesh, o3d.geometry.TriangleMesh) and mesh.has_vertex_normals()
    print(f"{name}: vertices={n_vertices}, triangles={n_triangles}, has_color={has_vtx_colors}, has_normals={has_vtx_normals}")

def pcd_stats(pcd, name="PointCloud"):
    n_points = np.asarray(pcd.points).shape[0]
    has_colors = pcd.has_colors()
    has_normals = pcd.has_normals()
    print(f"{name}: points={n_points}, has_color={has_colors}, has_normals={has_normals}")

def vox_stats(vox, name="VoxelGrid"):
    n_vox = len(vox.get_voxels())
    has_color = any([np.any(v.color) for v in vox.get_voxels()]) if n_vox > 0 else False
    print(f"{name}: voxels={n_vox}, has_color={has_color}")

def draw(geoms, title):
    if not isinstance(geoms, list):
        geoms = [geoms]
    o3d.visualization.draw_geometries(geoms, window_name=title)

def load_any_geometry(path):
    ext = Path(path).suffix.lower()
    if ext in [".ply", ".stl", ".obj", ".off", ".gltf", ".glb"]:
        try:
            mesh = o3d.io.read_triangle_mesh(path, enable_post_processing=True)
            if mesh is not None and len(mesh.triangles) > 0:
                return mesh, "mesh"
        except Exception:
            pass
    pcd = o3d.io.read_point_cloud(path)
    if pcd is None or np.asarray(pcd.points).size == 0:
        raise RuntimeError(f"Failed to load geometry from {path}")
    return pcd, "pcd"

def ensure_triangle_mesh(geom):
    if isinstance(geom, o3d.geometry.TriangleMesh):
        return geom
    elif isinstance(geom, o3d.geometry.PointCloud):
        hull, _ = geom.compute_convex_hull()
        hull.compute_vertex_normals()
        return hull
    else:
        raise TypeError("Unsupported geometry type")

def to_point_cloud_from_mesh(mesh):
    n_target = max(5000, min(100000, len(mesh.vertices) * 5))
    pcd = mesh.sample_points_poisson_disk(number_of_points=n_target, init_factor=5)
    return pcd

def estimate_normals_if_needed(pcd):
    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL_SIZE*3, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(k=50)
    return pcd

def create_plane(size=1.0, center=(0,0,0), normal=(0,0,1)):
    plane = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=1e-3)
    plane.compute_vertex_normals()
    plane.paint_uniform_color([0.3, 0.3, 0.3])
    plane.translate(np.array(center) - plane.get_center())
    z_axis = np.array([0,0,1.0])
    n = np.array(normal, dtype=float)
    n = n / (np.linalg.norm(n) + 1e-12)
    v = np.cross(z_axis, n)
    c = np.dot(z_axis, n)
    if np.linalg.norm(v) < 1e-8:
        R = np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]],[v[2], 0, -v[0]],[-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * (1/(1+c+1e-12))
    plane.rotate(R, center=plane.get_center())
    return plane

def clip_pointcloud_with_plane(pcd, plane_model):
    a,b,c,d = plane_model
    pts = np.asarray(pcd.points)
    side = pts @ np.array([a,b,c]) + d
    keep = side <= 0.0
    pcd_out = o3d.geometry.PointCloud()
    pcd_out.points = o3d.utility.Vector3dVector(pts[keep])
    if pcd.has_colors():
        pcd_out.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[keep])
    if pcd.has_normals():
        pcd_out.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals)[keep])
    return pcd_out

def create_highlight_cube(center, size, color=(1,0,0)):
    """Create a wireframe cube to highlight extreme points"""
    cube = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=size)
    cube.paint_uniform_color(color)
    
    # Convert to wireframe
    wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(cube)
    wireframe.paint_uniform_color(color)
    
    # Position the cube at the center point
    wireframe.translate(np.array(center) - wireframe.get_center())
    
    return wireframe

def create_solid_cube(center, size, color=(1,0,0), transparency=0.3):
    """Create a semi-transparent solid cube"""
    cube = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=size)
    cube.paint_uniform_color(color)
    
    # Make it semi-transparent
    cube.compute_vertex_normals()
    
    # Position the cube at the center point
    cube.translate(np.array(center) - cube.get_center())
    
    return cube

def color_gradient_by_axis(pcd, axis='z'):
    pts = np.asarray(pcd.points)
    if pts.shape[0] == 0:
        return pcd, None, None
    axes = {'x':0,'y':1,'z':2}
    ax = axes.get(axis.lower(), 2)
    vals = pts[:, ax]
    vmin, vmax = vals.min(), vals.max()
    denom = (vmax - vmin) if vmax > vmin else 1.0
    t = (vals - vmin) / denom
    colors = np.column_stack([t, np.zeros_like(t), 1.0 - t])
    pcd.colors = o3d.utility.Vector3dVector(colors)
    i_min = int(np.argmin(vals))
    i_max = int(np.argmax(vals))
    return pcd, pts[i_min], pts[i_max]

def create_arrow(center, color=(1,0,0), size=1.0):
    """Create an arrow pointing to the extreme point"""
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=size * 0.1,
        cone_radius=size * 0.2,
        cylinder_height=size * 0.8,
        cone_height=size * 0.4
    )
    arrow.paint_uniform_color(color)
    arrow.compute_vertex_normals()
    
    # Position arrow above the point
    arrow.translate(center + np.array([0, 0, size * 0.6]))
    
    return arrow

def main():
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = MODEL_PATH
    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        sys.exit(1)

    print_header("1) Load and visualize original model")
    geom, kind = load_any_geometry(model_path)
    if kind == "mesh":
        mesh = geom
        if not mesh.has_vertex_normals() and len(mesh.triangles) > 0:
            mesh.compute_vertex_normals()
        mesh_stats(mesh, "OriginalMesh")
        draw(mesh, "Original Mesh")
    else:
        pcd0 = geom
        pcd_stats(pcd0, "LoadedPointCloud")
        draw(pcd0, "Original Point Cloud")
        mesh = ensure_triangle_mesh(pcd0)

    print_header("2) Convert model to point cloud (sampling from mesh if needed)")
    if isinstance(geom, o3d.geometry.PointCloud):
        pcd = copy.deepcopy(geom)
    else:
        pcd = to_point_cloud_from_mesh(mesh)
    pcd_stats(pcd, "PointCloud (from model)")
    draw(pcd, "Sampled Point Cloud")

    print_header("3) Poisson surface reconstruction + crop artifacts")
    pcd = estimate_normals_if_needed(pcd)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
        mesh_rec, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=POISSON_DEPTH)
    bbox = pcd.get_axis_aligned_bounding_box()
    mesh_rec = mesh_rec.crop(bbox)
    mesh_rec.compute_vertex_normals()
    mesh_stats(mesh_rec, "ReconstructedMesh")
    draw(mesh_rec, "Reconstructed Mesh (Poisson)")

    print_header("4) Voxelization from point cloud")
    vox = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=VOXEL_SIZE)
    vox_stats(vox, "VoxelGrid")
    draw(vox, "Voxelized Model")

    print_header("5) Add plane next to object")
    center = pcd.get_center()
    extent = pcd.get_axis_aligned_bounding_box().get_extent()
    plane_size = float(max(extent) * 1.5)
    plane = create_plane(size=plane_size, center=center + np.array([0.0, 0.0, -0.25*extent[2]]), normal=(0.4, -0.2, 0.9))
    draw([plane, mesh if isinstance(geom, o3d.geometry.TriangleMesh) else pcd], "Plane + Original Model")

    print_header("6) Clip by plane (remove right side)")
    plane_normal = np.array([0.4, -0.2, 0.9], dtype=float)
    plane_normal /= np.linalg.norm(plane_normal) + 1e-12
    plane_point = np.asarray(plane.get_center())
    a, b, c = plane_normal
    d = -plane_normal @ plane_point
    plane_model = (a, b, c, d)

    pcd_clipped = clip_pointcloud_with_plane(pcd, plane_model)
    pcd_stats(pcd_clipped, "ClippedPointCloud")

    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_rec)
    kept_left = tmesh.clip_plane(
        point=o3d.core.Tensor(plane_point, dtype=o3d.core.Dtype.Float32),
        normal=o3d.core.Tensor((-plane_normal).astype(np.float32))
    )
    mesh_clipped = kept_left.to_legacy()
    mesh_clipped.compute_vertex_normals()
    mesh_stats(mesh_clipped, "ClippedMesh")
    draw(mesh_clipped, "Clipped Mesh (Right removed)")

    print_header("7) Gradient recolor and extrema")
    target_pcd = pcd_clipped if isinstance(pcd_clipped, o3d.geometry.PointCloud) and len(pcd_clipped.points) > 0 else pcd
    target_pcd.colors = o3d.utility.Vector3dVector(np.zeros((len(target_pcd.points), 3)))
    target_pcd, pmin, pmax = color_gradient_by_axis(target_pcd, axis='z')
    print(f"Extrema along Z: z_min at {pmin}, z_max at {pmax}")

    # Calculate appropriate cube size based on model extent
    bbox_extent = target_pcd.get_axis_aligned_bounding_box().get_extent()
    cube_size = float(max(bbox_extent) * 0.1)  # 10% of max extent

    # Create wireframe cubes for extreme points
    min_cube_wireframe = create_highlight_cube(pmin, cube_size, color=[0.0, 0.0, 1.0])  # Blue for minimum
    max_cube_wireframe = create_highlight_cube(pmax, cube_size, color=[1.0, 0.0, 0.0])  # Red for maximum

    # Create semi-transparent solid cubes
    min_cube_solid = create_solid_cube(pmin, cube_size, color=[0.0, 0.0, 1.0], transparency=0.3)
    max_cube_solid = create_solid_cube(pmax, cube_size, color=[1.0, 0.0, 0.0], transparency=0.3)

    # Create arrows pointing to extreme points
    arrow_size = cube_size * 2.0
    min_arrow = create_arrow(pmin, color=[0.0, 0.0, 1.0], size=arrow_size)
    max_arrow = create_arrow(pmax, color=[1.0, 0.0, 0.0], size=arrow_size)

    # Create coordinate frame for reference
    bbox = target_pcd.get_axis_aligned_bounding_box()
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=float(max(0.5, bbox.get_extent().max() * 0.35)), origin=np.zeros(3)
    )

    # Visualize everything together
    draw(
        [target_pcd, min_cube_wireframe, max_cube_wireframe, min_cube_solid, max_cube_solid, min_arrow, max_arrow, axes],
        "3D Model with Z-Extremes (highlighted with cubes)"
    )

if __name__ == "__main__":
    main()