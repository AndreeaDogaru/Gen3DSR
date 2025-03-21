import argparse
import numpy as np
import trimesh
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from util import compute_frustum_planes, create_triangles, depth_to_points, interpolate_array
import json


def project_from_camera_to_image(points_3d, K):
    """
    Project 3D points to the image plane using camera intrinsics matrix K.

    Parameters:
    - points_3d: numpy array of shape (n, 3) representing 3D points
    - K: numpy array of shape (3, 3) representing camera intrinsics matrix

    Returns:
    - points_2d: numpy array of shape (n, 2) representing projected points in the image plane
    """
    # Project 3D points to 2D using camera intrinsics matrix K
    points_2d_homogeneous = np.dot(K, points_3d.T).T
    points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2:]

    return points_2d


def load_all_objects(rec_path, frustum_planes):
    object_paths = rec_path.glob('*.glb')
    scene = trimesh.Trimesh()
    for object_path in object_paths:
        if object_path.stem != 'full_scene':
            mesh = trimesh.load(object_path)
            scene += mesh.geometry[list(mesh.geometry)[0]]

    for point_normal in frustum_planes:
        scene = scene.slice_plane(point_normal[:3], point_normal[3:])
    return scene


def mask_mesh_projection(mesh: trimesh.Trimesh, mask, K):
    vertices = mesh.vertices
    vertices2d = project_from_camera_to_image(vertices, K)
    valid = np.isclose(interpolate_array(mask, vertices2d[:, [1, 0]]), 1)
    new_vertices = vertices[valid]
    new_indices = np.empty(len(vertices))
    new_indices[valid] = np.arange(valid.sum())
    new_faces = new_indices[mesh.faces[np.all(valid[mesh.faces], 1)]]
    return trimesh.Trimesh(new_vertices, new_faces)


def load_bg_meshes(gt_path, rec_path, K):
    gt_depth = np.load(gt_path)
    depth_mask = gt_depth < 100
    points = depth_to_points(gt_depth[None], K)
    triangles = create_triangles(gt_depth.shape[0], gt_depth.shape[1])
    gt_bg_mesh = mask_mesh_projection(trimesh.Trimesh(points.reshape(-1, 3), triangles), depth_mask, K)
    if rec_path.is_file():
        rec_bg_mesh = mask_mesh_projection(trimesh.load(rec_path), depth_mask, K)
    else:
        rec_bg_mesh = trimesh.Trimesh()
    return gt_bg_mesh, rec_bg_mesh


def compute_metrics_meshes(gt, pred, num_points=1000000, thresholds=[0.1, 0.01, 0.001], eps=1e-6):
    metrics = {}
    gt_points = gt.as_open3d.sample_points_uniformly(num_points)
    pred_points = pred.as_open3d.sample_points_uniformly(num_points)

    dist_gt_pred = np.array(gt_points.compute_point_cloud_distance(pred_points))
    dist_pred_gt = np.array(pred_points.compute_point_cloud_distance(gt_points))
    metrics["Chamfer"] = (dist_gt_pred.mean() + dist_pred_gt.mean()) / 2

    for t in thresholds:
        precision = 100.0 * (dist_pred_gt < t).mean()
        recall = 100.0 * (dist_gt_pred < t).mean()
        f1 = (2.0 * precision * recall) / (precision + recall + eps)
        metrics[f"Precision@{t}"] = precision
        metrics[f"Recall@{t}"] = recall
        metrics[f"F1@{t}"] = f1
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", help="Path to the location of the Front3D data",
                        default="../imgs/FRONT3D", required=False)
    parser.add_argument("--rec_path", help="Path to the directory with the reconstructed scenes",
                        default="../out/front3d", required=False)
    parser.add_argument("--wo_bg", help="Only evaluates the reconstruction of objects in the scene",
                        action='store_true')
    args = parser.parse_args()
    data_root = Path(args.data_root)
    rec_path = Path(args.rec_path)

    metrics = defaultdict(list)

    with open(data_root / "scene_ids") as f:
        scene_ids = f.read().split('\n')

    for scene_id in tqdm(scene_ids):
        with open(data_root / "annotation" / f"annotation_00{scene_id}.json") as f:
            annotation = json.load(f)
        K = np.array(annotation['camera_intrinsics'])
        frustum_planes = compute_frustum_planes(K, 968, 1296)
        
        gt = trimesh.load(data_root/ "sceneobjgt" / f"sceneobjgt_00{scene_id}.ply")
        rec = load_all_objects(rec_path / f"rec_{scene_id}" / "reconstruction", frustum_planes)
        if not args.wo_bg:
            gt_bg, rec_bg = load_bg_meshes(
                data_root / "bgdepth" / f"bgdepth_00{scene_id}.npy", 
                rec_path / f"rec_{scene_id}" / "reconstruction" / "background.ply", 
                K
            )
            gt += gt_bg
            rec += rec_bg

        if len(rec.vertices) == 0:
            continue
        current_metrics = compute_metrics_meshes(gt, rec)
        print(f"scene: {scene_id} Chamfer: {current_metrics['Chamfer']} F0.1: {current_metrics['F1@0.1']}")
        for m in current_metrics:
            metrics[m].append(current_metrics[m])

    print("All scenes: ")
    avg_metrics = {}
    for m in metrics.keys():
        avg_metrics[f"{m}_mean"] = np.mean(metrics[m])
    print(avg_metrics)
    metrics.update(avg_metrics)

    with open(rec_path / f'metrics_{"obj" if args.wo_bg else "full"}.json', 'w') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)


