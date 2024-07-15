import argparse
from omegaconf import OmegaConf
import sys

import json
from dataset import get_scene

sys.path = [
    '../external/OneFormer-Colab',
    '../external/detectron2/projects/CropFormer',
    '../external/Depth-Anything/metric_depth',
    '../external/ovsam',
    '../external/dreamgaussian',
    '../external/One-2-3-45',
    '../external/Metric3D'
] + sys.path

from scipy.ndimage import binary_opening
from util import *
from modules import *
from background_model import BackgroundModel
from cam_utils import orbit_camera


def get_depth(scene, run_opt, K, out_dir, device):
    print(f"Scene depth obtained using {run_opt.relative} relative depth and {run_opt.metric} metric depth")

    if run_opt.relative == 'marigold':
        relative_depth = run_marigold(scene.image_pil, out_dir)
    elif run_opt.relative == 'metric3d':
        relative_depth = run_metric3d(scene.image_pil, K, device)
    else:
        relative_depth = None

    if run_opt.metric == 'gt' and scene.depth_map is not None:
        return scene.depth_map
    elif run_opt.metric == 'gt_ss' and scene.depth_map is not None:
        if relative_depth is None:
            raise Exception("Cannot use ground truth scale and shift without relative depth")
        return align_depth(relative_depth, scene.depth_map, scene.depth_mask)
    elif run_opt.metric == 'depthanything':
        metric_depth = run_depthanything(scene.image_pil, device)
    elif run_opt.metric == 'metric3d':
        metric_depth = run_metric3d(scene.image_pil, K, device)
    else:
        raise Exception("Metric depth is required")

    if relative_depth is None:
        return metric_depth
    else:
        return align_depth(relative_depth, metric_depth)


def get_components_masks(scene, run_opt, out_dir, device):
    print(f"Scene components segmented with {run_opt.holistic}, fg/bg separation using {run_opt.fg_bg} and instance labels using {run_opt.tagger}")

    if run_opt.holistic == 'entityv2':
        masks = run_entityv2(scene.image_np, threshold=0.1)
    else:
        raise Exception("Holistic image segmentation is required")
    print(f"Initial components in image: {len(masks)}")

    save_masks(masks, out_dir / 'initial_masks.png')
    masks = merge_masks(masks, run_opt.merge_masks.dilation, run_opt.merge_masks.threshold)
    save_masks(masks, out_dir / "processed_masks.png")
    print(f"Components in image after merging: {len(masks)}")

    if run_opt.fg_bg == 'gt':
        object_ids, background_ids = filter_component_masks(masks, scene.foreground_mask)
    elif run_opt.fg_bg == 'oneformer':
        object_ids, background_ids = run_oneformer(scene.image_pil, masks, device)
    elif run_opt.fg_bg == 'clipseg':
        object_ids, background_ids = run_clipseg(scene.image_pil, masks)
    elif run_opt.fg_bg == 'largest':        # assumes masks have been sorted in merge_masks function
        object_ids = np.arange(len(masks))[:-1]
        background_ids = [len(masks) - 1]
    else:
        raise Exception(f'fg/bg segmentation is required')

    print(f"Detected foreground instances in image: {len(object_ids)}")
    save_masks(masks[object_ids], out_dir / "foreground_instances.png")

    if run_opt.tagger == 'ovsam':
        instance_labels = run_ovsam(scene.image_pil, masks)
    else:
        instance_labels = ['' for _ in range(len(masks))]

    return masks, object_ids, background_ids, instance_labels


def get_camera(scene, run_opt):
    print(f"Using {run_opt} for camera calibration")
    if run_opt.intrinsics == 'gt' and scene.K is not None:
        K = scene.K
    elif run_opt.intrinsics == 'perspectivefields':
        K, _, _ = run_perspectivefields(scene.image_np)
    else:
        raise Exception('Camera calibration is required')

    if run_opt.pose == 'gt' and scene.c2w is not None:
        print("Using ground truth pose")
        pose = scene.c2w
    else:
        pose = np.eye(4)

    return K, pose


def get_background_mesh(image, pts3d, background_mask, K, depth_map, device):
    background_mask = binary_erosion(
        np.logical_and(background_mask, ~depth_edges_mask(depth_map)),
        np.ones((7, 7))
    )
    background_model = BackgroundModel().to(device)
    background_points = pts3d[background_mask]
    background_model.fit(background_points, image[background_mask], device)

    frustum_planes = compute_frustum_planes(K, image.shape[0], image.shape[1])
    background_mesh = background_model.extract_mesh(
        background_points.min(axis=0), background_points.max(axis=0),
        device, frustum_planes)
    return background_mesh


def complete_crop(crop, label, model, run_opt):
    if run_opt == 'our':
        completed = complete_object(crop, label, model)
        return segment_completed(completed, crop)
    else:
        return crop


def reconstruct_object(run_opt, out_dir, obj_id, obj_elevation, reprojected=True):
    if run_opt.obj_rec == 'dg':
        run_dreamgaussian(out_dir, obj_id, obj_elevation)
        mesh = trimesh.load(out_dir / "object_space" / f"{obj_id}.obj")
        blender2opencv = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )
        c2w_r = orbit_camera(obj_elevation, 0, 2) @ blender2opencv
        if reprojected:
            c2w_r[:3, -1] = 0
        mesh.apply_transform(np.linalg.inv(c2w_r))
    else:
        raise Exception(f'Reconstruction method not implemented')

    return mesh


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path to the yaml config file", default='configs/image.yaml', type=str)
    parser.add_argument('--gpu_idx', type=int, default=0, help='GPU index')
    parser.add_argument('--crop_size', type=int, default=512, help='object crop size')
    args, extras = parser.parse_known_args()
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    print("Running Gen3DSR with config")
    print(opt)
    
    assert (torch.cuda.is_available())
    device = f"cuda:{args.gpu_idx}"

    scene = get_scene(opt.scene.type, opt.scene.attributes)

    out_dir = Path(opt.scene.save_dir)
    print(f"Saving to {out_dir}")
    out_dir.mkdir(exist_ok=True, parents=True)
    (out_dir / "crops").mkdir(exist_ok=True)
    (out_dir / "object_space").mkdir(exist_ok=True)
    (out_dir / "reconstruction").mkdir(exist_ok=True)

    OmegaConf.save(config=opt, f=out_dir / "config.yaml")
    scene.image_pil.save(out_dir / 'input.png')

    K_img, pose = get_camera(scene, opt.run.calibration)
    depth_map = get_depth(scene, opt.run.depth, K_img, out_dir, device)
    masks, object_ids, background_ids, instance_labels = get_components_masks(scene, opt.run.segmentation, out_dir, device)

    pts3d = depth_to_points(depth_map[None], K_img)
    trimesh.PointCloud(pts3d.reshape(-1, 3), scene.image_np.reshape(-1, 3)).export(out_dir / 'depth_scene.ply')
    cam_params = {
        'K': K_img.tolist(),
        'c2w': pose.tolist(),
        'W': scene.image_pil.width,
        'H': scene.image_pil.height,
    }
    with open(out_dir / 'cam_params.json', 'w') as fp:
        json.dump(cam_params, fp)

    if not opt.run.only_foreground:
        background_mesh = get_background_mesh(
            scene.image_np, pts3d, np.any(masks[background_ids], axis=0),
            K_img, depth_map, device)
        background_mesh.apply_transform(pose)
        background_mesh.export(out_dir / 'reconstruction' / 'background.ply')
    else:
        background_mesh = None

    K_crop = get_crop_calibration(args.crop_size)

    acompletion_p = initialize_acompletion(device)
    zero123_p = initialize_zero123(device)

    if opt.run.single_object:
        masks = np.any(masks[object_ids], axis=0)[None]
        label = ",".join([instance_labels[i] for i in object_ids])
        label = label.replace(' (', ', ').replace(')', '')
        obj_id = f"fg_{label.replace(' ', '_')}"
        instance_labels = [label]
        object_ids = np.array([0])

    scene_mesh = trimesh.Scene([background_mesh])
    for i in range(len(masks[object_ids]) - 1, -1, -1):
        label = instance_labels[object_ids[i]]
        label = label.replace(' (', ', ').replace(')', '')
        obj_id = f"{i}_{label.replace(' ', '_')}"

        # prepare crop
        mask = binary_opening(masks[object_ids][i], np.ones((7, 7)))
        if mask.sum() < 400:
            print(f"Skipped too small object: {obj_id}")
            continue
        if opt.run.reproject:
            crop, normalization_mat, out_depth, c2w_def, mesh_pc = reproject_pixels(scene.image_np, mask, pts3d, K_crop, K_img, args.crop_size, opt.run.depth_edge_threshold)
        else:
            crop, crop_params = crop_object(scene.image_np, mask, args.crop_size)
        crop.save(out_dir / "crops" / f"{obj_id}_reproj.png")
        full_crop = complete_crop(crop, label, acompletion_p, opt.run.amodal_completion)
        full_crop.save(out_dir / "crops" / f"{obj_id}_rgba.png")

        estimate_elevation(np.array(full_crop.resize((256, 256))), out_dir / "object_space" / f"{obj_id}", zero123_p)
        obj_elevation = np.load(out_dir / "object_space" / f"{obj_id}" / "estimated_elevation.npy")
        try:
            clean()
            obj_mesh = reconstruct_object(opt.run, out_dir, obj_id, obj_elevation, opt.run.reproject)
            clean()
        except:
            print(f"Failed to reconstruct object {obj_id}")
            continue
        if opt.run.reproject:
            transform = align_to_depth_rep(obj_mesh, normalization_mat, out_depth, c2w_def, K_crop, args.crop_size)
        else:
            transform = align_to_depth(mask, pts3d, obj_mesh, K_crop, crop_params, args.crop_size)
        obj_mesh.apply_transform(transform)
        obj_mesh.apply_transform(pose)
        if opt.run.get('clip_bg', False):
            clip_object_with_bg(obj_mesh, background_mesh)
        obj_mesh.export(out_dir / 'reconstruction' / f"{obj_id}.glb")
        scene_mesh.add_geometry([obj_mesh])
        print(f"Saved, {obj_id}.glb")
    scene_mesh.export(out_dir / 'reconstruction' / 'full_scene.glb')
    
