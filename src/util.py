import copy

import cv2
import trimesh
import torch
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.interpolate import interpn
from sklearn.linear_model import RANSACRegressor, LinearRegression
import rembg
from scipy.spatial.transform import Rotation as R


def initialize_acompletion(device):
    from diffusers import StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel
    Pix2PixCfg = {
        'initial_model': "runwayml/stable-diffusion-v1-5",
        'finetuned_unet': "../external/checkpoints/amodal_completion",
    }
    unet = UNet2DConditionModel.from_pretrained(
        Pix2PixCfg['finetuned_unet'],
    )
    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        Pix2PixCfg['initial_model'], unet=unet
    ).to(device)

    def disabled_safety_checker(images, clip_input):
        if len(images.shape) == 4:
            num_images = images.shape[0]
            return images, [False] * num_images
        else:
            return images, False

    pipeline.safety_checker = disabled_safety_checker

    return pipeline


def initialize_zero123(device):
    from zero123 import Zero123Pipeline
    pipe = Zero123Pipeline.from_pretrained(
        "ashawkey/zero123-xl-diffusers",
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).to(device)
    pipe.image_encoder.eval()
    pipe.vae.eval()
    pipe.unet.eval()
    pipe.clip_camera_projection.eval()
    return pipe


def initialize_oneformer(device):
    from oneformer import (
        add_oneformer_config,
        add_common_config,
        add_swin_config,
        add_dinat_config,
        add_convnext_config,
    )
    from detectron2.config import get_cfg
    from detectron2.projects.deeplab import add_deeplab_config
    from detectron2.data import MetadataCatalog
    from demo.defaults import DefaultPredictor

    SWIN_CFG_DICT = {"cityscapes": "../external/OneFormer-Colab/configs/cityscapes/oneformer_swin_large_IN21k_384_bs16_90k.yaml",
                     "coco": "../external/OneFormer-Colab/configs/coco/oneformer_swin_large_IN21k_384_bs16_100ep.yaml",
                     "ade20k": "../external/OneFormer-Colab/configs/ade20k/oneformer_swin_large_IN21k_384_bs16_160k.yaml", }

    DINAT_CFG_DICT = {"cityscapes": "../external/OneFormer-Colab/configs/cityscapes/oneformer_dinat_large_bs16_90k.yaml",
                      "coco": "../external/OneFormer-Colab/configs/coco/oneformer_dinat_large_bs16_100ep.yaml",
                      "ade20k": "../external/OneFormer-Colab/configs/ade20k/oneformer_dinat_large_IN21k_384_bs16_160k.yaml", }

    def setup_cfg(dataset, model_path, use_swin):
        # load config from file and command-line arguments
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_common_config(cfg)
        add_swin_config(cfg)
        add_dinat_config(cfg)
        add_convnext_config(cfg)
        add_oneformer_config(cfg)
        if use_swin:
            cfg_path = SWIN_CFG_DICT[dataset]
        else:
            cfg_path = DINAT_CFG_DICT[dataset]
        cfg.merge_from_file(cfg_path)
        cfg.MODEL.DEVICE = device
        cfg.MODEL.WEIGHTS = model_path
        cfg.freeze()
        return cfg

    def setup_modules(dataset, model_path, use_swin):
        cfg = setup_cfg(dataset, model_path, use_swin)
        predictor = DefaultPredictor(cfg)
        metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST_PANOPTIC[0] if len(cfg.DATASETS.TEST_PANOPTIC) else "__unused"
        )
        return predictor, metadata

    predictor, metadata = setup_modules("ade20k", "../external/checkpoints/250_16_dinat_l_oneformer_ade20k_160k.pth", False)

    my_stuff = [
        'window ',
        'door',
        'curtain',
        'mirror',
        'fence',
        'rail',
        'column, pillar',
        'stairs',
        'screen door, screen',
        'bannister, banister, balustrade, balusters, handrail',
        'step, stair',
    ]

    my_thing = [
        'plant',
        'tent',
        'crt screen',
        'cradle',
        'blanket, cover'
    ]

    custom_thing = []
    for thing in metadata.thing_classes:
        if thing not in my_stuff:
            custom_thing.append(metadata.stuff_classes.index(thing))

    for thing in my_thing:
        custom_thing.append(metadata.stuff_classes.index(thing))

    return predictor, metadata, custom_thing


def save_masks(masks, path):
    n, i, j = np.where(masks)
    col = np.zeros((masks.shape[1], masks.shape[2], 3), dtype=np.uint8)
    palette = np.random.randint(25, 230, (len(masks), 3))
    col[i, j] = palette[n]
    cv2.imwrite(str(path), col)


def merge_masks(masks, dilate_size=9, over_threshold=0.5):
    kernel = np.ones((dilate_size, dilate_size))
    mask_sizes = masks.sum((-1, -2))
    order = np.argsort(mask_sizes)
    masks = masks[order]
    mask_sizes = mask_sizes[order]
    full_masks = np.ones(len(masks), dtype=bool)
    dilated_masks = np.stack([binary_dilation(m, kernel) for m in masks])
    for i in range(len(masks)):
        if full_masks[i]:
            intersects = np.logical_and(dilated_masks[i][None], masks[full_masks]).sum((-1, -2))
            overlaps = (intersects + 1e-6) / (mask_sizes[full_masks] + 1e-6) > over_threshold
            overlaps[full_masks[:i].sum()] = False
            masks[i] |= np.bitwise_or.reduce(masks[full_masks][overlaps], axis=0)
            dilated_masks[i] |= np.bitwise_or.reduce(dilated_masks[full_masks][overlaps], axis=0)
            mask_sizes[i] += mask_sizes[full_masks][overlaps].sum()
            full_masks[np.where(full_masks)[0][overlaps]] = False
    return masks[full_masks]


def depth_to_points(depth, K=None, R=None, t=None):
    """
    Reference: https://github.com/isl-org/ZoeDepth/blob/edb6daf45458569e24f50250ef1ed08c015f17a7/zoedepth/utils/geometry.py
    """
    Kinv = np.linalg.inv(K)
    if R is None:
        R = np.eye(3)
    if t is None:
        t = np.zeros(3)

    height, width = depth.shape[1:3]

    x = np.arange(width)
    y = np.arange(height)
    coord = np.stack(np.meshgrid(x, y), -1)
    coord = np.concatenate((coord, np.ones_like(coord)[:, :, [0]]), -1)  # z=1
    coord = coord.astype(np.float32)
    coord = coord[None]  # bs, h, w, 3

    D = depth[:, :, :, None, None]
    pts3D_1 = D * Kinv[None, None, None, ...] @ coord[:, :, :, :, None]
    # from reference to target viewpoint
    pts3D_2 = R[None, None, None, ...] @ pts3D_1 + t[None, None, None, :, None]
    return pts3D_2[:, :, :, :3, 0][0]


def depth_edges_mask(depth):
    """
    Reference: https://github.com/isl-org/ZoeDepth/blob/edb6daf45458569e24f50250ef1ed08c015f17a7/ui/gradio_im_to_3d.py
    Returns a mask of edges in the depth map.
    Args:
    depth: 2D numpy array of shape (H, W) with dtype float32.
    Returns:
    mask: 2D numpy array of shape (H, W) with dtype bool.
    """
    # Compute the x and y gradients of the depth map.
    depth_dx, depth_dy = np.gradient(depth)
    # Compute the gradient magnitude.
    depth_grad = np.sqrt(depth_dx ** 2 + depth_dy ** 2)
    # Compute the edge mask.
    mask = depth_grad > 0.05
    return mask


def compute_frustum_planes(K, h, w, near=0.1, far=5):
    coord = np.array([
        [0, 0], [w, 0], [0, h], [w, h],
        [0, 0], [w, 0], [0, h], [w, h]
    ])
    coord = np.concatenate((coord, np.ones_like(coord)[..., [0]]), -1)
    coord = coord.astype(np.float32)
    depths = np.ones_like(coord[:, :1])
    depths[:4] = near
    depths[4:] = far
    corners = depths * (np.linalg.inv(K) @ coord.T).T
    planes = np.array([
        [0, 2, 4, 6],
        [1, 5, 3, 7],
        [0, 4, 1, 5],
        [2, 3, 6, 7]
    ])
    normals = np.cross(
        corners[planes[:, 0]] - corners[planes[:, 3]],
        corners[planes[:, 1]] - corners[planes[:, 2]]
    )
    return np.concatenate([corners[planes[:, 0]], normals], axis=-1)


def rotation_matrix_to_align_vectors(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)

    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    axis = np.cross(v1, v2)

    if np.linalg.norm(axis) == 0:
        return np.eye(3)

    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(np.dot(v1, v2))
    rotation = R.from_rotvec(angle * axis)
    rotation_matrix = np.eye(4)
    rotation_matrix[:3, :3] = rotation.as_matrix()
    return rotation_matrix


def reproject_pixels(image, mask, pts3d, K_crop, K_image, crop_size, edge_th):
    triangles = create_triangles(image.shape[0], image.shape[1], mask=mask)
    mesh = trimesh.Trimesh(pts3d.reshape(-1, 3), faces=triangles, vertex_colors=image[..., :3].reshape(-1, 3))
    mesh.remove_unreferenced_vertices()

    normalization_mat = np.eye(4)

    # used to compute a tighter bounding box around the object
    cam_vec = -mesh.triangles_center.copy()
    cam_vec /= np.linalg.norm(cam_vec)
    prod = (cam_vec * np.array(mesh.face_normals)).sum(axis=-1)
    mesh = trimesh.Trimesh(pts3d.reshape(-1, 3), faces=triangles[prod > edge_th], vertex_colors=image[..., :3].reshape(-1, 3))
    mesh.remove_unreferenced_vertices()

    mid_point = (mesh.vertices.min(axis=0) + mesh.vertices.max(axis=0)) / 2
    mid_point *= pts3d[mask, 2].min() / mid_point[2]
    mesh.apply_translation(-mid_point)
    normalization_mat[:3, -1] -= mid_point

    rotation = rotation_matrix_to_align_vectors(mid_point, [0, 0, 1])
    mesh.apply_transform(rotation)
    normalization_mat = rotation @ normalization_mat

    v_mask = np.logical_and(
        mesh.vertices[:, -1] < np.quantile(mesh.vertices[:, -1], 0.9), 
        mesh.vertices[:, -1] > np.quantile(mesh.vertices[:, -1], 0.05)
        )
    vertices = mesh.vertices[v_mask]

    scale = 1.0 / np.max((vertices.max(axis=0) - vertices.min(axis=0))[:-1])
    mesh.apply_scale(scale)
    vertices *= scale
    normalization_mat[:3] *= scale

    translation = -(vertices.min(axis=0) + vertices.max(axis=0)) / 2
    translation[-1] = -(vertices[:, -1].min() + 0.5) 
    mesh.apply_translation(translation)
    normalization_mat[:3, -1] += translation

    c2w_def = np.eye(4)
    c2w_def[:3, -1] = [0, 0, -2]

    out = intersect_rays_mesh(mesh, K_crop, c2w_def, crop_size)
    inters = trimesh.PointCloud(out[0])
    inters.apply_transform(np.linalg.inv(normalization_mat))
    K = np.eye(4)
    K[:3, :3] = K_image
    inters.apply_transform(K)
    coords = inters[:, :2] / inters[:, 2:]
    inters_colors = interpolate_array(image, coords[:, ::-1]).round().clip(0, 255)
    reproj_mask = np.zeros_like(mask, shape=(crop_size, crop_size))
    reproj_img = np.zeros_like(image, shape=(crop_size, crop_size, 3))
    y, x = np.unravel_index(out[1], (crop_size, crop_size))
    reproj_img[y, x] = inters_colors
    reproj_mask[y, x] = 1
    reproj = Image.fromarray(np.concatenate([reproj_img, np.uint8(reproj_mask[:, :, None]) * 255], axis=-1))
    return reproj, normalization_mat, out, c2w_def, mesh


def intersect_rays_mesh(mesh, K_def, c2w_def, crop_size):
    tx = np.linspace(0, crop_size - 1, crop_size)
    ty = np.linspace(0, crop_size - 1, crop_size)
    pixels_x, pixels_y = np.meshgrid(tx, ty)
    p = np.stack([pixels_x, pixels_y, np.ones_like(pixels_y)], axis=-1)
    p = np.einsum('ij,mnj->mni', np.linalg.inv(K_def), p)
    rays_v = p / np.linalg.norm(p, ord=2, axis=-1, keepdims=True)
    rays_v = np.einsum('ij,mnj->mni', c2w_def[:3, :3], rays_v)
    rays_o = np.broadcast_to(c2w_def[:3, -1], rays_v.shape)
    intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
    out = intersector.intersects_location(rays_o.reshape(-1, 3), rays_v.reshape(-1, 3), multiple_hits=False)
    return out


def create_triangles(h, w, mask=None):
    """
    Reference: https://github.com/google-research/google-research/blob/e96197de06613f1b027d20328e06d69829fa5a89/infinite_nature/render_utils.py#L68
    Creates mesh triangle indices from a given pixel grid size.
        This function is not and need not be differentiable as triangle indices are
        fixed.
    Args:
    h: (int) denoting the height of the image.
    w: (int) denoting the width of the image.
    Returns:
    triangles: 2D numpy array of indices (int) with shape (2(W-1)(H-1) x 3)
    """
    x, y = np.meshgrid(range(w - 1), range(h - 1))
    tl = y * w + x
    tr = y * w + x + 1
    bl = (y + 1) * w + x
    br = (y + 1) * w + x + 1
    triangles = np.array([tl, bl, tr, br, tr, bl])
    triangles = np.transpose(triangles, (1, 2, 0)).reshape(
        ((w - 1) * (h - 1) * 2, 3))
    if mask is not None:
        mask = mask.reshape(-1)
        triangles = triangles[mask[triangles].all(1)]
    return triangles


def estimate_elevation(img, cache_dir, zero123, dtype=torch.float16):
    from elevation_estimate.utils.elev_est_api import elev_est_api
    cache_dir.mkdir(exist_ok=True)
    img = img.astype(np.float32) / 255.0
    img = img[..., :3] * img[..., -1:] + (1 - img[..., -1:])
    img = Image.fromarray(np.uint8((img * 255).clip(0, 255)))
    delta_x_2 = [-10, 10, 0, 0]
    delta_y_2 = [0, 0, -10, 10]
    tensor_def = {
        "device": zero123.device,
        "dtype": dtype
    }
    out = zero123(
        [img] * len(delta_x_2),
        torch.tensor(delta_x_2, **tensor_def),      # elevation
        torch.tensor(delta_y_2, **tensor_def),      # azimuth
        torch.zeros(len(delta_x_2), **tensor_def)   # distance
    )
    file_paths = []
    for i in range(len(out[0])):
        opath = str((cache_dir / f"{i}.png").absolute())
        file_paths.append(opath)
        out[0][i].save(opath)
    elev = elev_est_api(file_paths)
    if elev is not None:
        elev -= 90
    else:
        elev = 0
        print("!!!Failed to estimate elevation!!!")
    np.save(cache_dir / 'estimated_elevation.npy', elev)


def get_crop_calibration(crop_size, fov=49.1):
    # fov aligns with dreamgaussian which aligns with zero123
    focal = 1 / np.tan(np.deg2rad(fov) / 2)
    return np.array([
        [focal * crop_size / 2, 0, crop_size / 2],
        [0, focal * crop_size / 2, crop_size / 2],
        [0, 0, 1]
    ])


def interpolate_array(input_array, coordinates):
    """
    Interpolate values from the input array at specified coordinates using bilinear interpolation.

    Parameters:
    - input_array: 2D NumPy array (HxW)
    - coordinates: 2D NumPy array (nx2) containing coordinates (row, column)

    Returns:
    - sampled_values: 1D NumPy array containing sampled values
    """
    # Generate grid coordinates
    H, W = input_array.shape[:2]
    grid_x, grid_y = np.arange(W), np.arange(H)
    # Interpolate values at specified coordinates
    sampled_values = interpn((grid_y, grid_x), input_array, coordinates,
                             method='linear', bounds_error=False, fill_value=0)
    return sampled_values


def clip_object_with_bg(obj_mesh, bg_mesh):
    # assumes camera in origin with no rotation
    p = np.array(obj_mesh.vertices)
    rays_v = p / np.linalg.norm(p, ord=2, axis=-1, keepdims=True)
    intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(bg_mesh)
    out = intersector.intersects_location(np.zeros_like(rays_v), rays_v, multiple_hits=False)
    bg_closer = np.linalg.norm(out[0], axis=-1) < np.linalg.norm(obj_mesh.vertices[out[1]], axis=-1)
    obj_mesh.vertices[out[1]] = np.where(bg_closer[:, None], out[0], obj_mesh.vertices[out[1]])


def align_depth(relative_depth, metric_depth, mask=None, min_samples=0.2):
    regressor = RANSACRegressor(estimator=LinearRegression(fit_intercept=True), min_samples=min_samples)
    if mask is not None:
        regressor.fit(relative_depth[mask].reshape(-1, 1), metric_depth[mask].reshape(-1, 1))
    else:
        regressor.fit(relative_depth.reshape(-1, 1), metric_depth.reshape(-1, 1))
    depth = regressor.predict(relative_depth.reshape(-1, 1)).reshape(relative_depth.shape)
    return depth


def crop_object(image: np.ndarray, mask: np.ndarray, crop_size=256):
    x, y, w, h = cv2.boundingRect(np.uint8(mask))
    max_size = max(w, h)
    ratio = 0.7
    side_len = int(max_size / ratio)
    padded_image = np.zeros((side_len, side_len, 3), dtype=image.dtype)
    padded_mask = np.zeros((side_len, side_len), dtype=mask.dtype)
    center = side_len // 2
    padded_image[center - h // 2:center - h // 2 + h, center - w // 2:center - w // 2 + w] = image[y:y + h, x:x + w]
    padded_mask[center - h // 2:center - h // 2 + h, center - w // 2:center - w // 2 + w] = mask[y:y + h, x:x + w]
    resized_image = cv2.resize(padded_image, (crop_size, crop_size), cv2.INTER_LANCZOS4)
    resized_mask = cv2.resize(np.uint8(padded_mask), (crop_size, crop_size), cv2.INTER_LANCZOS4) == 1
    offset_x = x + (w - side_len) / 2
    offset_y = y + (h - side_len) / 2
    scale_factor = crop_size / side_len
    crop = Image.fromarray(np.concatenate([resized_image, np.uint8(resized_mask[:, :, None]) * 255], axis=-1))
    return crop, (offset_x, offset_y, scale_factor)


def segment_completed(completed_crop, original_crop):
    orig_mask = np.array(original_crop)[..., -1] / 255 > 0.5
    new_mask = np.array(rembg.remove(completed_crop, rembg.new_session("isnet-general-use"), post_process_mask=True))
    new_mask[:, :, :3][orig_mask] = np.array(completed_crop)[orig_mask]
    new_mask[:, :, -1][orig_mask] = 255
    return Image.fromarray(new_mask)


def align_to_depth(mask, pts3d, mesh, K_crop, crop_params, crop_size):
    import open3d as o3d
    out_source = intersect_rays_mesh(mesh, K_crop, np.eye(4), 512)
    y, x = np.unravel_index(out_source[1], (crop_size, crop_size))
    img_y = np.int32(np.round(y / crop_params[2] + crop_params[1])).clip(0, mask.shape[0]-1)
    img_x = np.int32(np.round(x / crop_params[2] + crop_params[0])).clip(0, mask.shape[1]-1)
    valid_points = (img_y > 0) & (img_y < mask.shape[0]) & (img_x > 0) & (img_x < mask.shape[1]) & mask[img_y, img_x]
    source_points = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(out_source[0][valid_points]))
    target_points = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts3d[img_y[valid_points], img_x[valid_points]]))
    estimator = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True)
    corres = o3d.utility.Vector2iVector(np.tile(np.arange(np.sum(valid_points)), 2).reshape(2, -1).T)
    transform = estimator.compute_transformation(source_points, target_points, corres)
    return transform


def align_to_depth_rep(obj_mesh, normalization_mat, out_depth, c2w_crop, K_crop, crop_size):
    crop_shape = (crop_size, crop_size)
    w2c_crop = np.linalg.inv(c2w_crop)
    out_rec = intersect_rays_mesh(obj_mesh, K_crop, c2w_crop, crop_size)

    depth_mask = np.zeros(crop_shape, dtype=bool)
    depth_mask[np.unravel_index(out_depth[1], crop_shape)] = 1
    rec_mask = np.zeros(crop_shape, dtype=bool)
    rec_mask[np.unravel_index(out_rec[1], crop_shape)] = 1
    mask_both = depth_mask & rec_mask

    pc_rec = trimesh.PointCloud(out_rec[0][mask_both[np.unravel_index(out_rec[1], crop_shape)]])
    pc_depth = trimesh.PointCloud(out_depth[0][mask_both[np.unravel_index(out_depth[1], crop_shape)]])

    pc_rec.apply_transform(w2c_crop)
    pc_depth.apply_transform(w2c_crop)

    regressor = RANSACRegressor(estimator=LinearRegression(fit_intercept=False), min_samples=0.2)
    regressor.fit(pc_rec[:, -1:].reshape(-1, 1), pc_depth[:, -1:].reshape(-1, 1))
    transform = w2c_crop.copy()
    transform[:3] *= regressor.estimator_.coef_[0, 0]
    transform = np.linalg.inv(normalization_mat) @ c2w_crop @ transform
    return transform


def filter_component_masks(masks, foreground_mask, threshold=0.5):
    all_instances = np.arange(len(masks))
    is_foreground = ((masks & foreground_mask).sum((-1, -2)) + 1e-6) / (masks.sum((-1, -2)) + 1e-6) > threshold
    return all_instances[is_foreground], all_instances[~is_foreground]


def clean():
    import gc
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()