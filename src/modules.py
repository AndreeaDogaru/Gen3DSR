import os
from pathlib import Path
import torchvision.transforms.functional as Ftv
from util import *


def run_perspectivefields(image):
    from perspective2d import PerspectiveFields
    from perspective2d.utils.utils import general_vfov_to_focal

    img_bgr = image[..., ::-1]
    H, W, _ = img_bgr.shape
    pf_model = PerspectiveFields('Paramnet-360Cities-edina-uncentered').eval().cuda()
    pred = pf_model.inference(img_bgr=img_bgr)

    roll = np.radians(pred['pred_roll'].cpu().item())
    pitch = np.radians(pred['pred_pitch'].cpu().item())
    vfov = np.radians(pred['pred_general_vfov'].cpu().item())
    cx_rel = pred['pred_rel_cx'].cpu().item()
    cy_rel = pred['pred_rel_cy'].cpu().item()
    focal_rel = general_vfov_to_focal(cx_rel, cy_rel, 1, vfov, degree=False)
    f = focal_rel * H
    cx = (cx_rel + 0.5) * W
    cy = (cy_rel + 0.5) * H
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]])
    return K, roll, pitch


def run_marigold(image:Image.Image, out_dir:Path):
    save_path = (out_dir / "marigold").absolute()
    save_path.mkdir(exist_ok=True)
    image.save(save_path / f"input.png")
    bash_script = (
        'python ../external/Marigold/run.py '
        f'--input_rgb_dir {save_path} '
        f'--output_dir {save_path} '
        '--checkpoint ../external/checkpoints/marigold-lcm-v1-0 '
        '--ensemble_size 5 '
        '--denoise_steps 2'
    )     
    print(bash_script)
    os.system(bash_script)
    relative_depth = np.load(save_path / 'depth_npy' / 'input_pred.npy')
    return relative_depth


def run_depthanything(image:Image.Image, device):
    from zoedepth.utils.config import get_config
    from zoedepth.models.builder import build_model
    config = get_config("zoedepth", "eval", "nyu", pretrained_resource="local::../external/checkpoints/depth_anything_metric_depth_indoor.pt")
    model = build_model(config).to(device)
    return model.infer_pil(image)


def run_entityv2(image: np.ndarray, threshold=0.1, max_size=1500):
    #For cleanup
    from detectron2.data import MetadataCatalog, DatasetCatalog
    from detectron2.modeling import BACKBONE_REGISTRY, SEM_SEG_HEADS_REGISTRY
    def_keys = list(DatasetCatalog.keys())
    ####

    from detectron2.config import get_cfg
    from demo_cropformer.predictor import VisualizationDemo
    from detectron2.projects.deeplab import add_deeplab_config
    from mask2former import add_maskformer2_config
    CropFormerCfg = {
        'config_file': "../external/detectron2/projects/CropFormer/configs/entityv2/entity_segmentation/cropformer_hornet_3x.yaml",
        'opts': ["MODEL.WEIGHTS", "../external/checkpoints/CropFormer_hornet_3x_03823a.pth"],
    }
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(CropFormerCfg['config_file'])
    cfg.merge_from_list(CropFormerCfg['opts'])
    cfg.freeze()
    segmentor = VisualizationDemo(cfg)

    # CropFormer expects BGR
    orig_size = image.shape
    if max(image.shape) > max_size:
        scale_factor = max_size / max(image.shape[:-1])
        image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
    img = image.copy()[:, :, ::-1]
    predictions = segmentor.run_on_image(img)

    #hacky cleanup - is there a better way?
    extra_keys = list(DatasetCatalog.keys())
    for key in extra_keys:
        if key not in def_keys:
            DatasetCatalog.remove(key)
            MetadataCatalog.remove(key)
    BACKBONE_REGISTRY._obj_map = {}
    SEM_SEG_HEADS_REGISTRY._obj_map = {}
    #####
    
    pred_masks = predictions["instances"].pred_masks
    pred_scores = predictions["instances"].scores
    selected_indexes = (pred_scores >= threshold)
    selected_masks = pred_masks[selected_indexes]
    selected_masks = Ftv.resize(selected_masks.cpu(), orig_size[:-1]).numpy() > 0.5  # make binary
    return selected_masks


def complete_object(crop, label, model):
    image, mask = np.split(np.array(crop) / 255, (3, ), axis=-1)
    image[mask[:, :, 0] < 0.5] = 0.5
    completed = model(prompt=label, image=image,
                      num_inference_steps=50, image_guidance_scale=1.5,
                      guidance_scale=8.5, num_images_per_prompt=1).images[0]
    return completed


def run_clipseg(image, masks):
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

    background_prompts = ["background", "floor", "wall", "curtain", "window", "ceiling", "table"]
    foreground_prompts = ["object", "furniture"]

    img = image
    inputs = processor(
        text=background_prompts + foreground_prompts,
        images=[image] * (len(background_prompts) + len(foreground_prompts)),
        padding="max_length", return_tensors="pt"
    )
    predicted = torch.sigmoid(model(**inputs).logits)
    back_pred = (predicted[:len(background_prompts)] > 0.5).any(dim=0)
    fore_pred = (predicted[-len(foreground_prompts):] > 0.1).any(dim=0)
    foreground_mask = torch.logical_or(~back_pred, fore_pred).numpy()
    foreground_mask = np.array(Image.fromarray(foreground_mask).resize(img.size))
    return filter_component_masks(masks, foreground_mask)


def run_oneformer(image, masks, device):
    predictor, metadata, thing_classes_ids = initialize_oneformer(device)
    W, H = image.size
    f = 640 * 4 / W
    img = np.array(image.resize((int(W * f), int(H * f)), Image.BILINEAR))[:, :, ::-1]  # not sure if RGB or BGR
    predictions = predictor(img, "semantic")['sem_seg'].argmax(dim=0).cpu().numpy()
    is_thing = Image.fromarray(np.isin(predictions, thing_classes_ids))
    is_thing = np.array(is_thing.resize((W, H), Image.NEAREST))
    return filter_component_masks(masks, is_thing)


def run_dreamgaussian(scene_dir, obj_id, elevation):
    save_path = (scene_dir / "object_space" / f"{obj_id}").absolute()
    main_dir_path = Path.cwd().absolute()
    img_path = (scene_dir / "crops" / f"{obj_id}_rgba.png").absolute()
    os.chdir('../external/dreamgaussian/')
    bash_script = f'python main.py --config configs/image.yaml input={img_path} save_path={save_path} elevation={elevation} force_cuda_rast=True'
    print(bash_script)
    os.system(bash_script)
    bash_script = f'python main2.py --config configs/image.yaml input={img_path} save_path={save_path} elevation={elevation} force_cuda_rast=True'
    print(bash_script)
    os.system(bash_script)
    os.chdir(main_dir_path)


def run_ovsam(image: Image.Image, masks, max_samples=5):
    from image_tagger import ImageTagger
    tagger = ImageTagger()
    tags, scores = tagger.infer(image, masks, max_samples=max_samples)
    return tags


def run_metric3d(image: Image.Image, K, device):
    from infer_metric3d import predict_depth_normal
    pred_depth, pred_normal = predict_depth_normal(image, fx=K[0, 0], fy=K[1, 1], c_xy=(K[0, 2], K[1, 2]), device=device)
    return pred_depth

