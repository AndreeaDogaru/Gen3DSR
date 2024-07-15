import torch
import cv2
import numpy as np

try:
    from mmcv.utils import Config, DictAction
except:
    from mmengine import Config, DictAction
from mono.model.monodepth_model import get_configured_monodepth_model
from mono.utils.running import load_ckpt
from mono.utils.do_test import transform_test_data_scalecano, get_prediction


def predict_depth_normal(img, model_selection="vit-large", fx=1000.0, fy=1000.0, c_xy=None, device="cuda"):
    if model_selection == "vit-small":
        cfg = Config.fromfile('../external/Metric3D/mono/configs/HourglassDecoder/vit.raft5.small.py')
        model = get_configured_monodepth_model(cfg, )
        model, _, _, _ = load_ckpt('../external/checkpoints/metric_depth_vit_small_800k.pth', model, strict_match=False)
    elif model_selection == "vit-large":
        cfg = Config.fromfile('../external/Metric3D/mono/configs/HourglassDecoder/vit.raft5.large.py')
        model = get_configured_monodepth_model(cfg, )
        model, _, _, _ = load_ckpt('../external/checkpoints/metric_depth_vit_large_800k.pth', model, strict_match=False)
    elif model_selection == "vit-giant":
        cfg = Config.fromfile('../external/Metric3D/mono/configs/HourglassDecoder/vit.raft5.giant2.py')
        model = get_configured_monodepth_model(cfg, )
        model, _, _, _ = load_ckpt('../external/checkpoints/metric_depth_vit_giant2_800k.pth', model, strict_match=False)
    else:
        raise "Not implemented model"
    model.eval()
    model.to(device)

    cv_image = np.array(img)
    img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    if c_xy is None:
        c_xy = (img.shape[1] / 2, img.shape[0] / 2)
    intrinsic = [fx, fy, c_xy[0], c_xy[1]]
    rgb_input, cam_models_stacks, pad, label_scale_factor = transform_test_data_scalecano(img, intrinsic,
                                                                                          cfg.data_basic)

    with torch.no_grad():
        pred_depth, pred_depth_scale, scale, output, confidence = get_prediction(
            model=model,
            input=rgb_input,
            cam_model=cam_models_stacks,
            pad_info=pad,
            scale_info=label_scale_factor,
            gt_depth=None,
            normalize_scale=cfg.data_basic.depth_range[1],
            ori_shape=[img.shape[0], img.shape[1]],
        )

        pred_normal = output['normal_out_list'][0][:, :3, :, :]
        H, W = pred_normal.shape[2:]
        pred_normal = pred_normal[:, :, pad[0]:H - pad[1], pad[2]:W - pad[3]]

    pred_depth = pred_depth.squeeze().cpu().numpy()
    pred_depth[pred_depth < 0] = 0

    pred_normal = torch.nn.functional.interpolate(pred_normal, [img.shape[0], img.shape[1]], mode='bilinear').squeeze()
    pred_normal = pred_normal.permute(1, 2, 0)
    pred_normal = pred_normal.cpu().numpy()

    return pred_depth, pred_normal
