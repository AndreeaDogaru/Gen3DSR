import cv2
import numpy as np
from PIL import Image
from skimage.morphology import binary_erosion, square

import torch
import torch.nn.functional as F

# mm libs
from mmdet.registry import MODELS
from mmengine import Config, print_log
from mmengine.structures import InstanceData

from ext.class_names.lvis_list import LVIS_CLASSES

LVIS_NAMES = LVIS_CLASSES


class IMGState:
    def __init__(self):
        self.img = None
        self.img_feat = None
        self.selected_points = []
        self.selected_points_labels = []
        self.selected_bboxes = []
        self.img_mask = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.available_to_set = True

    def set_img(self, img, img_feat):
        self.img = img
        self.img_feat = img_feat

        self.available_to_set = False

    def clear(self):
        self.img = None
        self.img_feat = None
        self.selected_points = []
        self.selected_points_labels = []
        self.selected_bboxes = []

        self.available_to_set = True

    def clean(self):
        self.selected_points = []
        self.selected_points_labels = []
        self.selected_bboxes = []

    def to_device(self):
        if self.img_feat is not None:
            for k in self.img_feat:
                if isinstance(self.img_feat[k], torch.Tensor):
                    self.img_feat[k] = self.img_feat[k].to(self.device)
                elif isinstance(self.img_feat[k], tuple):
                    self.img_feat[k] = tuple(v.to(self.device) for v in self.img_feat[k])

    @property
    def available(self):
        return self.available_to_set


class ImageTagger:

    def __init__(self) -> None:
        model_cfg = Config.fromfile('../external/ovsam/app/configs/sam_r50x16_fpn.py')

        self.model = MODELS.build(model_cfg.model)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device=self.device)
        self.model = self.model.eval()
        self.model.init_weights()

        self.mean = torch.tensor([123.675, 116.28, 103.53], device=self.device)[:, None, None]
        self.std = torch.tensor([58.395, 57.12, 57.375], device=self.device)[:, None, None]
        self.IMG_SIZE = 1024
        self.img_state = IMGState()

    def segment_with_points(self,
                            img_state,
                            ):
        if img_state.available:
            return None, None, "State Error, please try again."
        output_img = img_state.img
        h, w = output_img.shape[:2]

        input_points = torch.tensor(img_state.selected_points, dtype=torch.float32, device=self.device)
        prompts = InstanceData(
            point_coords=input_points[None],
        )

        try:
            img_state.to_device()
            masks, cls_pred = self.model.extract_masks(img_state.img_feat, prompts)
            img_state.to_device()

            masks = masks[0, 0, :h, :w]
            masks = masks > 0.5

            cls_pred = cls_pred[0][0]
            scores, indices = torch.topk(cls_pred, 1)
            scores, indices = scores.tolist(), indices.tolist()
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                img_state.clear()
                print_log(f"CUDA OOM! please try again later", logger='current')
                return None, None, "CUDA OOM, please try again later."
            else:
                raise
        names = []
        for ind in indices:
            names.append(LVIS_NAMES[ind].replace('_', ' '))

        return names, scores

    def extract_img_feat(self, img, img_state):
        w, h = img.size
        scale = self.IMG_SIZE / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
        img_numpy = np.array(img)
        print_log(f"Successfully loaded an image with size {new_w} x {new_h}", logger='current')

        try:
            img_tensor = torch.tensor(img_numpy, device=self.device, dtype=torch.float32).permute((2, 0, 1))[None]
            img_tensor = (img_tensor - self.mean) / self.std
            img_tensor = F.pad(img_tensor, (0, self.IMG_SIZE - new_w, 0, self.IMG_SIZE - new_h), 'constant', 0)
            feat_dict = self.model.extract_feat(img_tensor)
            img_state.set_img(img_numpy, feat_dict)
            img_state.to_device()
            print_log(f"Successfully generated the image feats.", logger='current')
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                img_state.clear()
                print_log(f"CUDA OOM! please try again later", logger='current')
                return None, None, "CUDA OOM, please try again later."
            else:
                raise
        return img, None, "Please try to click something."

    def infer(self, image, masks, erosion_rate=0.05, max_samples=10):

        self.extract_img_feat(image, self.img_state)
        tags = []
        scores = []

        for mask in masks:
            self.img_state.clean()
            mask_image = Image.fromarray(mask)
            w, h = image.size
            scale = self.IMG_SIZE / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            mask_image = mask_image.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
            mask_image = np.array(mask_image)
            # erode the mask erosion_rate% on the smallest dimension
            _, _, mw, mh = cv2.boundingRect(np.uint8(mask_image))
            erosion_amount = max(min(int(erosion_rate * min(mw, mh)), 15), 3)
            # print('erosion amount', erosion_amount)
            structuring_element = square(erosion_amount)
            eroded_mask = binary_erosion(mask_image, structuring_element)
            if not eroded_mask.sum():
                # no points after erosion, return to original mask
                eroded_mask = mask_image
            y, x = np.where(eroded_mask)
            num_samples = min(max_samples, len(x))
            indices = np.random.choice(x.size, size=num_samples, replace=False)
            self.img_state.selected_points.extend(np.stack([x[indices], y[indices]], axis=1).tolist())

            class_name, score = self.segment_with_points(self.img_state)
            tags.append(class_name[0])
            scores.append(score[0])

        return tags, scores
