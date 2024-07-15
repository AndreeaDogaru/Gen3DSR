from PIL import Image
import numpy as np


class BaseScene:
    def __init__(self, image: Image):
        self.image_pil: Image = image.convert('RGB')
        self.image_np: np.ndarray = np.array(image)
        self.depth_map: np.ndarray = None
        self.depth_mask: np.ndarray = None
        self.K: np.ndarray = None
        self.c2w: np.ndarray = None

