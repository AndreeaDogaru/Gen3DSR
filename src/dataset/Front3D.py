from pathlib import Path

import numpy as np
from PIL import Image
import json
from dataset.BaseScene import BaseScene


class Front3D(BaseScene):
    def __init__(self, data_root, render_task):
        render_task = f"{render_task:0>6}"
        data_root = Path(data_root)

        super().__init__(Image.open(data_root / "rgb" / f"rgb_{render_task}.jpeg"))

        with open(data_root / "annotation" / f"annotation_{render_task}.json") as f:
            annotation = json.load(f)
        self.depth_map = np.load(data_root / "depth" / f"depth_{render_task}.npy")
        self.depth_mask = self.depth_map < 10000    #invalid depth areas (e.g., windows) have a large depth value
        self.depth_map = self.depth_map.clip(0, 15)

        self.K = np.array(annotation['camera_intrinsics'])
        w2c = np.eye(4)
        w2c[:3] = np.array(annotation['camera_extrinsics'])
        self.c2w = np.linalg.inv(w2c)
        self.c2w[:3, [0, 3]] *= -1

