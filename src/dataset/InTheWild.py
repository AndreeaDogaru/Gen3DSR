from PIL import Image
import numpy as np
from dataset.BaseScene import BaseScene
from typing import Optional, List


class InTheWild(BaseScene):
    def __init__(self, img_path, K: Optional[List]=None):
        super().__init__(Image.open(img_path))
        if K is not None:
            K_np = np.array(K)
            if K_np.shape == (3, 3):
                self.K = K_np
            else:
                print(f"Ignoring input K as shape {K_np} does not match expected (3, 3)")

