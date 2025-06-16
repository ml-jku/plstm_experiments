import numpy as np
from .transform import Transform


class OneHotLabelTransform(Transform):
    def __init__(self, num_classes: int, *args):
        super().__init__()
        self.num_classes = num_classes

    def __call__(self, xcls):
        # checks
        assert isinstance(xcls, tuple) and len(xcls) == 2

        # setup
        x, cls = xcls

        if not isinstance(cls, np.ndarray):
            cls = np.array(cls)
        cls_onehot = np.eye(self.num_classes, dtype=np.float32)[cls]
        return x, cls_onehot
