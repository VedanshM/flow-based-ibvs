from utils.flownet_utils import get_flow
import numpy as np
from PIL import Image
import torch


class Dfvs:
    def __init__(self, des_img_path, LM_LAMBDA=0.001, LM_MU=0.03) -> None:
        self.des_img = np.array(Image.open(des_img_path))
        self.LM_LAMBDA = LM_LAMBDA
        self.LM_MU = LM_MU
        self.set_interaction_utils()

    @property
    def img_shape(self):
        return self.des_img.shape

    def get_next_velocity(self, cur_img, prev_img=None, depth=None) -> np.ndarray:
        assert (not prev_img) or (not depth)

        flow_error = get_flow(cur_img, self.des_img)
        if depth:
            L = self.get_interaction_mat(depth, False)
        else:
            flow_inv_depth = get_flow(cur_img, prev_img)
            L = self.get_interaction_mat(flow_inv_depth, True)

        H = L.T @ L
        vel = -self.LM_LAMBDA * (H + self.LM_MU*(H.diagonal())
                                   ).pinverse() @ L.T @ flow_error

        return vel.cpu().numpy()

    def set_interaction_utils(self):
        row_cnt, col_cnt = self.img_shape
        # v~y , u~x
        px = col_cnt // 2
        u_0 = col_cnt // 2
        py = row_cnt // 2
        v_0 = row_cnt // 2
        v, u = np.fromfunction(lambda i, j: (i, j), self.img_shape)
        x = (u - u_0)/px
        y = (v - v_0)/py

        x = torch.Tensor(x.flatten()).cuda()
        y = torch.Tensor(y.flatten()).cuda()

        self.inter_utils = {
            "x": x,
            "y": y,
            "zero": torch.zeros_like(x)
        }

    def get_interaction_mat(self, Z, inverse=True):
        Zi = (Z if inverse else 1/Z).flatten()
        x = self.inter_utils["x"]
        y = self.inter_utils["y"]
        zero = self.inter_utils["zero"]

        L = torch.stack([
            [-Zi, zero, x*Zi, x*y, -(1 + x**2), y],
            [zero, -Zi, y*Zi, 1 + y**2, -x*y, -x],
        ])

        assert L.shape == (2, 6, x.shape[0])

        L = L.permute(2, 0, 1).view(-1, 6)

        return L
