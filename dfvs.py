from utils.flownet_utils import get_flow
import numpy as np
from PIL import Image


class Dfvs:
    TRUEDEPTH = 0
    FLOWDEPTH = 1

    def __init__(self, des_img_path, LM_LAMBDA=0.01, LM_MU=0.03, mode=0) -> None:
        self.mode = mode
        self.des_img = np.array(Image.open(des_img_path).convert("RGB"))
        self.LM_LAMBDA = LM_LAMBDA
        self.LM_MU = LM_MU
        self.set_interaction_utils()

    @property
    def img_shape(self):
        return self.des_img.shape[:2]

    def get_next_velocity(self, cur_img, prev_img=None, depth=None, err_log_f=None) -> np.ndarray:
        assert (not prev_img) or (not depth)

        flow_error = get_flow(
            self.des_img, cur_img).transpose(1, 0, 2).flatten()
        if depth is not None:
            L = self.get_interaction_mat(depth, Dfvs.TRUEDEPTH)
        else:
            flow_inv_depth = get_flow(cur_img, prev_img)
            L = self.get_interaction_mat(flow_inv_depth, Dfvs.FLOWDEPTH)

        H = L.T @ L
        vel = -self.LM_LAMBDA * np.linalg.pinv(
            H + self.LM_MU*(H.diagonal())) @ L.T @ flow_error

        flow_err = np.sum((flow_error))
        print("FLOW ERROR: ", flow_err)
        if err_log_f is not None:
            err_log_f.write(str(flow_err) + "\n")

        return vel

    def set_interaction_utils(self):
        row_cnt, col_cnt = self.img_shape
        # v~y , u~x
        px = col_cnt // 2
        u_0 = col_cnt // 2
        py = row_cnt // 2
        v_0 = row_cnt // 2
        u, v = np.indices((col_cnt, row_cnt))
        x = (u - u_0)/px
        y = (v - v_0)/py

        x = x.flatten()
        y = y.flatten()

        self.inter_utils = {
            "x": x,
            "y": y,
            "zero": np.zeros_like(x)
        }

    def get_interaction_mat(self, Z, mode=None):
        assert Z.shape == self.img_shape

        if mode is None:
            mode = self.mode
        if mode == self.TRUEDEPTH:
            Zi = 10/(Z + 1)
        elif mode == self.FLOWDEPTH:
            Zi = Z + 1
        Zi = Zi.T.flatten()

        x = self.inter_utils["x"]
        y = self.inter_utils["y"]
        zero = self.inter_utils["zero"]

        L = np.array([
            [-Zi, zero, x*Zi, x*y, -(1 + x**2), y],
            [zero, -Zi, y*Zi, 1 + y**2, -x*y, -x],
        ])

        assert L.shape == (2, 6, x.shape[0])
        L = L.transpose(2, 0, 1).reshape(-1, 6)

        return L
