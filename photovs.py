from PIL import Image
import numpy as np


class PhotoVS:
    def __init__(self, des_img_path,
                 Z=1, steps_thresh=9000,
                 LM_LAMBDA=0.1, LM_MU=0.01,
                 GN_LAMBDA=0.1, GN_MU=0.0001,
                 ):
        self.des_img = np.array(Image.open(des_img_path).convert('RGB'))
        self.Id = PhotoVS.rgb_to_I(self.des_img)
        self.LM_LAMBDA = LM_LAMBDA
        self.LM_MU = LM_MU
        self.GN_LAMBDA = GN_LAMBDA
        self.GN_MU = GN_MU
        self.steps_thresh = steps_thresh
        self.set_interaction_matrices(Z)

    def get_next_velocity(self, steps, cur_img):
        I = PhotoVS.rgb_to_I(cur_img)
        error = (I - self.Id).flatten()
        if steps <= self.steps_thresh:
            mat = self.matrices["LM_mat"]
        else:
            mat = self.matrices["GN_mat"]

        vel = mat @ error
        return vel

    @property
    def img_shape(self):
        return self.des_img.shape[:2]

    def set_interaction_matrices(self, Z):
        row_cnt, col_cnt = self.img_shape
        Zi = 1/Z
        # v~y , u~x
        px = col_cnt // 2
        u_0 = col_cnt // 2
        py = row_cnt // 2
        v_0 = row_cnt // 2
        v, u = np.fromfunction(lambda i, j: (i, j), self.img_shape)
        x = (u - u_0)/px
        y = (v - v_0)/py
        x = x.flatten()
        y = y.flatten()

        Iy, Ix = np.gradient(self.Id)
        Ix = Ix.flatten()
        Iy = Iy.flatten()

        L = np.array([
            Ix*Zi, Iy*Zi,
            -Zi*(x*Ix + y*Iy), -(Ix*x*y + (1+y**2)*Iy),
            (1+x*x)*Ix + Iy*x*y, Iy*x-Ix*y
        ])
        L = L.T

        assert L.shape == (x.shape[0], 6)

        H = L.T @ L

        LM_mat = - self.LM_LAMBDA * np.linalg.pinv(
            H + self.LM_MU*H
        ) @ L.T

        GN_mat = - self.GN_LAMBDA * np.linalg.pinv(
            H + self.GN_MU*H
        ) @ L.T

        self.matrices = {
            "LM_mat": LM_mat,
            "GN_mat": GN_mat,
        }

    @staticmethod
    def rgb_to_I(rgb):
        return np.array(Image.fromarray(rgb).convert('L'))
