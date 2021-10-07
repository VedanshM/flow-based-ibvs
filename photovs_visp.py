from PIL import Image
import numpy as np
import cv2


class PhotoVS:
    def __init__(self, des_img_path,
                 Z=1, steps_thresh=9000,
                 LM_LAMBDA=3, LM_MU=0.01,
                 GN_LAMBDA=30, GN_MU=0.0001,
                 ):
        self.des_img = np.array(Image.open(des_img_path))
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
        return self.des_img.shape

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

        Iy = []
        Ix = []
        I = self.Id

        for i in range(row_cnt):
            for j in range(col_cnt):
                ix = px*((2047.0*(I[i][j+1] - I[i][j-1]) +
                          913.0*(I[i][j+2] - I[i][j-2]) +
                          112.0*(I[i][j+3] - I[i][j-3]))/8418.0)
                iy = py*((2047.0*(I[i+1][j] - I[i-1][j]) +
                          913.0*(I[i+2][j] - I[i-2][j]) +
                          112.0*(I[i+3][j] - I[i-3][j]))/8418.0)
                Ix.append(ix)
                Iy.append(iy)

        L = np.zeros((x.shape[0], 6))
        for m in range(x.shape[0]):
            L[m][0] = Ix*Zi
            L[m][1] = Iy*Zi
            L[m][2] = -(x*Ix+y*Iy)*Zi
            L[m][3] = -Ix*x*y-(1+y*y)*Iy
            L[m][4] = (1+x*x)*Ix + Iy*x*y
            L[m][5] = Iy*x-Ix*y

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
        bgr = rgb[:, :, ::-1]
        return cv2.cvtColor(bgr, cv2.COLOR_RGB2GRAY)
