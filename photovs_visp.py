from PIL import Image
import numpy as np
import cv2


class PhotoVS:
    def __init__(self, des_img_path,
                 Z=1, steps_thresh=9000,
                 LM_LAMBDA=0.1, LM_MU=0.01,
                 GN_LAMBDA=0.1, GN_MU=0.0001,
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

        Iy = []
        Ix = []
        I = self.Id

        for i in range(row_cnt):
            for j in range(col_cnt):
                ix = px*((2047.0*(I[i][min(j+1, col_cnt-1)] - I[i][max(j-1, 0)]) +
                          913.0*(I[i][min(j+2, col_cnt-1)] - I[i][max(j-2, 0)]) +
                          112.0*(I[i][min(j+3, col_cnt-1)] - I[i][max(j-3, 0)]))/8418.0)
                iy = py*((2047.0*(I[min(i+1, row_cnt-1)][j] - I[max(i-1, 0)][j]) +
                          913.0*(I[min(i+2, row_cnt-1)][j] - I[max(i-2, 0)][j]) +
                          112.0*(I[min(i+3, row_cnt-1)][j] - I[max(i-3, 0)][j]))/8418.0)
                Ix.append(ix)
                Iy.append(iy)
        Ix = np.array(Ix)
        Iy = np.array(Iy)

        L = np.zeros((x.shape[0], 6))
        for m in range(x.shape[0]):
            ix = Ix[m]
            iy = Iy[m]
            x_ = x[m]
            y_ = y[m]
            L[m][0] = ix*Zi
            L[m][1] = iy*Zi
            L[m][2] = -(x_*ix+y_*iy)*Zi
            L[m][3] = -ix*x_*y_-(1+y_*y_)*iy
            L[m][4] = (1+x_*x_)*ix + iy*x_*y_
            L[m][5] = iy*x_-ix*y_

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
