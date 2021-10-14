from utils.flownet_utils import get_flow
import numpy as np
from PIL import Image


class Dfvs:
    def __init__(self, des_img_path, LM_LAMBDA=0.01, LM_MU=0.03) -> None:
        self.des_img = np.array(Image.open(des_img_path).convert("RGB"))
        self.LM_LAMBDA = LM_LAMBDA
        self.LM_MU = LM_MU
        self.set_interaction_utils()

    @property
    def img_shape(self):
        return self.des_img.shape[:2]

    def get_next_velocity(self, cur_img, prev_img=None, depth=None) -> np.ndarray:
        assert (not prev_img) or (not depth)

        flow_error = get_flow(self.des_img, cur_img).transpose(1,0,2).flatten()
        if depth is not None:
            L = self.get_interaction_mat(depth, False)
        else:
            flow_inv_depth = get_flow(cur_img, prev_img)
            L = self.get_interaction_mat(flow_inv_depth, True)

        H = L.T @ L
        vel = -self.LM_LAMBDA * np.linalg.pinv(
            H + self.LM_MU*(H.diagonal())) @ L.T @ flow_error

        print("FLOW ERROR: ", np.sum((flow_error)))
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

    def get_interaction_mat(self, Z, inverted=True):
        nx, ny = 512, 384
        KK = np.array([[nx/2,0,nx/2],[0,ny/2, ny/2 ],[ 0, 0, 1]])
        px, py = KK[0, 0], KK[1, 1]
        v_0, u_0 = KK[0, 2], KK[1, 2]
        s = vik.copy()
        L = np.zeros((nx*ny*2, 6))

        for m in range(0, nx*ny*2 - 2, 2):
            x = (int(s[m]) - int(u_0))/px
            y = (int(s[m+1]) - int(v_0))/py
            t = int(s[m])
            u = int(s[m+1])
            Zinv = 10/(Z[u, t] + 1)
            
            
            L[m, 0] = -Zinv
            L[m,1]=0
            L[m,2]=x*Zinv
            L[m,3]=x*y
            L[m,4]=-(1+x**2)
            L[m,5]=y
            
            L[m+1,0]= 0
            L[m+1,1]= -Zinv
            L[m+1,2]= y*Zinv
            L[m+1,3]= 1+y**2
            L[m+1,4]= -x*y
            L[m+1,5]= -x

        return L

nx, ny = (512,384)
vik=np.array([])
for i in range(nx):
    for j in range(ny):
        vik=np.hstack((vik,np.array([i,j])))
