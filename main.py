from utils.flownet_utils import get_flow
from utils.utils import load_final_image, photometric_error
from PIL import Image
from os.path import join as pathjoin
import numpy as np
import numpy.linalg as LA
from habitatenv import HabitatEnv
from config import DEST_IMG_PATH, ERROR_THRESH, IBVS_LAMBDA, IBVS_MU, MAX_STEPS, RESULTS_PATH, sim_settings    


def get_interaction_mat(cam_prop, depth_inv, imgshape):
    px = cam_prop[0,0]
    py = cam_prop[1,1]
    u_0 = cam_prop[1,2]
    v_0 = cam_prop[0,2]

    X,Y = np.fromfunction(lambda i,j: [i,j], imgshape)
    X = X.flatten()
    Y = Y.flatten()
    x = (X - u_0)/px
    y = (Y - v_0)/py
    Zi = depth_inv.flatten() + 1
    zero = np.zeros_like(x)

    print(x.shape, y.shape, Zi.shape, zero.shape)
    L = np.array([
        [-Zi, zero ,x*Zi, x*y, -(1+ x**2), y],
        [zero, -Zi, y*Zi, 1 + y**2, -x*y, -x],
    ])

    L = L.transpose(2,0,1).reshape(-1, 6)

    return L


def get_next_velocity(prev_img: np.ndarray, dest_img: np.ndarray, sim:HabitatEnv):
    cur_img = sim.color_obs_img
    cam_prop = sim.cam_prop

    flow_err = get_flow(cur_img, dest_img)
    flowproxy = get_flow(prev_img, cur_img)
    print(prev_img.shape, cur_img.shape, flowproxy.shape)
    proxy_depth_inv = LA.norm(flowproxy, axis=2)

    L = get_interaction_mat(cam_prop, proxy_depth_inv, sim.imgshape)
    
    #using LM algo
    H = L.T @ L
    vel  = -IBVS_LAMBDA * LA.pinv(H + IBVS_MU*(H.diagonal())) @ L.T @ flow_err.flatten()

    return vel

    

def main():
    final_img = load_final_image(DEST_IMG_PATH)
    sim = HabitatEnv(sim_settings, 
                     [-1.792603,  1.6131673,  19.256025, 1, 0, 0, 0])


    step_cnt = 0
    sim.save_color_obs(RESULTS_PATH + "frame_%05d.png" % step_cnt)
    prev_img = sim.color_obs_img
    perror = photometric_error(prev_img, final_img)
    print("Init error: ", perror)

    while perror < ERROR_THRESH and step_cnt < MAX_STEPS:
        vel = get_next_velocity(prev_img, final_img, sim)
        sim.step_agent(vel)

        step_cnt += 1
        curr_img = sim.color_obs_img
        perror = photometric_error(curr_img, final_img)
        print(f"step : {step_cnt}")
        print(f"error: {perror}")
        print(f"state: {sim.agent_state}")

        sim.save_color_obs(path= pathjoin(RESULTS_PATH, f"frame_%05d.png"%step_cnt))




if __name__ == '__main__':
    main()
