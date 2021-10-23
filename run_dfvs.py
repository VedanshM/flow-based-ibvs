import os
from dfvs import Dfvs
from utils.utils import photometric_error
from time import time
import sys
from os.path import join as pathjoin
from habitatenv import HabitatEnv
from config import (DEST_IMG_PATH, ERROR_THRESH, MAX_STEPS,
                    sim_settings, RESULTS_PATH, LOGS_PATH)


def main():
    if len(sys.argv) > 1:
        global DEST_IMG_PATH, RESULTS_PATH, LOGS_PATH
        folder = sys.argv[1]
        sim_settings["scene_id"] = pathjoin(
            folder, os.path.basename(folder).capitalize() + ".glb")
        DEST_IMG_PATH = pathjoin(folder, "des.png")
        RESULTS_PATH = pathjoin(folder, "results_dfvs")
        LOGS_PATH = pathjoin(folder, "logs_dfvs")

    os.makedirs(RESULTS_PATH, exist_ok=True)
    os.makedirs(LOGS_PATH, exist_ok=True)
    p_err_log_f = open(pathjoin(LOGS_PATH, "p_err.txt"), "w+")
    sim = HabitatEnv(sim_settings)
    dfvs = Dfvs(DEST_IMG_PATH)

    steps = 0
    sim.save_color_obs(pathjoin(RESULTS_PATH, "frame_%05d.png" % steps))
    prev_img = sim.obs_rgb
    photo_err = photometric_error(prev_img, dfvs.des_img)
    print("Init error: ", photo_err)

    while photo_err > ERROR_THRESH and steps < MAX_STEPS:
        stime = time()
        vel = dfvs.get_next_velocity(sim.obs_rgb,
                                    #  depth=sim.obs_depth,       # for true depth
                                     prev_img=prev_img,       # for flow depth
                                     )
        print("TIME: ", time()-stime)
        sim.step_agent(vel, steps_per_sec=100)

        steps += 1
        prev_img = sim.obs_rgb
        photo_err = photometric_error(sim.obs_rgb, dfvs.des_img)

        print(f"step : {steps}")
        print(f"vel: ", vel.round(11))
        print(f"state: {sim.agent_state_compact}")
        print(f"photo metric error: {photo_err}")

        sim.save_color_obs(pathjoin(RESULTS_PATH, f"frame_%05d.png" % steps))
        p_err_log_f.write(str(photo_err) + "\n")
        p_err_log_f.flush()


if __name__ == '__main__':
    main()
