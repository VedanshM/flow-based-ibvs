from dfvs import Dfvs
from utils.utils import photometric_error
from os.path import join as pathjoin
from habitatenv import HabitatEnv
from config import (DEST_IMG_PATH, ERROR_THRESH, MAX_STEPS,
                    FLOW_ERR_LOG_FILE, PHOTO_ERR_LOG_FILE, RESULTS_PATH)


def main():
    sim = HabitatEnv()
    dfvs = Dfvs(DEST_IMG_PATH, mode=Dfvs.TRUEDEPTH)         # for true depth
    dfvs = Dfvs(DEST_IMG_PATH, mode=Dfvs.FLOWDEPTH)         # for flow depth
    f_err_log_f = open(FLOW_ERR_LOG_FILE, "w+")
    p_err_log_f = open(PHOTO_ERR_LOG_FILE, "w+")

    steps = 0
    sim.save_color_obs(RESULTS_PATH + "frame_%05d.png" % steps)
    prev_img = sim.obs_rgb
    photo_err = photometric_error(prev_img, dfvs.des_img)
    print("Init error: ", photo_err)

    while photo_err > ERROR_THRESH and steps < MAX_STEPS:
        vel, flow_error = dfvs.get_next_velocity(sim.obs_rgb,
                                     depth=sim.obs_depth,       # for true depth
                                    #  prev_img=prev_img,       # for flow depth
                                     )
        sim.step_agent(vel, FPS=1)

        steps += 1
        prev_img = sim.obs_rgb
        photo_err = photometric_error(sim.obs_rgb, dfvs.des_img)

        print(f"step : {steps}")
        print(f"vel: ", vel.round(11))
        print(f"state: {sim.agent_state_compact}")
        print(f"photo metric error: {photo_err}")
        print(f"flow error:", flow_error)

        sim.save_color_obs(pathjoin(RESULTS_PATH, f"frame_%05d.png" % steps))
        p_err_log_f.write(str(photo_err) + "\n")
        f_err_log_f.write(str(flow_error) + "\n")
        p_err_log_f.flush()
        f_err_log_f.flush()


if __name__ == '__main__':
    main()
