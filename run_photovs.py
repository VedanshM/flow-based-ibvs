from photovs import PhotoVS
from utils.utils import photometric_error
from os.path import join as pathjoin
from habitatenv import HabitatEnv
from config import DEST_IMG_PATH, ERROR_THRESH, MAX_STEPS, PERR_LOG_FILE, RESULTS_PATH, sim_settings


def main():
    sim = HabitatEnv()
    photovs = PhotoVS(DEST_IMG_PATH, Z=1)
    err_log_f = open(PERR_LOG_FILE, "w+")

    steps = 0
    sim.save_color_obs(RESULTS_PATH + "frame_%05d.png" % steps)
    prev_img = sim.obs_rgb
    photo_err = photometric_error(prev_img, photovs.des_img)
    print("Init error: ", photo_err)

    while photo_err > ERROR_THRESH and steps < MAX_STEPS:
        vel = photovs.get_next_velocity(steps, sim.obs_rgb)
        sim.step_agent(vel)

        steps += 1
        prev_img = sim.obs_rgb
        photo_err = photometric_error(sim.obs_rgb, photovs.des_img)
        if steps % 10 == 0:
            print(f"step : {steps}")
            print(f"error: {photo_err}")
            print(f"vel: ", vel)
            print(f"state: {sim.agent_state_compact}")

        err_log_f.write(str(photo_err)+"\n")
        sim.save_color_obs(path=pathjoin(
            RESULTS_PATH, f"frame_%05d.png" % steps), verbose=False)
        err_log_f.flush()


if __name__ == '__main__':
    main()
