from photovs import PhotoVS
from utils.utils import photometric_error
from os.path import join as pathjoin
from habitatenv import HabitatEnv
from config import DEST_IMG_PATH, ERROR_THRESH, MAX_STEPS, RESULTS_PATH, sim_settings


def main():
    sim = HabitatEnv(sim_settings, None)
    photovs = PhotoVS(DEST_IMG_PATH, Z=1)

    step_cnt = 0
    sim.save_color_obs(RESULTS_PATH + "frame_%05d.png" % step_cnt)
    prev_img = sim.color_obs_img
    perror = photometric_error(prev_img, photovs.des_img)
    print("Init error: ", perror)

    while perror > ERROR_THRESH and step_cnt < MAX_STEPS:
        vel = photovs.get_next_velocity(step_cnt, sim.color_obs_img)
        sim.step_agent(vel)

        step_cnt += 1
        prev_img = sim.color_obs_img
        perror = photometric_error(sim.color_obs_img, photovs.des_img)
        print(f"step : {step_cnt}")
        print(f"error: {perror}")
        print(f"state: {sim.agent_state}")

        sim.save_color_obs(path=pathjoin(
            RESULTS_PATH, f"frame_%05d.png" % step_cnt))


if __name__ == '__main__':
    main()
