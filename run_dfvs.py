from dfvs import Dfvs
from utils.utils import photometric_error
from os.path import join as pathjoin
from habitatenv import HabitatEnv
from config import DEST_IMG_PATH, ERROR_THRESH, MAX_STEPS, RESULTS_PATH, sim_settings


def main():
    sim = HabitatEnv()
    dfvs = Dfvs(DEST_IMG_PATH)

    step_cnt = 0
    sim.save_color_obs(RESULTS_PATH + "frame_%05d.png" % step_cnt)
    prev_img = sim.obs_rgb
    perror = photometric_error(prev_img, dfvs.des_img)
    print("Init error: ", perror)

    while perror > ERROR_THRESH and step_cnt < MAX_STEPS:
        vel = dfvs.get_next_velocity(sim.obs_rgb, prev_img=prev_img)
        sim.step_agent(vel)

        step_cnt += 1
        curr_img = sim.obs_rgb
        perror = photometric_error(curr_img, dfvs.des_img)
        print(f"step : {step_cnt}")
        print(f"error: {perror}")
        print(f"state: {sim.agent_state}")

        sim.save_color_obs(path=pathjoin(
            RESULTS_PATH, f"frame_%05d.png" % step_cnt))


if __name__ == '__main__':
    main()
