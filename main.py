from PIL import Image
from utils.flownet_utils import flow_arr_to_img, get_flow
from utils.utils import load_final_image, photometric_error
import numpy as np
from habitatenv import HabitatEnv
from config import DEFAULT_FINAL_IMG_PATH, OUTPUT_PATH, sim_settings


def main():
    final_img = load_final_image(DEFAULT_FINAL_IMG_PATH)
    sim = HabitatEnv(sim_settings, [-0.6, 0, 0, 1, 0, 0, 0])

    frame_cnt = 0
    sim.save_color_obs('init', OUTPUT_PATH + "frame%05d.png" % frame_cnt)

    velocities = np.identity(6)*10
    for vel in velocities:
        sim.step_agent(vel)
        frame_cnt += 1
        sim.save_color_obs(
            caption=f'frame: {frame_cnt}, vel:{vel}', path=OUTPUT_PATH + "frame%05d.png" % frame_cnt)

        err = photometric_error(sim.color_obs_img, final_img)
        print(f"Photometric Error:\t\t {err}")

        if final_img is not None:
            flow = get_flow(sim.color_obs_img, final_img)
            flow_arr_to_img(flow, OUTPUT_PATH + "flow%05d.png" % frame_cnt)

    Image.fromarray(sim.color_obs_img).save(DEFAULT_FINAL_IMG_PATH)


if __name__ == '__main__':
    main()
