from utils.habitat_utils import save_obs
import habitat_sim
from PIL import Image
import numpy as np

from utils.utils import save_obs, generate_config, load_final_image, photometric_error
from utils.flownet_utils import flow_arr_to_img, get_flow
from config import *


ACTIONS_LIST = "LFFFRRRR"

def initialize():
    global sim, action_names, expand_actionname, frame_cnt, initial_img

    frame_cnt = 0
    cfg = generate_config(sim_settings)
    sim = habitat_sim.Simulator(cfg)

    # initialize an agent
    # Set agent state
    agent_state = habitat_sim.AgentState()
    agent_state.position = np.array([-0.6, 0.0, 0.0])  # in world space
    sim.initialize_agent(sim_settings["default_agent"], agent_state)
    action_names = list(
        cfg.agents[sim_settings["default_agent"]].action_space.keys())

    expand_actionname = {
        'R': "turn_right",
        'L': "turn_left",
        'F': "move_forward",
    }
    initial_img = sim.get_sensor_observations()['color_sensor']


def navigate_and_see(action: str = ""):
    global frame_cnt
    assert action in action_names, f"Invalid action: {action}"
    print(f"Taking action:{action}")
    observations = sim.step(action)
    save_obs(observations, f"frame: {frame_cnt}, action: {action}", OUTPUT_PATH+"frame%05d.png"%frame_cnt)
    # display_sample(obs_img, frame_cnt, action)
    frame_cnt += 1
    return observations["color_sensor"][:, :, :3]


def main():
    initialize()
    # display_sample(initial_img)
    final_img = load_final_image()

    for act in map(lambda x: expand_actionname[x], ACTIONS_LIST):
        obs_img = navigate_and_see(act)
        err = photometric_error(final_img, obs_img)
        print(f"Photometric Error:\t\t {err}")

        if final_img is not None:
            flow = get_flow(obs_img, final_img)
            flow_arr_to_img(flow, OUTPUT_PATH +  "flow%05d.png"%frame_cnt)

    Image.fromarray(obs_img).save(DATA_PATH + "final.png")


if __name__ == "__main__":
    main()
