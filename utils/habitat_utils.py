import habitat_sim
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np


def generate_config(sim_settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = sim_settings["scene_id"]

    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    # In the 1st example, we attach only one sensor,
    # a RGB visual sensor, to the agent
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [
        sim_settings["height"], sim_settings["width"]]
    rgb_sensor_spec.position = [0.0, sim_settings["sensor_height"], 0.0]

    agent_cfg.sensor_specifications = [rgb_sensor_spec]

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


# def display_sample(rgba_obs: np.ndarray, frame_cnt=0, action: str = "", semantic_obs=np.array([]), depth_obs=np.array([])):

#     rgb_img = Image.fromarray(rgba_obs, mode="RGBA")

#     arr = [rgb_img]
#     titles = [f"rgb : frame {frame_cnt} : action: {action}"]
#     if semantic_obs.size != 0:
#         from habitat_sim.utils.common import d3_40_colors_rgb
#         semantic_img = Image.new(
#             "P", (semantic_obs.shape[1], semantic_obs.shape[0]))
#         semantic_img.putpalette(d3_40_colors_rgb.flatten())
#         semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
#         semantic_img = semantic_img.convert("RGBA")
#         arr.append(semantic_img)
#         titles.append("semantic")

#     if depth_obs.size != 0:
#         depth_img = Image.fromarray(
#             (depth_obs / 10 * 255).astype(np.uint8), mode="L")
#         arr.append(depth_img)
#         titles.append("depth")

#     plt.figure(figsize=(12, 8))
#     for i, data in enumerate(arr):
#         ax = plt.subplot(1, 3, i + 1)
#         ax.axis("off")
#         ax.set_title(titles[i])
#         plt.imshow(data)
#     file_name = get_file_name(frame_cnt)
#     plt.savefig(file_name)
#     print(f"{file_name} saved.")
#     plt.clf()

def save_obs(obs, caption:str, path:str):

    rgb_img = Image.fromarray(obs["color_sensor"], mode="RGBA")

    img = ImageOps.expand(rgb_img, border=33, fill=(255,255,255))

    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("FreeSans.ttf", 30)
    draw.text((30,0), caption, (0,0,0), font)

    img.save(path)
    print(f"{path} saved.")
