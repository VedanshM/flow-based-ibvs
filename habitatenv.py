import habitat_sim
from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np
from config import sim_settings


class HabitatEnv:
    def __init__(self, settings=sim_settings):
        self.make_sim(settings)
        self.cur_obs = self.sim.get_sensor_observations()
        self.imgshape = nx, ny = 384, 512
        self._cam_prop = np.array([
            [nx/2, 0, nx/2],
            [0, ny/2, ny/2],
            [0, 0, 1]
        ])

    @property
    def cam_prop(self) -> np.ndarray:
        return self._cam_prop[:]

    @property
    def obs_rgb(self):
        return self.cur_obs["color_sensor"][:, :, :3]

    @property
    def obs_depth(self):
        return self.cur_obs["depth_sensor"]

    def make_sim(self, settings: dict):
        # simulator backend
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = settings["scene_id"]

        # agent
        agent_cfg = habitat_sim.agent.AgentConfiguration()

        sensor_specs = []

        def create_camera_spec(**kw_args):
            camera_sensor_spec = habitat_sim.CameraSensorSpec()
            camera_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
            camera_sensor_spec.resolution = [
                settings["height"], settings["width"]]
            camera_sensor_spec.position = [0, settings["sensor_height"], 0]
            for k in kw_args:
                setattr(camera_sensor_spec, k, kw_args[k])
            return camera_sensor_spec

        # Note: all sensors must have the same resolution
        if settings["color_sensor"]:
            color_sensor_spec = create_camera_spec(
                uuid="color_sensor",
                hfov=settings["hfov"],
                sensor_type=habitat_sim.SensorType.COLOR,
                sensor_subtype=habitat_sim.SensorSubType.PINHOLE,
            )
            sensor_specs.append(color_sensor_spec)

        if settings["depth_sensor"]:
            depth_sensor_spec = create_camera_spec(
                uuid="depth_sensor",
                hfov=settings["hfov"],
                sensor_type=habitat_sim.SensorType.DEPTH,
                channels=1,
                sensor_subtype=habitat_sim.SensorSubType.PINHOLE,
            )
            sensor_specs.append(depth_sensor_spec)

        if settings["semantic_sensor"]:
            semantic_sensor_spec = create_camera_spec(
                uuid="semantic_sensor",
                hfov=settings["hfov"],
                sensor_type=habitat_sim.SensorType.SEMANTIC,
                channels=1,
                sensor_subtype=habitat_sim.SensorSubType.PINHOLE,
            )
            sensor_specs.append(semantic_sensor_spec)

        agent_cfg.sensor_specifications = sensor_specs
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(0.)
            ),
            "move_backward": habitat_sim.agent.ActionSpec(
                "move_backward", habitat_sim.agent.ActuationSpec(0.)
            ),
            "move_down": habitat_sim.agent.ActionSpec(
                "move_down", habitat_sim.agent.ActuationSpec(0.)
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(0.)
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(0.)
            ),
            "look_up": habitat_sim.agent.ActionSpec(
                "look_up", habitat_sim.agent.ActuationSpec(0.)
            ),
            "look_down": habitat_sim.agent.ActionSpec(
                "look_down", habitat_sim.agent.ActuationSpec(0.)
            ),
            "look_left": habitat_sim.agent.ActionSpec(
                "look_left", habitat_sim.agent.ActuationSpec(0.)
            ),
            "look_right": habitat_sim.agent.ActionSpec(
                "look_right", habitat_sim.agent.ActuationSpec(0.)
            ),
            "look_anti": habitat_sim.agent.ActionSpec(
                "rotate_sensor_anti_clockwise", habitat_sim.agent.ActuationSpec(
                    0.)
            ),
            "look_clock": habitat_sim.agent.ActionSpec(
                "rotate_sensor_clockwise", habitat_sim.agent.ActuationSpec(0.)
            ),
            "move_down": habitat_sim.agent.ActionSpec(
                "move_down", habitat_sim.agent.ActuationSpec(0.)
            ),
            "move_up": habitat_sim.agent.ActionSpec(
                "move_up", habitat_sim.agent.ActuationSpec(0.)
            ),
            "move_left": habitat_sim.agent.ActionSpec(
                "move_left", habitat_sim.agent.ActuationSpec(0.)
            ),
            "move_right": habitat_sim.agent.ActionSpec(
                "move_right", habitat_sim.agent.ActuationSpec(0.)
            ),
        }

        config = habitat_sim.Configuration(sim_cfg, [agent_cfg])
        self.sim = habitat_sim.Simulator(config)
        self.sim.initialize_agent(settings["default_agent"])

    def step_agent(self, velocity, FPS=100):
        v = velocity * [1, -1, -1, 1, -1, -1]/FPS
        v[3:] = np.rad2deg(v[3:])
        self.sim.config.agents[0].action_space['move_right'].actuation.amount = v[0]
        self.sim.config.agents[0].action_space['move_up'].actuation.amount = v[1]
        self.sim.config.agents[0].action_space['move_backward'].actuation.amount = v[2]
        self.sim.config.agents[0].action_space['look_left'].actuation.amount = v[4]
        self.sim.config.agents[0].action_space['look_up'].actuation.amount = v[3]
        self.sim.config.agents[0].action_space['look_anti'].actuation.amount = v[5]

        self.sim.step("move_right")
        self.sim.step("move_backward")
        self.sim.step("move_up")
        self.sim.step("look_up")
        self.sim.step("look_left")
        self.cur_obs = self.sim.step("look_anti")
        return self.cur_obs

    def save_color_obs(self, path: str, caption: str = None, verbose: bool = True):
        rgb_img = Image.fromarray(self.cur_obs["color_sensor"], mode="RGBA")

        if caption is not None:
            img = ImageOps.expand(rgb_img, border=33, fill=(255, 255, 255))
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype("FreeSans.ttf", 30)
            draw.text((30, 0), caption, (0, 0, 0), font)
        else:
            img = rgb_img

        img.save(path)
        if verbose:
            print(f"{path} saved.")

    @property
    def agent_state(self):
        return self.sim._default_agent.state.sensor_states["color_sensor"]

    @property
    def agent_state_compact(self):
        state = self.agent_state
        ret = state.position.tolist()
        ret.append(state.rotation.real)
        ret += state.rotation.imag.tolist()
        ret = np.round(ret, 6)
        return ret


def testing():
    sim = HabitatEnv()
    print(sim.agent_state_compact)
    Image.fromarray(sim.cur_obs["color_sensor"], mode="RGBA").show()


if __name__ == '__main__':
    testing()
