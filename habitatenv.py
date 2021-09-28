import habitat_sim
from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np

class   HabitatEnv:
    def __init__(self, sim_settings, agent_init_state):
        self.make_sim(sim_settings)
        self.initialize_agent(agent_init_state)
        self.cur_obs = self.sim.get_sensor_observations()
        

    def initialize_agent(self, init_state):
        start_state = habitat_sim.agent.AgentState()
        x, y, z, w, p, q, r = init_state
        start_state.position = np.array([x, y, z]).astype('float32')
        start_state.rotation = np.quaternion(w, p, q, r)
        self.agent = self.sim.initialize_agent(
            self.sim._default_agent_id, start_state)
        
        agent_object_id = self.sim.add_object(1, 
            self.sim._default_agent.scene_node)
        self.sim.set_object_motion_type(
            habitat_sim.physics.MotionType.KINEMATIC, agent_object_id
        )
        self.agent_vel_controller = self.sim.get_object_velocity_control(agent_object_id)
        assert (
            self.sim.get_object_motion_type(agent_object_id)
            == habitat_sim.physics.MotionType.KINEMATIC
        )
    
    @property
    def color_obs_img(self):
        return self.cur_obs["color_sensor"][:, :, :3]

    def make_sim(self, sim_settings:dict):
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
        agent_cfg.action_space = {
            "move_right": habitat_sim.agent.ActionSpec("move_right", {"amount": 0}),
            "move_left": habitat_sim.agent.ActionSpec("move_left", {"amount": 0}),
            "move_up": habitat_sim.agent.ActionSpec("move_up", {"amount": 0}),

            "move_down": habitat_sim.agent.ActionSpec("move_down", {"amount": 0}),
            "move_forward": habitat_sim.agent.ActionSpec("move_forward", {"amount": 0}),
            "move_backward": habitat_sim.agent.ActionSpec("move_backward", {"amount": 0}),

            "look_left": habitat_sim.agent.ActionSpec("look_left", {"amount": 0}),
            "look_right": habitat_sim.agent.ActionSpec("look_right", {"amount": 0}),
            "look_up": habitat_sim.agent.ActionSpec("look_up", {"amount": 0}),
            "look_down": habitat_sim.agent.ActionSpec("look_down", {"amount": 0}),
            "look_anti": habitat_sim.agent.ActionSpec("look_anti", {"amount": 0}),
            "look_clock": habitat_sim.agent.ActionSpec("look_clock", {"amount": 0}),

        }

        config =  habitat_sim.Configuration(sim_cfg, [agent_cfg])
        self.sim = habitat_sim.Simulator(config)

    
    def step_agent(self,velocity):
        self.agent_vel_controller.linear_velocity = np.array(velocity[0:3])
        self.agent_vel_controller.angular_velocity = np.array(velocity[3:])
        self.agent_vel_controller.controlling_lin_vel = True
        self.agent_vel_controller.controlling_ang_vel = True
        # step with world time
        self.sim.step_physics(1/60)
        self.agent_vel_controller.lin_vel_is_local = True
        self.agent_vel_controller.ang_vel_is_local = True
        self.cur_obs = self.sim.get_sensor_observations()
        return self.cur_obs

    def save_color_obs(self, path: str, caption: str=None, verbose: bool = True):
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
