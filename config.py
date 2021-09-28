DATA_PATH = './data/'
RESULTS_PATH = './results/'
TEST_SCENE =  DATA_PATH+ "mp3d_example/17DRP5sb8fy/17DRP5sb8fy.glb"
DEST_IMG_PATH = DATA_PATH + 'final.png'

sim_settings = {
    "scene_id": TEST_SCENE,        # Scene path
    "default_agent": 0,         # Index of the default agent
    "sensor_height": 1.5,       # Height of sensors in meters, relative to the agent
    "width": 512,               # Spatial resolution of the observations
    "height": 512,
}
