DATA_PATH = './data/'
RESULTS_PATH = './results/'
TEST_SCENE = DATA_PATH + "test/Test.glb"
DEST_IMG_PATH = DATA_PATH + 'final.png'
ERROR_THRESH = 500
MAX_STEPS = 5000
IBVS_LAMBDA = 0.01
IBVS_MU = 0.03

sim_settings = {
    "scene_id": TEST_SCENE,        # Scene path
    "default_agent": 0,         # Index of the default agent
    "sensor_height": 1.5,       # Height of sensors in meters, relative to the agent
    "width": 512,               # Spatial resolution of the observations
    "height": 384,
}
