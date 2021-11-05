DATA_PATH = './data/'
RESULTS_PATH = './results/'
LOGS_PATH = "./logs"
TEST_SCENE = "./data/stokes/Stokes.glb"
DEST_IMG_PATH = "./data/stokes/des.png"
WEIGHTS_PATH = './data/FlowNet2_checkpoint.pth.tar'
FLOW_ERR_LOG_FILE = "./results/flow_err_log.txt"
PHOTO_ERR_LOG_FILE = "./results/photo_err_log.txt"
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
    "hfov": 90,
    "color_sensor": True,  # RGB sensor
    "semantic_sensor": False,  # Semantic sensor
    "depth_sensor": True,  # Depth sensor
    "seed": 1,
}
