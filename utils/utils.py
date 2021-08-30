from config import DATA_PATH
from utils.habitat_utils import *
from utils.flownet_utils import *

def photometric_error(img1: np.ndarray, img2: np.ndarray) -> float:
    if img1 is None or img2 is None or img1.shape != img2.shape:
        return None
    img1 = img1[:, :, :3]
    img2 = img2[:, :, :3]
    error = (img1 - img2)**2
    error = error/float(img1.shape[0]*img1.shape[1])
    return np.sum(error)


def load_final_image(path=DATA_PATH+ "final.png"):
    try:
        final_img = Image.open(path).convert("RGB")
    except FileNotFoundError:
        print(f"========OUTPUT FILE {path} NOT FOUND======")
        return None
    final_img:np.ndarray = np.array(final_img)
    return final_img
